# Secure AI by Design
## Tasarımdan Üretime Güvenli Yapay Zekâ

**Makale serisi yayın planı — 12 bölüm**  
**Dil:** Türkçe; yerleşik teknik terimler gerektiğinde İngilizce karşılıklarıyla.  
**Hedef okur:** AI/ML mühendisleri, yazılım ve platform ekipleri, siber güvenlik uzmanları ve teknik karar vericiler.

## Serinin description metni

Secure AI by Design; yapay zekâ güvenliğini veri, model, uygulama, ajan ve operasyon katmanları boyunca ele alan bir makale serisidir. Tehdit modellemeden hassas veri korumaya, guardrail tasarımından red teaming ve olay müdahalesine kadar her bölüm, bir tasarım sorusunu somut bir kontrol ve doğrulanabilir çıktıyla ilişkilendirir. Seri, LLM ve ajan uygulamalarına ağırlık verirken AI yaşam döngüsündeki veri ve model güvenliğini de kapsar.

**Kısa description:** Yapay zekâ güvenliğini tasarımdan üretime taşıyan; veri, model, ajan ve operasyon kontrollerini pratik örneklerle açıklayan 12 bölümlük seri.

**Serinin ana fikri:** Her bölüm bir riski açıklar, tasarım kararını gösterir ve kontrolün nasıl doğrulanacağını tanımlar.

## Editoryal çerçeve

- Seri, okuyucunun her bölümde aynı sistemin bir sonraki katmanını kurmasını sağlayacak biçimde sıralanır.
- Ortak örnek: Müşteri kaydını getiren, şikâyeti özetleyen ve kullanıcı onayıyla e-posta gönderebilen kurumsal asistan. Örnek veriler sentetiktir; gerçek kurum içi sistemler veya sırlar kullanılmaz.
- Kapsam LLM/RAG/ajan ağırlıklıdır. Eğitim verisi, model edinimi ve klasik adversarial ML riskleri de ilgili bölümlerde ele alınır.
- Güvenlik, mahremiyet, emniyet ve kalite arasındaki bağlantılar açıklanır; bu başlıkların tamamı tek bir filtreye indirgenmez.
- Anlatı ürün bağımsızdır. MUHAFIZ istenirse uygulama örneği olabilir; kavramları anlamak için ön koşul değildir. Tasarım ile uygulanmış sonuç açıkça ayrılır.
- Her makale için önerilen yapı: gerçekçi problem → tehdit/güven sınırı → mimari kontrol → küçük örnek/test → sınırlılıklar → pratik çıktı.
- Ana gövde için önerilen uzunluk 1.200–1.800 kelimedir; kod ve ek tablolar ayrı tutulabilir. Bu bir editoryal tercihtir.

## Seri haritası

| Bölüm | Başlık | Ana içerik |
| --- | --- | --- |
| 1 | Yapay Zekâ Güvenliği Tasarımda Başlar | Varlıklar, güvenlik ilkeleri, sorumluluklar. |
| 2 | AI Sistemlerinde Tehdit Modelleme ve Güven Sınırları | Tehdit senaryoları, veri akışı, güven sınırları. |
| 3 | Güvenli AI Verisi: Kaynaktan Silmeye Kadar Koruma | PII, veri kökeni, poisoning, saklama ve silme. |
| 4 | Modelden Bağımlılığa: AI Tedarik Zincirini Güvenceye Almak | Model registry, bağımlılıklar, provenance, güvenli yayın. |
| 5 | Prompt Injection: Veri ile Talimat Arasındaki Güven Sınırı | Dolaylı injection, jailbreak, kaynak ve işlem sınırları. |
| 6 | Katmanlı Guardrails: Regex, NLP, Karar Modelleri ve SLM | Regex, NER, embedding, Jev, SLM, eşik ve politika. |
| 7 | RAG ve Bellek Güvenliği: Doğru Bilgiye Yetkili Erişim | RAG ACL, tenant izolasyonu, vektör verisi, bellek zehirleme. |
| 8 | AI Ajanlarında En Az Yetki ve Güvenli Araç Kullanımı | Araç yetkisi, MCP, onay, alt ajan ve bütçe. |
| 9 | Güvenli AI Sunumu: API Gateway'den Model Sunucusuna | Gateway, vLLM, API, ağ, streaming ve fallback. |
| 10 | AI Güvenliğini Ölçmek: Red Teaming ve Türkçe Değerlendirme | Red team, Türkçe testler, metrikler, bağımsız doğrulama. |
| 11 | AI Operasyonlarında İzlenebilirlik ve Olay Müdahalesi | Audit, trace, gizlilik, SOC, müdahale ve rollback. |
| 12 | Standarttan Kanıta: AI Güvenliğini Denetlenebilir Hale Getirmek | Standart eşleştirme, risk kabulü, kanıt ve yayın kapısı. |

## 1. Yapay Zekâ Güvenliği Tasarımda Başlar

**Description**  
Yapay zekâ güvenliğini veri, model, uygulama ve operasyon katmanları üzerinden ele alarak güvenli tasarımın temel ilkelerini ve kurum içindeki sorumlulukları açıklıyoruz.

**Bölümün temel sorusu:** Sistemimizde neyi koruyoruz ve bunun sahibi kim?

**İçerik akışı**

1. Korunacak varlıklar: veri, model, işlem yetkileri ve hizmet sürekliliği.
2. Gizlilik, bütünlük, erişilebilirlik ve güvenli davranış arasındaki ilişki.
3. Güvenlik, emniyet, mahremiyet ve güvenilirlik kavramlarının ayrımı.
4. En az yetki, katmanlı savunma, güvenli varsayılanlar ve insan denetimi.
5. İş sahibi, AI ekibi, yazılım, platform ve güvenlik ekiplerinin sorumlulukları.

**Okuyucunun elde edeceği pratik çıktı:** Tek sayfalık Secure AI by Design ilkeleri ve sorumluluk matrisi.

**Başlangıç referansları:** NIST AI RMF; ISO/IEC 42001; ETSI EN 304 223.

## 2. AI Sistemlerinde Tehdit Modelleme ve Güven Sınırları

**Description**  
Bir AI uygulamasının veri akışını çıkarıyor; saldırganın erişebildiği yüzeyleri, aşabileceği sınırları ve ortaya çıkabilecek iş etkilerini sistematik biçimde inceliyoruz.

**Bölümün temel sorusu:** Saldırgan hangi noktadan girip hangi iş sonucunu etkileyebilir?

**İçerik akışı**

1. Veri kaynağından iş sonucuna kadar bileşen ve veri akışı haritası.
2. Varlık, saldırgan, yetenek, erişim ve güven sınırlarının tanımı.
3. Klasik ML ve üretken AI için poisoning, evasion, model extraction ve injection senaryoları.
4. Doğrudan kullanıcı girdisi, dış belge, araç sonucu ve servis kimliği sınırları.
5. Tehdit–kontrol–test ilişkisinin kurulması ve kalan riskin kaydı.

**Okuyucunun elde edeceği pratik çıktı:** Örnek bir asistan için veri akış diyagramı ve tehdit kaydı.

**Başlangıç referansları:** MITRE ATLAS; NIST AI 100-2 E2025; OWASP LLM/Agentic Top 10.

## 3. Güvenli AI Verisi: Kaynaktan Silmeye Kadar Koruma

**Description**  
Eğitim, değerlendirme ve çalışma zamanı verilerini; kaynak doğrulama, erişim sınırları, hassas veri koruma ve saklama politikalarıyla birlikte tasarlıyoruz.

**Bölümün temel sorusu:** Hangi veri nerede işleniyor, kim erişiyor ve ne zaman kaldırılıyor?

**İçerik akışı**

1. Eğitim, fine-tuning, test, kullanıcı ve retrieval verisinin farklı güvenlik ihtiyaçları.
2. Veri kökeni, değişiklik geçmişi ve poisoning riskleri.
3. Yapılandırılmış PII için regex/checksum; bağlamsal veriler için NER yaklaşımı.
4. Maskeleme, pseudonymization ve anonimleştirme arasındaki farklar.
5. Saklama/silme kapsamı: kaynak, indeks, cache, log, export ve yedekler.

**Okuyucunun elde edeceği pratik çıktı:** Veri envanteri, PII işleme akışı ve veri yaşam döngüsü kontrol listesi.

**Başlangıç referansları:** ISO/IEC 27701; NIST AI 600-1; OWASP AISVS.

## 4. Modelden Bağımlılığa: AI Tedarik Zincirini Güvenceye Almak

**Description**  
Model ağırlıkları, veri setleri, kütüphaneler ve araç bağlantılarının sisteme kabulünden güncellenmesine kadar güven zincirini nasıl kurabileceğimizi ele alıyoruz.

**Bölümün temel sorusu:** Çalıştırdığımız artefaktın beklediğimiz kaynaktan ve süreçten geldiğini nasıl doğrularız?

**İçerik akışı**

1. Temel model, tokenizer, adapter, veri seti ve paket bağımlılıklarının envanteri.
2. Model edinimi/fine-tuning süreçlerinde kaynak, lisans ve erişim kontrolleri.
3. Hash, imza, SBOM ve provenance bilgilerinin neyi gösterdiği.
4. Güvenilmeyen dosya biçimleri, yükleme kodları ve build ortamlarının değerlendirilmesi.
5. Model/kural/araç değişikliğinde regresyon, yayın onayı ve geri dönüş.

**Okuyucunun elde edeceği pratik çıktı:** Model kabul formu ve sürümlenmiş artefakt yayın akışı.

**Başlangıç referansları:** NIST SP 800-218A; SLSA; OWASP AISVS.

## 5. Prompt Injection: Veri ile Talimat Arasındaki Güven Sınırı

**Description**  
Kullanıcı mesajları, belgeler ve araç sonuçları üzerinden gelen talimat manipülasyonlarını inceliyor; tespit yöntemlerini erişim ve yürütme sınırlarıyla birlikte değerlendiriyoruz.

**Bölümün temel sorusu:** Dış içerik model davranışını etkilediğinde gerçek işleme hangi sınırlar uygulanıyor?

**İçerik akışı**

1. Doğrudan ve dolaylı injection; jailbreak ile ilişkisi ve kavramsal farkları.
2. RAG belgeleri, web içeriği, araç sonuçları ve konuşma geçmişi.
3. Çok turlu saldırılar, gizleme ve farklı içerik türleri.
4. Kaynak etiketleme, bağlam ayrımı, minimizasyon ve dedektörlerin sınırları.
5. Model manipüle olduğunda zararı azaltan araç, veri çıkışı ve yetki kontrolleri.

**Okuyucunun elde edeceği pratik çıktı:** Injection senaryo kataloğu ve her senaryoya bağlı kontrol/test tablosu.

**Başlangıç referansları:** OWASP LLM Top 10; OWASP Agentic Top 10; MITRE ATLAS.

## 6. Katmanlı Guardrails: Regex, NLP, Karar Modelleri ve SLM

**Description**  
Farklı güvenlik yöntemlerinin hangi işlerde kullanılabileceğini karşılaştırıyor; bulgu üretimini bağlayıcı politika kararından ayıran bir kontrol mimarisi kuruyoruz.

**Bölümün temel sorusu:** Hangi kontrolü hangi riske karşı çalıştırıyoruz ve ek maliyetine değiyor mu?

**İçerik akışı**

1. Regex/checksum, token kuralları, NER ve embedding tabanlı yöntemlerin görevleri.
2. Klasik ML risk sınıflandırıcıları ile Jev türü decision modellerin ayrımı.
3. Güvenlik SLM'leri, görev kapsamı ve Türkçe değerlendirme ihtiyacı.
4. Model skoru, kalibrasyon, eşik seçimi ve yanlış engelleme maliyeti.
5. Kontrol seçimi, fail-safe davranış ve yöntemlerin ablasyonla karşılaştırılması.

**Okuyucunun elde edeceği pratik çıktı:** Yöntem–risk matrisi, örnek politika ve karşılaştırmalı değerlendirme planı.

**Başlangıç referansları:** OWASP AISVS; NIST AI 600-1; Model sağlayıcılarının resmî belgeleri.

## 7. RAG ve Bellek Güvenliği: Doğru Bilgiye Yetkili Erişim

**Description**  
Belge, embedding ve kalıcı bellek katmanlarında erişim izolasyonunu ve içerik bütünlüğünü birlikte ele alıyor; oturumlar arasında taşınabilen riskleri inceliyoruz.

**Bölümün temel sorusu:** Model yalnızca kullanıcının görmeye yetkili olduğu ve uygun biçimde yönetilen bilgiyi mi kullanıyor?

**İçerik akışı**

1. Belge/parça yetkileri ve retrieval sırasında ACL uygulama.
2. Kullanıcı/tenant izolasyonu ve embedding/metadata güvenliği.
3. İndeksleme ve kaynak güncellemelerinde poisoning riskleri.
4. Kalıcı bellek yazımı, köken bilgisi, karantina ve silme.
5. Alıntı/kaynak desteği ve erişilebilir kaynak ile doğru cevap arasındaki ayrım.

**Okuyucunun elde edeceği pratik çıktı:** Yetkili retrieval ve bellek yazımı için referans akış ve testler.

**Başlangıç referansları:** OWASP LLM Top 10; OWASP AISVS C8; OWASP Agentic Top 10.

## 8. AI Ajanlarında En Az Yetki ve Güvenli Araç Kullanımı

**Description**  
Araç kullanan ajanlarda kimlik, kaynak kapsamı, parametre doğrulama ve insan onayını gerçek yürütme noktalarına yerleştiriyoruz.

**Bölümün temel sorusu:** Ajanın yapabildiği her işlem, kullanıcı ve görev için gerçekten gerekli ve yetkili mi?

**İçerik akışı**

1. Modelin araç önermesi ile aracın çalıştırılması arasındaki ayrım.
2. Servis/son kullanıcı kimliği, yetki devri ve alt ajan izinleri.
3. Araç allowlist'i, parametre/kaynak/hedef denetimi ve MCP güven sınırları.
4. Alıcı ve işlem kapsamına bağlı onay; tekrar kullanım ve parametre değişikliği.
5. Adım, maliyet, süre bütçeleri; iptal yayılımı ve idempotency.

**Okuyucunun elde edeceği pratik çıktı:** Araç yetki matrisi ve onaylı işlem yürütme tasarımı.

**Başlangıç referansları:** OWASP Agentic Top 10; OWASP ACS; OWASP AISVS C9/C10.

## 9. Güvenli AI Sunumu: API Gateway'den Model Sunucusuna

**Description**  
Model erişimini ve güvenlik kontrollerini gateway, uygulama ve çıkarım sunucusu arasında konumlandırarak denetimin atlanamadığı bir sunum mimarisi tasarlıyoruz.

**Bölümün temel sorusu:** Bir istemci veya fallback yolu zorunlu güvenlik kontrolünü atlayabiliyor mu?

**İçerik akışı**

1. Gateway, güvenlik servisi ve model sunucusunun farklı sorumlulukları.
2. LiteLLM/vLLM üzerinden sağlayıcıdan bağımsız bir referans bağlantı.
3. Ağ izolasyonu, kimlik doğrulama, sır yönetimi ve endpoint envanteri.
4. Girdi/çıktı kontrolleri, streaming tamponlama ve veri salım politikası.
5. Cache, retry, fallback ve guard çağrılarının bütçe/güven sınırları.

**Okuyucunun elde edeceği pratik çıktı:** Dağıtım diyagramı ve gateway bypass/streaming/fallback kabul testleri.

**Başlangıç referansları:** OWASP ASVS/API Security; ETSI EN 304 223; LiteLLM/vLLM resmî belgeleri.

## 10. AI Güvenliğini Ölçmek: Red Teaming ve Türkçe Değerlendirme

**Description**  
Güvenlik iddialarını saldırı başarısı, yanlış engelleme, görev tamamlama ve performans ölçümleriyle sınayan tekrarlanabilir bir değerlendirme yaklaşımı oluşturuyoruz.

**Bölümün temel sorusu:** Kontrol gerçekten işe yarıyor mu ve normal kullanıcıya maliyeti ne?

**İçerik akışı**

1. Tehdit modelinden izinli saldırı ve meşru görev senaryoları türetme.
2. Türkçe ekler, yazım varyasyonları, kod değiştirme ve sektörel bağlam.
3. Geliştirme, kalibrasyon ve bağımsız test verisinin ayrılması.
4. Saldırı başarısı, sızıntı, yetkisiz işlem, FPR/FNR ve gecikme ölçümleri.
5. Model-as-judge sınırlılıkları, insan incelemesi, belirsizlik ve sürüm regresyonu.

**Okuyucunun elde edeceği pratik çıktı:** Sürüm/model/politika bilgili benchmark raporu şablonu.

**Başlangıç referansları:** MITRE ATLAS; NIST AI 100-2; OWASP AISVS.

## 11. AI Operasyonlarında İzlenebilirlik ve Olay Müdahalesi

**Description**  
Güvenlik kararlarını gerçek işlem sonuçlarıyla ilişkilendiriyor; hassas veriyi koruyan kayıt, alarm, müdahale ve toparlanma süreçlerini tasarlıyoruz.

**Bölümün temel sorusu:** Bir olayda ne olduğunu görebiliyor, etkisini durdurabiliyor ve güvenle toparlanabiliyor muyuz?

**İçerik akışı**

1. Trace, güvenlik audit olayı, servis logu ve metriğin ayrımı.
2. Model/politika/feature sürümü, karar gerekçesi ve yürütme korelasyonu.
3. Log minimizasyonu, export öncesi maskeleme ve erişim/saklama sınırları.
4. Kalıcı olay teslimi, kayıt arızası ve sonucu bilinmeyen işlemler.
5. Ajan durdurma, yetki iptali, model/politika rollback ve olay sonrası inceleme.

**Okuyucunun elde edeceği pratik çıktı:** Örnek olay şeması ve AI güvenlik olayı müdahale runbook'u.

**Başlangıç referansları:** NIST CSF 2.0; NIST SP 800-61 Rev.3; OWASP AISVS C12.

## 12. Standarttan Kanıta: AI Güvenliğini Denetlenebilir Hale Getirmek

**Description**  
Serinin kontrollerini uluslararası referanslarla eşliyor; teknik testleri kurumsal risk, etki değerlendirmesi ve üretime geçiş kararına bağlayan bir kanıt paketi hazırlıyoruz.

**Bölümün temel sorusu:** Güvenlik iddiamızı hangi kanıtla ve hangi kapsam sınırıyla savunuyoruz?

**İçerik akışı**

1. ISO/IEC 27090, ETSI, NIST ve OWASP kaynaklarının farklı amaçları.
2. ISO/IEC 42001/23894/42005 ile yönetim, risk ve etki sorumlulukları.
3. Standart/çerçeve, risk listesi, taksonomi ve sertifikasyon ayrımı.
4. Kaynak → kontrol → özellik/politika → test → kanıt → kalan risk zinciri.
5. Örnek asistanın birleşik referans mimarisi ve risk bazlı yayın kapısı.

**Okuyucunun elde edeceği pratik çıktı:** Standart eşleştirme matrisi, kanıt dizini ve üretime geçiş kontrol listesi.

**Başlangıç referansları:** ISO/IEC 27090; ISO/IEC 42001/23894/42005; NIST AI RMF + OWASP AISVS.

## Bölümler arasındaki ilerleme

| Aşama | Bölümler | Ortaya çıkan çalışma |
| --- | --- | --- |
| Temeli kur | 1–2 | Korunacak varlıklar, sahipler, veri akışı ve tehdit modeli |
| Veri/model temelini güvenceye al | 3–4 | Veri yaşam döngüsü ve güvenilir artefakt kabulü |
| Çalışma zamanı kontrollerini tasarla | 5–9 | Injection, guardrails, RAG, ajan ve sunum mimarisi |
| Ölç ve işlet | 10–11 | Değerlendirme raporu, kayıt ve olay müdahalesi |
| Kanıtla ve yayınla | 12 | Kontrol eşleştirmesi, kalan risk ve üretime geçiş kararı |

## Yayın sırasında korunacak doğruluk kuralları

- Güvenlik ve performans rakamları yalnız ölçülmüş ortam/model/veri/politika sürümleriyle yayımlanır. Taslak sonuçlar gerçek benchmark gibi sunulmaz.
- “Tam koruma”, “sıfır sızıntı garantisi” veya “standart uyumlu” gibi kapsamı aşan ifadeler kullanılmaz.
- Standartların adı, türü, sürümü ve incelenen kaynak yazılır. ISO kamu özetinden madde düzeyinde uygunluk sonucu çıkarılmaz.
- Her bölüm yayımlanırken model, kütüphane ve standart sürümleri resmî kaynaklardan yeniden kontrol edilir. Bu plan otomatik veya zamanlanmış yayın görevi oluşturmaz.
- Güvenlik örneklerinde hedef yetkisi, sentetik veri, test bütçesi ve ortam sınırı açıkça belirtilir.
- LLM-as-judge çıktıları tek başarı kanıtı sayılmaz; meşru görev testleri ve gerekli insan incelemesi korunur.

## Başlangıç kaynak listesi

Aşağıdaki kaynaklar seri için başlangıç okuma listesidir. Bölüm metinleri yazılırken ilgili tam kaynaklar ve o tarihteki sürümler doğrulanmalıdır.

- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1)
- [NIST AI 100-2 E2025](https://csrc.nist.gov/pubs/ai/100/2/e2025/final)
- [ISO/IEC 27090:2026](https://www.iso.org/standard/27090)
- [ISO/IEC 42001:2023](https://www.iso.org/standard/42001)
- [ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html)
- [ISO/IEC 42005:2025](https://www.iso.org/standard/42005)
- [ISO/IEC 27701:2025](https://www.iso.org/standard/27701)
- [ETSI EN 304 223](https://portal.etsi.org/webapp/workprogram/Report_WorkItem.asp?WKI_ID=74952)
- [OWASP LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)
- [OWASP Agentic Top 10](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/)
- [OWASP AISVS](https://owasp.github.io/www-project-artificial-intelligence-security-verification-standard-aisvs-docs/)
- [OWASP ASVS](https://asvs.owasp.org/)
- [OWASP API Security](https://owasp.github.io/www-project-api-security/)
- [OWASP ACS](https://genai.owasp.org/resource/agent-control-standard-acs/)
- [MITRE ATLAS](https://atlas.mitre.org/)
- [SLSA](https://slsa.dev/spec/v1.2/)
- [NIST SP 800-218A](https://csrc.nist.gov/pubs/sp/800/218/a/final)
- [NIST SP 800-61 Rev.3](https://csrc.nist.gov/pubs/sp/800/61/r3/final)
- [LiteLLM Custom Guardrail](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail)
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)

