Kurumsal yapay zekâ asistanının bağımsız bir siber güvenlik firması tarafından gerçekleştirilen sızma testi (penetration test) raporu masaya geldiğinde yönetim şok geçirdi. Raporda kritik dereceli üç temel güvenlik açığı listelenmişti ve bunların hiçbiri modelin kendi ağırlıklarıyla veya eğitim verisiyle ilgili değildi:

1. **Model Sunucusunun Açıkta Olması:** vLLM çıkarım sunucusunun çalıştığı 8000 portuna şirket içi yerel ağdaki (LAN) tüm kullanıcılar doğrudan erişebiliyordu. Mühendislik ekibi sunucuyu `--api-key` bayrağı ile başlatmıştı; bu nedenle standart `/v1/chat/completions` uç noktası yetkisiz isteklere 401 Unauthorized dönüyordu. Ancak vLLM mimarisinde API anahtarı yalnızca belirli yol öneklerini koruduğu için; `/invocations`, `/tokenize`, `/metrics`, `/pause` ve `/update_weights` gibi uç noktalar kimlik doğrulaması olmaksızın herkese yanıt veriyordu.
2. **Korumasız Yedek (Fallback) Rotası:** AI Gateway üzerinde, kurum içi GPU kümesi aşırı yüklendiğinde gelen trafiği harici bir bulut modeline yönlendiren otomatik bir yük dengeleme (fallback) rotası kurulmuştu. Ancak girdi ve çıktı guardrail filtreleri yalnızca yerel kümeye giden birincil yola bağlanmıştı. Şirket içi sunucular her yüke girdiğinde veya bir saldırgan sunucuyu kasıtlı olarak istek bombardımanına tuttuğunda, tüm trafik güvenlik kontrollerini tamamen atlayarak harici sağlayıcıya korumasız akıyordu.
3. **Akış Sırasında Sızıntı (Streaming Leakage):** Çıktı guardrail filtresi, token akışı (streaming) yapan uç noktada sonradan denetim (`post_call`) kipinde çalışıyordu. Model, başka bir müşterinin T.C. kimlik numarasını ve telefonunu içeren bir yanıt ürettiğinde guardrail ihlali tespit etti; ancak tespit anında metin kullanıcının tarayıcısına token token çoktan aktarılmıştı.

Önceki sekiz bölümde tasarladığımız tüm kontroller mevcuttu; fakat saldırganların bu kontrolleri aşması gerekmedi. Tek yapmaları gereken, **kontrollerden geçmeyen bir kestirme yol bulmaktı**.

Sunum katmanının (serving layer) varlık sebebi tam olarak budur: Her çıkarım isteğinin, her alternatif rotada, her sistem arızasında tavizsiz biçimde aynı güvenlik süzgeçlerinden geçmesini sağlamak ve gateway arkasındaki donanım ve yazılım bileşenlerine doğrudan erişimi imkânsız kılmak.

Bu rehberde, bir Principal AI Security Architect perspektifiyle kurumsal AI sunum altyapısını tahkim ediyoruz. Model sunucularını (vLLM) ve AI Gateway (LiteLLM) katmanını donanım seviyesinden ağ sınırına kadar izole ediyoruz. **NVIDIA H100 Confidential Computing** mimarisini inceliyoruz: SPDM (Security Protocol and Data Model) ile cihaz kanıtlaması, CPU TEE ile GPU arasındaki şifreli aktarım ve **PCIe IDE (Integrity and Data Encryption)** standardının bu resimdeki yeri. vLLM'in önek önbelleğinin (prefix cache) kiracılar arasında açtığı zamanlama yan kanalını kiracı başına önbellek tuzuyla kapatıyoruz. Akış gecikmesi (TTFT) ile sızıntı penceresi arasındaki gerilimi bir bekletme penceresi ve **spekülatif geri çekme** (speculative rollback) ile yönetiyor; çıkarım zamanının belgelenmiş yan kanallarını ele alıyoruz.

> **Önceki bölümden devraldığımız temel:** [Altıncı bölüm](post.html?slug=katmanli-guardrails), tespit katmanlarını kurdu ve hata davranışına işlem bazında karar verdi. [Sekizinci bölüm](post.html?slug=ajanlarda-en-az-yetki), yürütücüyü harici yan etkilere giden tek kapı yaptı. Şimdi gateway bileşenini model çıkarımına giden tek meşru kapı hâline getiriyoruz.

> **Bu yazının sonunda**
>
> - Gateway, güvenlik servisi ve model sunucusu arasındaki mimari güven sınırlarını ve sorumluluk matrisini kurabileceksiniz.
> - GPU gizli bilişiminin neyi koruduğunu (kanıtlama, SPDM, şifreli aktarım), neyi korumadığını ve PCIe IDE'nin nereye oturduğunu açıklayabileceksiniz.
> - vLLM sunucusunun API anahtarı arkasında kalmayan açık uç noktalarını ağ politikaları ve ters vekillerle izole edebileceksiniz.
> - Kiracılar arasındaki önek önbelleği zamanlama kanalını, gateway'in koyduğu kiracı başına `cache_salt` ile kapatabileceksiniz.
> - Akış (streaming) uç noktalarında, TTFT'ye bilinen küçük bir bedelle yapılandırılmış PII'ı ekrana düşmeden yakalayan bekletme tamponları uygulayabileceksiniz.

**Bu yazıda**

- [1. Üç bileşen, üç sorumluluk: Gateway, güvenlik servisi ve model sunucusu](#1-üç-bileşen-üç-sorumluluk-gateway-güvenlik-servisi-ve-model-sunucusu)
- [2. Donanım güvenliği ve Confidential AI: H100 TEE, SPDM ve PCIe IDE](#2-donanım-güvenliği-ve-confidential-ai-h100-tee-spdm-ve-pcie-ide)
  - [2.1. Donanım güven kökü ve SPDM firmware doğrulaması](#21-donanım-güven-kökü-ve-spdm-firmware-doğrulaması)
  - [2.2. PCIe veriyolunda şifreli aktarım ve korunmayan yerler](#22-pcie-veriyolunda-şifreli-aktarım-ve-korunmayan-yerler)
- [3. Sağlayıcıdan bağımsız referans mimari: vLLM önünde LiteLLM](#3-sağlayıcıdan-bağımsız-referans-mimari-vllm-önünde-litellm)
- [4. Ağ, kimlik, sırlar ve model sunucusu uç nokta envanteri](#4-ağ-kimlik-sırlar-ve-model-sunucusu-uç-nokta-envanteri)
  - [4.1. Bir API anahtarının kapsamadıkları: vLLM saldırı yüzeyi](#41-bir-api-anahtarının-kapsamadıkları-vllm-saldırı-yüzeyi)
  - [4.2. Çok kiracılı önek önbelleği: zamanlama sızıntısı ve önbellek tuzu](#42-çok-kiracılı-önek-önbelleği-zamanlama-sızıntısı-ve-önbellek-tuzu)
- [5. Akış güvenliği (streaming guardrails): Gecikme bütçesi ve spekülatif geri çekme](#5-akış-güvenliği-streaming-guardrails-gecikme-bütçesi-ve-spekülatif-geri-çekme)
  - [5.1. Akışın gecikme ve sızıntı penceresi ikilemi](#51-akışın-gecikme-ve-sızıntı-penceresi-ikilemi)
  - [5.2. Kayan pencere tamponlaması ve spekülatif iptal algoritması](#52-kayan-pencere-tamponlaması-ve-spekülatif-iptal-algoritması)
- [6. Çıkarım zamanı yan kanal saldırıları (side-channel attacks)](#6-çıkarım-zamanı-yan-kanal-saldırıları-side-channel-attacks)
- [7. Önbellek, yeniden deneme, yedek yol ve guard çağrıları](#7-önbellek-yeniden-deneme-yedek-yol-ve-guard-çağrıları)
- [8. Güvenlik kontrolü olarak kotalar, hız sınırları ve bütçe](#8-güvenlik-kontrolü-olarak-kotalar-hız-sınırları-ve-bütçe)
- [9. Pratik çıktı: Dağıtım mimarisi, kabul testleri ve Python güvenlik motoru](#9-pratik-çıktı-dağıtım-mimarisi-kabul-testleri-ve-python-güvenlik-motoru)
  - [9.1. Dağıtım kabul testleri matrisi](#91-dağıtım-kabul-testleri-matrisi)
  - [9.2. Üretim tipi Python sunum güvenlik motoru](#92-üretim-tipi-python-sunum-güvenlik-motoru)
- [Bölümün pedagojik durakları](#bölümün-pedagojik-durakları)
  - [Kendinizi sınayın](#kendinizi-sınayın)
  - [Düşünce deneyi](#düşünce-deneyi)
  - [Kavramları ayrıştırma](#kavramları-ayrıştırma)
  - [Okuyucu artık hangi sorulara kendi başına yanıt verebilir?](#okuyucu-artık-hangi-sorulara-kendi-başına-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Üç bileşen, üç sorumluluk: Gateway, güvenlik servisi ve model sunucusu

Yüksek güvenlikli bir banka merkezini hayal edin: Girişte bir resepsiyon turnikesi, arkasında x-ray ve güvenlik tarama ofisi, en arkada ise kilitli bir çelik kasa bulunur. Resepsiyon kimliğinizi doğrular ve ziyaret yetkinizi kontrol eder. Güvenlik ofisi üzerinizde silah veya kayıt cihazı olup olmadığını tarar. Çelik kasa ise yalnızca resepsiyon ve güvenliğin ortak onayından geçmiş kişilere açılır; sokaktan kasaya açılan hiçbir doğrudan kapı yoktur.

Kurumsal bir AI sunum platformu da tam olarak bu üçlü sorumluluk ayrımına dayanır:

| Bileşen | Temel Sorumluluk | Kesinlikle Yapmaması Gereken Hata |
| :--- | :--- | :--- |
| **AI Gateway** | İstemcileri doğrula; hız ve kota sınırlarını uygula; rotalama yap; her güzergâhta guardrail'i tetikle; mTLS mühürle | Güvenlik kancalarını atlayan korumasız bir kestirme rotaya sahip olmak |
| **Güvenlik Servisi** | Girdi ve çıktı dedektörlerini çalıştır; politika motoruna göre izin/ret kararı üret | İsteğe bağlı kalmak; yüksek etkili işlemlerde zaman aşımında açık kalmak (fail-open) |
| **Model Sunucusu** | Yalnızca gateway'den gelen doğrulanmış istekler için GPU çıkarımı yap | Gateway dışındaki ağlara açık olmak; yönetim uç noktalarını ağa açmak |

<!-- fig:p9_deploy -->
<svg viewBox="0 0 720 460" role="img" aria-label="Dağıtım diyagramı. İstemci bölgesinde sohbet arayüzü orkestratör ve yürütücüyle konuşur. Orkestratör gateway&#x27;i karşılıklı TLS ile çağırır. Gateway kimlik doğrular, kotaları uygular, yönlendirir, yedek yol dahil her güzergâhta güvenlik servisini çağırır ve sürümleri loglar; sırları bir sır yöneticisinden alır. Model bölgesi, vLLM düğümlerine yalnızca gateway&#x27;den karşılıklı TLS ile erişilebilen özel bir alt ağdır; KV aktarımı ve veri paralel trafiği için iç portlar yalnızca düğümler arasında açıktır; /v1 dışındaki çıkarım, kontrol ve yardımcı uç noktalar yönlendirilmez. Uygulamadan vLLM&#x27;e doğrudan yol engellidir. Barındırılan bir yedek sağlayıcıya yalnızca aynı guard&#x27;larla gateway üzerinden ulaşılır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p9d-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p9d-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
<marker id="p9d-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
<marker id="p9d-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Dağıtım: modele tek yol, her güzergâhta korumalı</text>
<rect x="16" y="44" width="160" height="290" rx="8" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="26" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">istemci bölgesi</text>
<rect x="192" y="44" width="196" height="290" rx="8" style="fill:var(--c-warn);fill-opacity:.05;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="202" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">gateway bölgesi</text>
<rect x="404" y="44" width="300" height="186" rx="8" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.3;stroke-dasharray:5 4"/>
<text x="414" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">model bölgesi (özel alt ağ)</text>
<rect x="28" y="76" width="136" height="50" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="97.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Sohbet arayüzü</text>
<text x="96.0" y="113.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">görsel/bağlantı yok</text>
<rect x="28" y="150" width="136" height="64" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="170.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Orkestratör</text>
<text x="96.0" y="186.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">yürütücü, kaynak</text>
<text x="96.0" y="202.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">etiketleri (Bölüm 8)</text>
<line x1="96" y1="126" x2="96" y2="148" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="204" y="76" width="172" height="138" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="290" y="98" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Gateway (LiteLLM)</text>
<text x="216" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">istemciyi doğrula</text>
<text x="216" y="136" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kota: token, maliyet</text>
<text x="216" y="152" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yönlendir + yedek yol</text>
<text x="216" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her yolda guard</text>
<text x="216" y="184" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yolu, sürümleri logla</text>
<text x="216" y="200" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">giriş ve çıkışta mTLS</text>
<line x1="164" y1="182" x2="202" y2="182" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="204" y="240" width="172" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="290.0" y="256.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Güvenlik servisi</text>
<text x="290.0" y="272.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">dedektör, politika,</text>
<text x="290.0" y="288.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">guard SLM (Bölüm 6)</text>
<line x1="276" y1="214" x2="276" y2="238" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="304" y1="238" x2="304" y2="216" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M28 253 V291 A68.0 7 0 0 0 164 291 V253" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<ellipse cx="96.0" cy="253" rx="68.0" ry="7" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="96.0" y="273.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Sır yöneticisi</text>
<text x="96.0" y="288.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">sağlayıcı anahtarları</text>
<line x1="164" y1="270" x2="202" y2="202" marker-end="url(#p9d-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="420" y="84" width="128" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="484.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">vLLM düğümü</text>
<text x="484.0" y="116.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">yalnızca /v1</text>
<text x="484.0" y="132.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">gateway&#x27;den</text>
<rect x="562" y="84" width="128" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="626.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">vLLM düğümü</text>
<text x="626.0" y="116.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">yalnızca /v1</text>
<text x="626.0" y="132.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">gateway&#x27;den</text>
<line x1="548" y1="112" x2="562" y2="112" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="555" y="156" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">KV :14579, DP :29500 yalnızca düğümler arası</text>
<line x1="376" y1="112" x2="418" y2="112" marker-end="url(#p9d-o)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="420" y="176" width="270" height="40" rx="6" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="430" y="192" text-anchor="start" style="fill:var(--c-text);font-size:11px">/invocations, /score, /pooling, /pause,</text>
<text x="430" y="206" text-anchor="start" style="fill:var(--c-text);font-size:11px">/update_weights, gRPC, geliştirme kipi: kapalı</text>
<path d="M164 204 H184 V338 H400 V196 H416" marker-end="url(#p9d-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="420" y="322" text-anchor="start" style="fill:var(--c-danger);font-size:11px">uygulama → vLLM doğrudan: ağ politikası engeller</text>
<rect x="420" y="250" width="156" height="50" rx="2" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:2"/>
<text x="498.0" y="272.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Barındırılan yedek</text>
<text x="498.0" y="287.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dış sağlayıcı</text>
<path d="M376 180 C 398 180, 398 275, 418 275" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="590" y="270" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">birincille aynı</text>
<text x="590" y="285" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">guard&#x27;lar</text>
<rect x="16" y="346" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.3"/>
<text x="28" y="365" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Guard modelleri ve yedek yollar dahil her model çağrısı gateway&#x27;den geçer: doğrulanır, kontrol edilir, sayılır.</text>
<text x="28" y="381" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">S-1&#x27;den S-10&#x27;a kabul testleri bu diyagramın her çizgisini kanıtlar.</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="388" width="688" height="60" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="407" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 SUNUM GÜVENLİĞİ: MODEL SUNUCUSUNA ASLA DOĞRUDAN ERİŞİM VERİLEMEZ</text>
<text x="28" y="425" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Model sunucusunun /invocations ve /metrics uç noktaları kimliksizdir; yalnızca allowlist uygulayan Gateway arkasında mTLS ile barındırılmalıdır.</text>
<text x="28" y="440" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Confidential AI (NVIDIA H100 TEE / AMD SEV-SNP), hipervizör operatörünün dahi GPU belleğindeki ağırlıkları ve KV cache'i okumasını donanımsal olarak engeller.</text>
</svg>

---

## 2. Donanım güvenliği ve Confidential AI: H100 TEE, SPDM ve PCIe IDE

### 2.1. Donanım güven kökü ve SPDM firmware doğrulaması

Geleneksel bir bulut kurulumunda hypervisor'ü ya da ana makine işletim sistemini kontrol eden kişi, bir misafirin GPU'suna gönderdiği her şeyi okuyabilir: model ağırlıkları, gelen prompt'lar ve KV cache içeriği, ana makinenin inceleyebildiği bellekten geçer.

**Confidential computing** (gizli bilişim), ana makineyi güvenilen tabandan çıkarır. İş yükü gizli bir sanal makinede (CPU TEE: AMD SEV-SNP ya da Intel TDX) çalışır; H100 kuşağından itibaren NVIDIA GPU'ları da bu TEE'yi hızlandırıcıya uzatan gizli bir kipe alınabilir. Tasarımı iki mekanizma taşır:

1. **Kanıtlama (attestation).** GPU'da bir donanım güven kökü ve NVIDIA sertifika otoritesine zincirlenen bir cihaz kimlik sertifikası bulunur. İstendiğinde firmware'inin ve yapılandırmasının imzalı ölçümlerini, yani *kanıtlama raporunu* üretir. Rapor, NVIDIA Remote Attestation Service (NRAS) üzerinden ya da dış ağa kapalı ortamlarda yerel olarak referans değerlerle karşılaştırılır.
2. **SPDM oturumu.** Gizli sanal makinenin *içindeki* GPU sürücüsü, GPU ile SPDM (Security Protocol and Data Model; cihaz kimlik doğrulaması ve anahtar değişimi için DMTF standardı) üzerinden güvenli bir oturum açar ve veriyolundan geçecek her şeyin anahtarlarını türetir.

Bir kurulum için asıl önemli kontrol **kanıtlamaya bağlı anahtar teslimidir**: model ağırlıklarını çözen anahtar, anahtar servisiniz tarafından ancak iki kanıtlama da (CPU TEE ve GPU) doğrulandıktan sonra verilir. Firmware'i kurcalanmış bir düğüm anahtarı hiç alamaz.

### 2.2. PCIe veriyolunda şifreli aktarım ve korunmayan yerler

Ana bellek ile GPU arasındaki veriyolu en belirgin dinleme noktasıdır: cihazla paylaşılan bellek hypervisor'e görünür; fiziksel erişimi olan biri de PCIe yuvasına bir ara aygıt (interposer) yerleştirebilir.

H100'de koruma, SPDM oturum anahtarlarıyla sürücü ve GPU tarafından uygulanır. Veri, paylaşılan bellekteki **şifreli bir sıçrama tamponundan** (encrypted bounce buffer) geçer; komut tamponları ve CUDA çekirdekleri veriyoluna çıkmadan önce şifrelenir ve imzalanır. Ana makine de ara aygıt da şifreli metin görür.

Üç sınır tasarımın parçasıdır ve tehdit modelinize yazılmalıdır:

- **GPU belleğinin kendisi şifrelenmez.** NVIDIA'nın kendi anlatımına göre paket üstü HBM, gündelik fiziksel saldırı araçlarına karşı güvenli kabul edilir ve şifrelenmez. Sınır, VRAM üzerindeki bir şifre değil, GPU paketinin kendisidir.
- **PCIe IDE standarttır, H100'ün mekanizması değildir.** Veriyolunun bağlantı düzeyinde şifrelenmesi PCIe IDE (Integrity and Data Encryption, AES-GCM) olarak standartlaşmıştır; TDISP ise bir TEE'nin bir cihaz arayüzünü nasıl sahiplendiğini tanımlar. Bunların üzerine kurulan platformlar (Intel TDX Connect, AMD SEV-TIO) korumayı sürücüden bağlantıya taşır. Bir satıcı "PCIe IDE" diyorsa, gerçekte hangi CPU, GPU ve firmware kuşağını çalıştırdığınıza bakın.
- **Gizli kip bir kiracılık sınırı değildir.** Tek bir iş yükünü altındaki altyapıdan korur. Aynı model sunucusu sürecini paylaşan iki kiracı için hiçbir şey yapmaz; o konu 4.2'dedir.

---

## 3. Sağlayıcıdan bağımsız referans mimari: vLLM önünde LiteLLM

Referans mimaride çıkarım motoru olarak açık kaynaklı **vLLM**, önündeki AI Gateway olarak ise **LiteLLM Proxy** konumlandırılmıştır.

LiteLLM'in sunduğu özel kanca (custom guardrail) mekanizmaları üç farklı yürütme kipine sahiptir:

| Guardrail Kipi | Yürütülme Zamanı | İçeriği Değiştirebilir mi? | Çağrıyı Engelleyebilir mi? |
| :--- | :--- | :--- | :--- |
| `pre_call` | Model çağrısından önce istek gövdesinde | Evet (PII maskeleme, bağlam kırpma) | Evet (PDP kararıyla 403 / 400 döner) |
| `during_call` | Model çıkarımıyla eşzamanlı paralel | Hayır (yalnızca denetim) | Evet (çıkarım bitince yanıtı düşürür) |
| `post_call` | Model çıktısı tamamlandığında | Standart yanıtta evet; **akışta hayır** | Standart yanıtta evet; **akışta engellemez** |

`post_call` kipinin akış (streaming) yanıtlarında hiçbir şeyi engelleyememesi kritik bir mimari kısıttır: İlk token istemciye ulaştığı anda sızıntı gerçekleşmiş olur. Bu nedenle Kısım 5'te akış denetimini spekülatif tamponlama katmanına taşıyoruz.

Örnek güvenli Gateway yapılandırması:

```yaml
model_list:
  - model_name: enterprise-assistant
    litellm_params:
      model: hosted_vllm/mistral-7b-instruct-v0.3
      api_base: "https://vllm-node.internal:8443/v1"
      ssl_verify: "/etc/ssl/certs/internal-ca.crt"
  - model_name: enterprise-assistant-fallback
    litellm_params:
      model: openai/gpt-4o
      api_key: "os.environ/SECURE_FALLBACK_KEY"

router_settings:
  fallbacks: [{"enterprise-assistant": ["enterprise-assistant-fallback"]}]

guardrails:
  - guardrail_name: input-pep-policy
    litellm_params:
      guardrail: enterprise_security.InputPolicy
      mode: pre_call
      default_on: true
  - guardrail_name: streaming-speculative-guard
    litellm_params:
      guardrail: enterprise_security.StreamingSpeculativeGuard
      mode: post_call          # akışta bu kanca parçaları kendisi tamponlar (5. bölüm)
      default_on: true
```

---

## 4. Ağ, kimlik, sırlar ve model sunucusu uç nokta envanteri

### 4.1. Bir API anahtarının kapsamadıkları: vLLM saldırı yüzeyi

vLLM'in güvenlik belgeleri açık bir mimari uyarıda bulunur: `--api-key` parametresi yalnızca `/v1`, `/v2`, `/inference` ve `/cohere` yol öneklerini korur. API anahtarı arkasında **olmayan** ve yerel ağdan doğrudan çağrılabilen kritik uç noktalar şunlardır:

- **Yetkisiz Çıkarım Yolları:** `/invocations`, `/pooling`, `/classify`, `/score`, `/rerank`. Saldırgan API anahtarı olmaksızın modeli bu yollardan çalıştırabilir.
- **Operasyonel Yollar:** `/pause`, `/resume`, `/abort_requests`, `/update_weights` ve diğerleri. Saldırgan çıkarımı durdurabilir, başka kullanıcıların isteklerini iptal edebilir ya da model ağırlıklarını değiştirebilir.
- **Yardımcı Yollar:** `/metrics`, `/tokenize`, `/detokenize`, `/health`, `/version`. Yük, model kimliği ve tokenizer davranışı sızdırırlar.
- **Dahili Dağıtık Portlar:** Veri paralelliği ana portu (varsayılan `:29500`) ve KV cache aktarım portu (`--kv-port`, varsayılan `:14579`). vLLM belgeleri düğümler arası bu iletişimi varsayılan olarak güvensiz diye tanımlar: şifresiz ve kimlik doğrulamasızdır, kendi ağına izole edilmelidir. İsteğe bağlı gRPC portunda da kimlik doğrulama ya da şifreleme yoktur.

Çözüm: Model sunucusu private bir alt ağa (`subnet-private-inference`) hapsedilir; Kubernetes `NetworkPolicy` veya bulut güvenlik grupları ile `:8443` portuna yalnızca Gateway pod'unun mTLS sertifikasıyla erişmesine izin verilir. Diğer tüm portlar düğüm dışına tamamen kapatılır.

```text
Kurumsal Çıkarım Uç Noktası Envanteri ve Güvenlik Matrisi:
--------------------------------------------------------------------------------
Uç Nokta / Port         Erişim İzni           Uygulanan Güvenlik Kontrolü
--------------------------------------------------------------------------------
vllm :8443 /v1/*        SADECE Gateway        mTLS + API Anahtarı + Ağ Politikası
vllm :8443 /invocations HİÇ KİMSE             Ters vekilde bloklu (404)
vllm :8443 /pause, ...  HİÇ KİMSE             Ters vekilde bloklu (404)
vllm :8443 /metrics     SADECE Prometheus     mTLS + RBAC
vllm :14579 (KV Trans)  SADECE vLLM Düğümleri İzole alt ağ, düğüm dışına kapalı
gateway :443 /v1/*      Uygulama / İstemci    OAuth2 JWT + Hız Sınırı (Rate Limit)
--------------------------------------------------------------------------------
```

### 4.2. Çok kiracılı önek önbelleği: zamanlama sızıntısı ve önbellek tuzu

Ortak prompt öneklerini yeniden hesaplamamak için vLLM'in otomatik önek önbelleği (automatic prefix caching) KV cache bloklarını saklar ve istekler arasında yeniden kullanır. Her blok, kendi token'larının özetinin bir önceki bloğun özetiyle zincirlenmesiyle tanımlanır; dolayısıyla bir istek bir bloğu ancak prompt'u **o blok dahil oraya kadar birebir aynıysa** yeniden kullanır.

Bu bir korkuyu ortadan kaldırır, daha ince bir korkuyu bırakır. Bir isteğe başka bir kiracının *farklı* bağlamı verilemez: isabet aynı token'ları gerektirir; aynı token'lar da aynı KV tensörlerini üretir, yani içerik yeni bir şey açığa vurmaz. Sızan şey **isabetin kendisidir**. Önbellekteki bir önek prefill'i atlar; yanıt ölçülebilir biçimde erken başlar ve ilk token süresi (TTFT) bir kehanet makinesine dönüşür:

1. Saldırgan aday bir önek gönderir ve TTFT'yi ölçer.
2. Hızlı yanıt, başka bir kullanıcının yakın zamanda tam o öneki gönderdiği anlamına gelir.
3. Tahmini blok blok uzatmak, bir sistem prompt'unu ya da bir başkasının belgesinin başını yeniden kurar.

vLLM bunu bir zamanlama yan kanalı olarak belgeler (CVE-2025-46570) ve yalnızca 8 token'lık bir önekin 0,99 ROC AUC ile ayırt edilebildiğini gösteren araştırmaya atıf yapar.

Önlem yerleşiktir. Bir istek `cache_salt` taşıyabilir; vLLM bu değeri ilk bloğun özetine karıştırır ve yalnızca aynı tuza sahip istekler önbellekteki önekleri paylaşabilir:

$$
h_1 = H(\text{cache\_salt} \parallel \text{tokens}_{\text{blok }1}), \qquad h_i = H(h_{i-1} \parallel \text{tokens}_{\text{blok }i})
$$

Bunu ipucu olmaktan çıkarıp kontrole çeviren üç kural vardır:

- **Tuzu istemci değil gateway koyar.** Kiracı başına, doğrulanmış kimlikten türetilir; örneğin $\text{cache\_salt} = \text{HMAC}_{K}(\text{tenant\_id})$. İstemcinin gönderdiği değerin üzerine yazılır.
- **Tuzlar gizlidir ve rastgele görünür** (256 bit); tahmin edilebilir kiracı kimlikleri değildir.
- **Tuz bir izolasyon sınırı değildir.** vLLM'in kendi yönergesi, aynı sunucu sürecini paylaşan kiracıları izole etmediğidir; birbirini hiçbir biçimde etkilememesi gereken kiracılar için kiracı başına ayrı örnek çalıştırın.

Bedeli önbellek verimidir: birebir aynı sistem prompt'u artık toplamda bir kez değil, kiracı başına bir kez hesaplanır.

---

## 5. Akış güvenliği (streaming guardrails): Gecikme bütçesi ve spekülatif geri çekme

### 5.1. Akışın gecikme ve sızıntı penceresi ikilemi

Son kullanıcı uygulamalarında ilk token gecikmesi (TTFT - Time-to-First-Token) kullanıcı deneyiminin kalbidir; kullanıcılar ilk kelimenin 500 ms altında görünmesini bekler. Ancak bu durum güvenlik açısından derin bir ikilem yaratır:
- **Tamponlama Yok (Gecikme 0, Sızıntı Penceresi %100):** Token'lar üretildiği an sokete yazılır. Kötü niyetli çıktı çoktan istemci ekranına düşmüştür; sonradan engelleme yapılamaz.
- **Tam Yanıt Tamponlama (Gecikme %100, Sızıntı Penceresi 0):** Yanıtın tamamı üretilip guardrail tarafından denetlenene kadar hiçbir şey gönderilmez. Kullanıcı 3-5 saniye boş ekrana bakar; akış özelliği çöker.

Modelin üretim hızı saniyede $v$ karakter ve geride tutulan pencere boyutu $W$ karakter olduğunda; istemciye yansıyan ilk gecikme artışı ($\Delta t$) ve ihlalin kullanıcıya gösterilmeden yakalanma şartı şudur:

$$
\Delta t = \frac{W}{v}, \qquad \text{gösterilmeden yakalanır} \iff s \le W
$$

Burada $s$, bir dedektörün ihlali tespit etmesi için ihtiyaç duyduğu karakter sayısıdır ($s_{\text{TCKN}} = 11$, $s_{\text{IBAN}} \approx 26$, $s_{\text{API\_KEY}} \approx 40$).

```text
Saniyede 40 token (v ≈ 160 karakter/sn) hızında çalışan bir model için analiz:
--------------------------------------------------------------------------------
W = 64 karakter (TCKN, IBAN, Kredi Kartı, API anahtarı için güvenli marj):
    Gecikme Artışı: Δt = 64 / 160 = 0,40 saniye (Kullanıcı fark etmez!)
    Sonuç: Tüm yapılandırılmış PII ve sırlar EKRANA DÜŞMEDEN yakalanır.

W = 200 karakter (Cümle düzeyinde prompt injection ve hakaret):
    Gecikme Artışı: Δt = 200 / 160 = 1,25 saniye.
--------------------------------------------------------------------------------
```

<!-- fig:p9_stream -->
<svg viewBox="0 0 720 356" role="img" aria-label="Zaman çizelgesi üzerinde üç strateji: post-call denetimi, tam tamponlama ve 200 karakterlik kayan pencere geride tutma. 300. karakterde üretilen TCKN yalnızca geride tutma penceresinde ekrana ulaşmadan yakalanır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Akan Çıktı: 500 Karakterlik Tek Bir Yanıt İçin Üç Koruma Stratejisi</text>
<!-- Time Axis -->
<line x1="16" y1="52" x2="704" y2="52" style="stroke:var(--c-border);stroke-width:1"/>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">t = 0.0s (başlangıç)</text>
<text x="430" y="44" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-weight:700;font-family:var(--font-mono)">t = 1.9s (ihlal üretildi)</text>
<text x="704" y="44" text-anchor="end" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">t = 3.1s (500 karakter)</text>

<!-- Strategy 1 -->
<rect x="16" y="64" width="688" height="60" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="28" y="84" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">1. Post-Call Denetim (Sıfır Gecikme)</text>
<rect x="28" y="94" width="664" height="14" rx="3" style="fill:var(--c-danger);fill-opacity:.18;stroke:var(--c-danger);stroke-width:1"/>
<circle cx="430" cy="101" r="5" style="fill:var(--c-danger)"/>
<text x="28" y="118" text-anchor="start" style="fill:var(--c-danger);font-size:10px;font-weight:600">Gecikme: 0s · Kontrol sonda çalışır · Sızıntı çoktan ekranda! ❌</text>

<!-- Strategy 2 -->
<rect x="16" y="134" width="688" height="60" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="28" y="154" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2. Tam Tamponlama (Maksimum Güvenlik)</text>
<rect x="28" y="164" width="664" height="14" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="650" y="164" width="42" height="14" rx="3" style="fill:var(--c-warn);fill-opacity:.3"/>
<text x="28" y="188" text-anchor="start" style="fill:var(--c-warn);font-size:10px;font-weight:600">Gecikme: 3,1s · Sona kadar hiçbir şey gösterilmez · TTFT felç! ❌</text>

<!-- Strategy 3 -->
<rect x="16" y="204" width="688" height="64" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.5"/>
<text x="28" y="224" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">3. Kayan Pencere Holdback (Optimal Denge)</text>
<rect x="28" y="234" width="402" height="14" rx="3" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1"/>
<rect x="430" y="234" width="60" height="14" rx="3" style="fill:var(--c-warn);fill-opacity:.4;stroke:var(--c-warn);stroke-width:1"/>
<line x1="490" y1="228" x2="490" y2="254" style="stroke:var(--c-danger);stroke-width:2.5;stroke-dasharray:3 2"/>
<text x="28" y="260" text-anchor="start" style="fill:var(--c-success);font-size:10px;font-weight:700">Gecikme: 1,25s (200 karakter) · TCKN ekrana düşmeden yakalanır! ✅</text>
<rect x="430" y="208" width="262" height="18" rx="4" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1"/>
<text x="561" y="221" text-anchor="middle" style="fill:var(--c-danger);font-size:9.5px;font-weight:700">🛡️ SPECULATIVE_REVOKE: Akış iptal edildi</text>

<!-- Bottom Educational Banner -->
<rect x="16" y="278" width="688" height="66" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="297" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 AKIŞ DENGESİ: GECİKME İLE GÜVENLİK ARASINDAKİ OPTİMAL NOKTA</text>
<text x="28" y="316" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Post-call kontrolleri yalnızca denetimdir, veri sızıntısını durduramaz; token istemciye bir kez aktığında geri alınamaz.</text>
<text x="28" y="332" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Kayan pencere (holdback window), hassas veriyi kullanıcı ekranına düşmeden önce yakalayıp spekülatif akışı iptal etmek (SPECULATIVE_REVOKE) için gereken en düşük gecikme maliyetidir.</text>
</svg>

Yakalama koşulu iki yönlü çalışır: pencereden uzun bir ihlal ya da ancak daha fazla bağlamla tanınabilen bir ihlal (cümle düzeyinde bir politika ihlali), yakalanmadan önce kısmen ekrana düşer. $W$'yi dedektörün eşleştirmesi gereken en uzun örüntüye göre boyutlandırın; akış üzerindeki anlamsal denetimleri ise elden gelenin en iyisi sayın.

### 5.2. Kayan pencere tamponlaması ve spekülatif iptal algoritması

En gelişmiş mimari çözüm, Server-Sent Events (SSE) protokolü üzerinde çalışan **Spekülatif Geri Çekme (Speculative Rollback)** algoritmasıdır:
1. Gateway, modelden gelen token'ları $W = 64$ karakterlik bir kayan pencerede tutar.
2. Pencere boyutu aşıldıkça, pencerenin arkasında kalan güvenli metin istemciye `TOKEN` olayı olarak akıtılır.
3. Tampon içindeki metinde bir ihlal (PII, enjeksiyon) yakalandığı anda gateway üretimi keser, **gateway'den hiç çıkmamış olan bekletilen metni atar** ve SSE hattına bir `SPECULATIVE_REVOKE` olayı gönderir.
4. İstemci bu olayı aldığında, ekrandaki yarım yanıtın yerine *"Güvenlik kuralı nedeniyle yanıt sonlandırıldı"* uyarısını koyar. O metin denetimden geçmişti; ama bekletilen bir tanımlayıcının hemen öncesinde kesilen bir cümle başlı başına bir ipucudur, bu yüzden o da geri çekilir. "Geri çekme" ekrana aittir; hassas hiçbir şey gönderilmemiştir.

---

## 6. Çıkarım zamanı yan kanal saldırıları (side-channel attacks)

TLS bir yanıtın içeriğini gizler. *Biçimini* gizlemez: her paketin ne kadar büyük olduğunu ve ne zaman geldiğini. Bir sunum katmanı için belgelenmiş iki kanal önemlidir:

- **Token uzunluğu kanalı.** Her token kendi olayında akıtıldığında, şifreli her paketin boyutu o token'ın uzunluğunu açığa vurur. Weiss ve ark. (USENIX Security 2024) tam olarak bu uzunluk dizisini kullanarak şifreli trafikten asistan yanıtlarının büyük bölümünü yeniden kurdu. Sonraki çalışmalar, yalnızca paket boyutlarının ve zamanlamalarının bir konuşmanın *konusunu* sınıflandırmaya yettiğini gösterdi.
- **Önbellek zamanlaması kanalı.** 4.2'deki önek önbelleği kehaneti: hızlı gelen ilk token, saldırgana başka birinin aynı öneki gönderdiğini söyler.

Önlemler ucuzdur ve gateway'de durur: akıtılan her olayı sabit ya da rastgele bir boyuta doldurun (padding) ve olay başına tek token yerine birkaç token'ı birlikte gönderin. İkisi de uzunluk dizisini bulanıklaştırır. Önbellek zamanlamasına karşı kontrol, kiracı başına `cache_salt`'tır; rastgele gecikmeler yalnızca saldırganın ortalamayla silebileceği bir gürültü ekler.

---

## 7. Önbellek, yeniden deneme, yedek yol ve guard çağrıları

Sistem güvenilirliğini artırmak için eklenen her yedek bileşen, dikkatle yönetilmezse bir güvenlik zaafiyetine dönüşür:

**Yedek Yollar Asla Guard'ları Atlayamaz:** Birincil modelin erişilemezlik oranı $u_{\text{primary}}$ olduğunda, korumasız bir yedek yolun kontrolsüz geçireceği trafik doğrudan bu orana eşittir:

$$
N_{\text{uncontrolled}} = N_{\text{total}} \cdot u_{\text{primary}}
$$

```text
Günde 100.000 istek alan bir sistemde birincil model erişilebilirliği %99,5 ise (u = 0,005):
    Günlük kontrolsüz istek sayısı = 100.000 × 0,005 = 500 istek!
```

Bu 500 istek rastgele dağılmaz; tam da birincil sunucunun saldırı altında olduğu kriz anlarında yoğunlaşır. Saldırgan birincil kümeyi DoS ile çökerterek trafiği korumasız yedek rotaya sürükleyebilir. Kural kesindir: Guard kancaları model düzeyine değil, Gateway'in ana giriş kapısına mühürlenir.

**Guard Çağrıları da Çıkarımdır:** Güvenlik SLM'leri veya denetleyici modeller de birer çıkarım sürecidir. Bir guard modelinin yanıt vermemesi durumunda yüksek riskli işlemlerde kesinlikle **kapalı kalarak başarısız olma (fail-closed)** ilkesi uygulanır.

---

## 8. Güvenlik kontrolü olarak kotalar, hız sınırları ve bütçe

Bir model sunucusunun kullanılabilirliği ve maliyet bütünlüğü birincil varlıktır. Klasik web uygulamalarında IP başına istek sayısı yeterli bir metrik iken; AI sistemlerinde bir istek başka bir isteğin 1.000 katı GPU kaynağı tüketebilir. Bu nedenle kotalar **token ve maliyet** cinsinden tanımlanır:

| Kota Türü | Uygulama Kapsamı | Örnek Değer |
| :--- | :--- | :--- |
| **İstek Sayısı** | İstemci / API Key başına | 60 istek / dakika |
| **Token Hacmi** | Kullanıcı oturumu başına | 40.000 girdi+çıktı token / dakika |
| **Maksimum Çıktı Sınırı** | Rota / Fonksiyon başına | Özet için 1.024 token, kod için 4.096 token |
| **Finansal Bütçe Tavanı** | Departman / Proje başına | Günlük belirlenen limit; %80'de alarm, %100'de blokaj |
| **Eşzamanlı Akış Sayısı** | Kullanıcı başına | Maksimum 3 aktif soket |

Bu kotalar, Bölüm 5'teki $1 - (1-p)^n$ ifadesinde $n$ üssü üzerindeki deterministik kontroldür: saldırganın dakikada kaç deneme yapabileceğine tavan koyarlar.

---

## 9. Pratik çıktı: Dağıtım mimarisi, kabul testleri ve Python güvenlik motoru

### 9.1. Dağıtım kabul testleri matrisi

Aşağıdaki kabul testleri, altyapıdaki hiçbir bileşenin güvenlik kontrollerini atlatamadığını sürekli entegrasyon (CI/CD) hattında doğrular:

| ID | Kabul Testi | Test Yöntemi ve Simülasyon | Geçme Ölçütü (Pass Criterion) |
| :--- | :--- | :--- | :--- |
| **S-1** | Doğrudan Sunucu İzolasyonu | Gateway dışındaki bir pod'dan vLLM uç noktalarına doğrudan TCP isteği | Ağ seviyesinde `Connection Refused` / Drop |
| **S-2** | mTLS İstemci Sertifikası | Gateway'den vLLM'e sertifikasız bağlanmayı dene | TLS Handshake Failure (403/Drop) |
| **S-3** | Açık Port Taraması | Model sunucusu alt ağını dışarıdan Nmap ile tara | Yalnızca `:8443` çıkarım portu görünür |
| **S-4** | Korumalı Yedek Rota | Birincil sunucuyu kasıtlı olarak devreden çıkarıp injection testi gönder | İstekler harici yedek sağlayıcıya akar; guardrail %100 aktif |
| **S-5** | Akışta Spekülatif İptal | 300. karakterinde sahte TCKN bulunan yanıtı akıt | TCKN kullanıcıya gösterilmeden akış kesilir; `REVOKE` üretilir |
| **S-6** | İstemci Tarafı Markdown Koruması| Model çıktısında `![exfil](https://attacker.com/leak)` gönder | UI görselleri ve linkleri render etmez; salt metin kalır |
| **S-7** | Guardrail Zaman Aşımı | Güvenlik servisini yanıt vermez hâle getir | Yüksek riskli istekler derhal 503 ile reddedilir (Fail-Closed) |
| **S-8** | Önek Önbelleği İzolasyonu | Birebir aynı prompt önekini iki kiracı kimliğiyle gönder; istemcinin seçtiği bir `cache_salt` gönder | Kiracılar arasında önbellek isabeti olmaz (isabet sayacı değişmez, TTFT kısalmaz); gateway istemcinin tuzunun üzerine yazar |
| **S-9** | Token ve Maliyet Sınırı | Belirlenen dakika/token bütçesini kasıtlı olarak aş | 429 Too Many Requests anlaşılır JSON ile döner |
| **S-10**| Statik Gizli Anahtar Taraması| Dağıtım imajlarında ve konfigürasyonlarda API anahtarı ara | CI/CD tarayıcısı sır yöneticisi dışında 0 anahtar bulur |

### 9.2. Üretim tipi Python sunum güvenlik motoru

Aşağıdaki çalıştırılabilir Python betiği; bir önek önbelleğinin kiracı başına tuzlanmasını ve bekletme pencereli bir akış guardrail'ini simüle eder:

```python
import hmac
import hashlib
import json
from typing import List, Dict, Any, Generator, Tuple

class TenantPrefixCacheManager:
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.block_cache: Dict[str, str] = {}

    def cache_salt(self, tenant_id: str) -> str:
        # Tuzu gateway doğrulanmış kiracıdan türetir; istemci asla seçmez
        return hmac.new(self.secret_key, tenant_id.encode(), hashlib.sha256).hexdigest()

    def get_or_set_prefix(self, tenant_id: str, model_id: str, tokens: List[int], kv_pointer: str) -> Tuple[bool, str]:
        # vLLM'in ilk blok özeti gibi: tuz, önbellekteki önekin anahtarına karışır
        payload = f"{self.cache_salt(tenant_id)}:{model_id}:{tokens}".encode()
        cache_key = hashlib.sha256(payload).hexdigest()
        if cache_key in self.block_cache:
            return True, self.block_cache[cache_key]
        self.block_cache[cache_key] = kv_pointer
        return False, kv_pointer

class SpeculativeStreamingGuardrail:
    def __init__(self, window_chars: int = 20):
        self.window_chars = window_chars
        self.blocked_patterns = ["TCKN:", "IBAN:", "SECRET_KEY"]

    def evaluate_stream(self, token_chunks: List[str]) -> Generator[Dict[str, str], None, None]:
        speculative_buffer = ""
        for token in token_chunks:
            speculative_buffer += token
            # Kayan pencere içinde ihlal kontrolü yap
            if any(p in speculative_buffer for p in self.blocked_patterns):
                yield {"event": "SPECULATIVE_REVOKE", "action": "ERASE_CLIENT_BUFFER", "detail": "Hassas veri deseni tespit edildi"}
                return
            # Geride tutma penceresini aşan güvenli kısmı istemciye sal
            if len(speculative_buffer) > self.window_chars:
                safe_len = len(speculative_buffer) - self.window_chars
                emitted_text = speculative_buffer[:safe_len]
                speculative_buffer = speculative_buffer[safe_len:]
                yield {"event": "TOKEN_EMIT", "text": emitted_text}
        # Akış sonunda kalan temiz kısmı boşalt
        if speculative_buffer:
            yield {"event": "TOKEN_EMIT", "text": speculative_buffer}

# Test ve Doğrulama
cache_mgr = TenantPrefixCacheManager(b"kv-master-salt-key-32b")
tokens = [1, 512, 1024, 2048]
MODEL = "mistral-7b-instruct-v0.3"
hit_a, ptr_a = cache_mgr.get_or_set_prefix("tenant-alpha", MODEL, tokens, "kv-block-0x1A")
hit_b, ptr_b = cache_mgr.get_or_set_prefix("tenant-beta", MODEL, tokens, "kv-block-0x9F")
hit_a2, ptr_a2 = cache_mgr.get_or_set_prefix("tenant-alpha", MODEL, tokens, "kv-block-0x2C")

print("--- 1. Multi-Tenant Prefix Cache Isolation ---")
print(f"Tenant-A first request, cache hit: {hit_a} (Block: {ptr_a})")
print(f"Tenant-B same prefix tokens, cache hit: {hit_b} (Block: {ptr_b})")
print(f"Tenant-A same prefix again, cache hit: {hit_a2} (Block: {ptr_a2})")

guard = SpeculativeStreamingGuardrail(window_chars=16)
incoming_tokens = ["Sayın çalışan, ", "ilgili müşterinin ", "TCKN: 10987654321", " numaralı hesabına iade yapılmıştır."]
print("\n--- 2. Speculative Streaming Guardrail ---")
for event in guard.evaluate_stream(incoming_tokens):
    print(f"Event: {event['event']} -> {event.get('text', event.get('detail'))}")
```

Program çalıştırıldığında üretilen çıktı:

```text
--- 1. Multi-Tenant Prefix Cache Isolation ---
Tenant-A first request, cache hit: False (Block: kv-block-0x1A)
Tenant-B same prefix tokens, cache hit: False (Block: kv-block-0x9F)
Tenant-A same prefix again, cache hit: True (Block: kv-block-0x1A)

--- 2. Speculative Streaming Guardrail ---
Event: TOKEN_EMIT -> Sayın çalışan, il
Event: SPECULATIVE_REVOKE -> Hassas veri deseni tespit edildi
```

---

## Bölümün pedagojik durakları

### Kendinizi sınayın

**Soru 1:** vLLM model sunucusunu `--api-key` bayrağı ile çalıştırmak sunucunun tüm HTTP ve RPC yollarını tam olarak korumak için neden yeterli değildir?
- A) API anahtarları yalnızca tek kullanımlık olduğu için.
- B) `--api-key` yalnızca `/v1`, `/v2`, `/inference` ve `/cohere` öneklerini korur; `/invocations`, `/tokenize`, `/pause` gibi yolları ve düğümler arası portları korumaz.
- C) vLLM açık kaynak kodlu olduğu için API anahtarlarını şifreleyemez.
- D) vLLM yalnızca UDP trafiğini şifreleyebilir.
*(Doğru Cevap: B. Belgelerde açıklandığı üzere API anahtarı yönetimsel ve yardımcı yolları kapsamaz; ters vekil ve ağ politikası zorunludur.)*

**Soru 2:** NVIDIA H100 gizli bilişiminde veri, gizli sanal makine ile GPU arasındaki PCIe veriyolundan geçerken nasıl korunur?
- A) Korunmaz; veriyolu güvenilir kabul edilir.
- B) Sürücü ve GPU bir SPDM oturumunda anahtar türetir ve veriyi şifreli bir sıçrama tamponundan (bounce buffer) geçirir.
- C) GPU belleği (HBM) şifrelidir, bu yüzden veriyolunun korunmasına gerek yoktur.
- D) Hypervisor veriyi misafir adına şifreler.
*(Doğru Cevap: B. Bağlantı düzeyindeki PCIe IDE, yeni platformların üzerine kurulduğu standarttır; paket üstü HBM'in kendisi şifrelenmez.)*

**Soru 3:** Akış yapan (streaming) bir LLM çıktısında `post_call` guardrail kullanmanın temel yapısal güvenlik zaafiyeti nedir?
- A) Modelin bellek tüketimini iki katına çıkarması.
- B) Kontrolün ancak tüm token'lar istemciye ulaştıktan sonra tamamlanması; bu nedenle sızıntının ekrana çoktan düşmüş olması ve engellenememesi.
- C) Token üretim hızını saniyede 1 token'a düşürmesi.
- D) Ağ paketlerinin kaybolmasına yol açması.
*(Doğru Cevap: B. Akışta post-call yalnızca denetim amaçlıdır; önlemek için spekülatif tamponlama şarttır.)*

**Soru 4:** Çok kiracılı bir vLLM sunucusu önek önbelleğini kiracı başına `cache_salt` olmadan paylaştırırsa hangi risk doğar?
- A) GPU çekirdekleri aşırı ısınarak kapanır.
- B) Önbellek isabeti ilk token süresini kısaltır; bir kiracı yanıt sürelerini ölçerek başka bir kiracının prompt öneki hakkındaki tahminlerini sınayabilir.
- C) Modelin parametre sayısı kendiliğinden azalır.
- D) LLM yalnızca İngilizce yanıt vermeye başlar.
*(Doğru Cevap: B. Önbellek içerik adreslidir, yabancı içerik sunulmaz; sızan şey isabetin zamanlamasıdır, CVE-2025-46570.)*

### Düşünce deneyi

**Gizli kipteki bir GPU'ya karşı PCIe ara aygıtı:**
Bir siber casusluk grubu, hedef şirketin veritabanı sunucularının barındığı ortak yerleşimli (co-location) veri merkezine sızıyor. Sunucu kabinini açarak H100 GPU kartının takılı olduğu PCIe x16 yuvasına yüksek hızlı bir donanımsal veri kaydedici (PCIe bus analyzer / interposer) yerleştiriyor.

*Soru:* Şirket standart çıkarım yapıyorsa saldırganlar ne elde eder? GPU gizli kipteyse ne elde ederler?
*Cevap:* Standart çıkarımda CPU ana belleğinden GPU'ya akan tüm model ağırlıkları, kullanıcı prompt'ları ve açık KV cache sayfaları PCIe hattından düz metin tensörler olarak geçer; saldırganlar tüm şirket sırlarını anında kaydeder. Gizli kipte ise veriyolundan geçen yükler, gizli sanal makinenin içindeki sürücü tarafından SPDM oturumundan gelen anahtarlarla şifrelenmiştir; kaydedilen şey şifreli metindir. Ara aygıtın değiştirmediği şey kalan risktir: GPU paketindeki HBM şifrelenmez; paketin kendisini yoklayabilen bir saldırgan bu tasarımın savunduğu alanın dışındadır.

### Kavramları ayrıştırma

- **mTLS vs veriyolu şifrelemesi:** mTLS, iki yazılım uç noktası (gateway ile vLLM pod'u) arasındaki ağ trafiğini şifreler. Veriyolu şifrelemesi ise CPU TEE ile GPU arasındaki veriyi korur: H100'de SPDM anahtarlı şifreli sıçrama tamponlarıyla, yeni platformlarda PCIe IDE ile bağlantının kendisinde. Biri ağ dinlemesine, diğeri düşman bir ana makineye ya da veriyolu sondasına yanıttır.
- **Yalnızca Denetim (Audit-Only) vs Spekülatif Geri Çekme (Speculative Rollback):** Yalnızca denetim, sızıntı gerçekleştikten sonra log tutar. Spekülatif geri çekme ise token'ları kısa bir kayan pencerede askıya alır; ihlal anında gönderilmemiş metni atar ve istemciye yarım yanıtı geri çekmesini bildirir.

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

1. Neden kurumsal bir mimaride uygulama kodunun doğrudan vLLM sunucusuna bağlanmasına asla izin verilmemelidir?
2. NVIDIA H100 Confidential Computing ve SPDM kanıtlaması, model ağırlıklarının kötü niyetli bulut yöneticilerinden çalınmasını nasıl engeller?
3. vLLM'in `--api-key` bayrağı hangi yolları korumasız bırakır ve bu yollar nasıl kapatılır?
4. Akış yanıtlarında ilk token gecikmesini (TTFT) bozmadan yapılandırılmış PII sızıntıları nasıl engellenir?
5. Birincil model çöktüğünde devreye giren yedek rotalar nasıl tasarlanmalıdır ki sistem güvenlik kontrollerini atlatmasın?

---

## Bütün hikâye altı satırda

- Saldırganlar güvenlik kontrollerinin etrafından dolaşabiliyorsa onları aşmak zorunda kalmaz; sunum katmanı her yolu tavizsiz kontrol eder.
- Gateway doğrular, kotaları uygular ve yönlendirir; model sunucusu yalnızca izole mTLS bağlantısıyla gateway'e yanıt verir.
- GPU gizli bilişimi cihazı kanıtlar, bir SPDM oturumu anahtarlar ve veriyolundan geçeni şifreler; paket üstü HBM şifrelenmez, PCIe IDE ise yeni platformların benimsediği bağlantı düzeyi standarttır.
- vLLM'in API anahtarı dört öneki kapsar, operasyonel yolları açık bırakır; paylaşılan önek önbelleği zamanlama üzerinden sızdırır, bu yüzden gateway kiracı başına `cache_salt` koyar.
- Akışta post-call kontroller sızıntıyı durduramaz; bir bekletme penceresi yapılandırılmış PII'ı gateway'den çıkmadan yakalar.
- Korumasız yedek yollar erişilemezlik anında trafiği savunmasız bırakır; token ve para kotaları hizmet sürekliliğini teminat altına alır.

---

## Terimler sözlüğü

- **AI Gateway:** İstemcilerin kimliğini doğrulayan, kotaları ve hız sınırlarını uygulayan, istekleri yönlendiren ve guardrail kancalarını yöneten tek giriş kapısı.
- **Model Sunucusu:** vLLM gibi büyük dil modellerini optimize tensör motorlarıyla GPU üzerinde çalıştıran çıkarım motoru.
- **Confidential Computing (Gizli Bilişim):** Verilerin işlenme anında (in-use) donanımsal olarak izole edilmiş Güvenilir Yürütme Ortamlarında (TEE) şifreli tutulması standardı.
- **PCIe IDE (Integrity and Data Encryption):** Bağlantı düzeyinde şifreleme için PCIe standardı (AES-GCM). TDISP'li platformlar CPU TEE ile cihaz arasındaki trafiği bununla korur; H100 ise aktarımı sürücüde, SPDM oturum anahtarlarıyla şifreler.
- **SPDM (Security Protocol and Data Model):** Donanım aygıtlarının kimliğini ve firmware bütünlüğünü sertifikalarla kanıtlayan standart güvenlik protokolü.
- **Önek Önbelleği ve `cache_salt`:** vLLM, birebir aynı prompt önekleri için KV bloklarını zincirli blok özetleriyle tanıyıp yeniden kullanır; ilk özete karışan kiracı başına `cache_salt`, kiracıların birbirinin öneklerini paylaşmasını ve zamanlamasını engeller.
- **Spekülatif Geri Çekme (Speculative Rollback):** Bekletme penceresinde bir ihlal görüldüğünde gateway'in gönderilmemiş metni atması ve istemciye ekrandaki yarım yanıtı geri çekmesini bildirmesi.
- **TTFT (Time-to-First-Token):** Kullanıcının isteği göndermesiyle modelin ilk token'ı üretmesi arasında geçen süre.
- **Yan Kanal Saldırısı (Side-Channel Attack):** Şifreli verinin kendisini kırmak yerine, paket boyutları ve token üretim sürelerindeki zamanlama desenlerinden veri çıkarımı yapılması.
- **Fail-Closed (Kapalı Başarısız Olma):** Bir güvenlik kontrolü zaman aşımına uğradığında veya çöktüğünde işlemin güvenliği korumak amacıyla reddedilmesi ilkesi.

---

## Daha derine inmek için

- Önceki bölüm: [Secure AI by Design (8): AI Ajanlarında En Az Yetki ve Güvenli Araç Kullanımı](post.html?slug=ajanlarda-en-az-yetki)
- Sonraki bölüm: [Secure AI by Design (10): AI Güvenliğini Ölçmek, Red Teaming ve Türkçe Değerlendirme](post.html?slug=ai-guvenligini-olcmek)
- NVIDIA, [*Confidential Computing on NVIDIA H100 GPUs for Secure and Trustworthy AI*](https://developer.nvidia.com/blog/confidential-computing-on-h100-gpus-for-secure-and-trustworthy-ai/) ve [*Hardware-Rooted Trust with SPDM*](https://docs.nvidia.com/confidential-computing/).
- vLLM, [*Security Best Practices*](https://docs.vllm.ai/en/latest/usage/security.html) ve [*Automatic Prefix Caching Architecture*](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html).
- LiteLLM, [*Custom Guardrail Implementation and Streaming Hooks*](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail).
- OWASP, [*API Security Top 10 (2023)*](https://owasp.github.io/www-project-api-security/) (API4, API9) ve [*Top 10 for LLM Applications 2026*](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10) (LLM06, LLM10).
- R. Weiss ve ark., [*What Was Your Prompt? A Remote Keylogging Attack on AI Assistants*](https://arxiv.org/abs/2403.09751) (USENIX Security 2024).
- OWASP, [*AISVS 1.0*](https://owasp.github.io/www-project-artificial-intelligence-security-verification-standard-aisvs-docs/), C4 (Altyapı ve Dağıtım Güvenliği).
- Bu blogdan: Çıkarım motoru optimizasyonları için [vLLM derinlemesine mimarisi](post.html?slug=vllm-derinlemesine) ve [LLM maliyet ve gecikme optimizasyonu](post.html?slug=llm-maliyet-ve-gecikme-optimizasyonu).
