*Bir GPU’nun içindeki 32 thread’den 16.384 GPU’luk eğitim kümesine, oradan LLM/VLM sistemlerine: hesaplamayı, belleği, ağı, depolamayı, güvenliği ve ölçeklendirmeyi sezgisel örneklerle, matematikle ve yayımlanmış teknik çalışmalar üzerinden birlikte anlamaya çalışalım.*

Bir veri merkezine girdiğinizi hayal edin. Karşınızda sıra sıra sunucular, her birinin içinde binlerce paralel işlemi gerçekleştirebilen GPU'lar var. Toplamda **1.024 NVIDIA H100** satın aldığınızı varsayalım. Yalnızca BF16 yoğun Tensor Core hesapları açısından, kâğıt üzerinde saniyede yaklaşık **1 exaFLOP**'luk bir tepe güce sahipsiniz.

Ekip heyecanla büyük dil modelinin eğitimini başlatıyor. İlk ölçümler geliyor: **Model FLOPs Utilization (MFU) = %38**.

Birisi soruyor: “Bu kadar GPU aldık, peki kalan %62 nerede?”

Bu çok iyi bir soru. Fakat cevabı “GPU'lar boş” kadar basit değil. Bir GPU hesaplama yaparken bellekten veri bekleyebilir. Başka bir GPU diğer sunucudan gelecek gradyana ihtiyaç duyabilir. Bir başka aşama ise checkpoint'in depolamaya yazılmasını bekliyor olabilir. Hatta GPU tamamen meşgul görünürken modelin faydalı matematiksel işi beklediğinizden az gerçekleşiyor olabilir.

**Bu yazının temel fikri şu:** Büyük ölçekli yapay zekâ, yalnızca *hesaplama gücü* problemi değildir. Aynı zamanda **veriyi doğru yere, doğru zamanda, doğru güvenlik sınırları içinde ve mümkün olan en düşük maliyetle ulaştırma** problemidir.

Bunu gerçekten anlamak için önce GPU'nun içine bakacağız. Thread nedir, warp neden 32 kişilik bir gruptur, Tensor Core ne yapar, HBM neden bu kadar kritiktir? Daha sonra aynı fikirlerin sekiz GPU'lu DGX sunucusuna, ardından binlerce GPU'luk veri merkezine nasıl taşındığını göreceğiz. En sonunda tek bir eğitim adımını baştan sona izleyecek ve “MFU neden düşük?” sorusuna kanıta dayalı cevap arayacağız.

> **Nasıl okuyabilirsiniz?** Bir kavram ilk kez geçtiğinde önce ne işe yaradığını ve basit bir benzetmesini vereceğim. Ardından teknik tanımını, mümkünse küçük bir hesaplamasını göstereceğim. Tablolar ve sayılar ezberlemek için değil; sistemin davranışını zihinde canlandırabilmek için var.

**Bu yazının amacı bilgi paylaşımı.** Bu, belirli bir ürünün kurulum kılavuzu, hazır mimari reçetesi veya bütün arızaları çözen bir işletim kitabı değil. GPU’dan kümeye uzanan temel mekanizmaları, model iş yüklerini ve bunların kapasite, güvenilirlik, güvenlik, kullanıcı deneyimi ve maliyetle ilişkisini aynı öğrenme hikâyesinde ele alıyoruz. Tablolar, örnek test akışları ve komutlar anlatılan kavramları somutlaştırmak içindir; gerçek bir sisteme doğrudan uygulanacak doğrulanmış prosedürler olarak okunmamalıdır.

### Birlikte izleyeceğimiz öğrenme yolculuğu

1. **GPU'nun içinde:** Thread → Warp → SM → Tensor Core; GPU hesaplamayı nasıl organize eder?
2. **GPU'nun belleğinde:** Register → Shared Memory → L2 → HBM; bir işlem veriyi neden bekler?
3. **LLM'in çalışma biçiminde:** Training, prefill, decode, KV cache, quantization; aynı model niçin farklı darboğazlara girer?
4. **Tek sunucudan büyük kümeye:** NVLink, NVSwitch, NIC, InfiniBand/RoCE, ağ topolojileri; GPU'lar nasıl konuşur?
5. **Dağıtık eğitimde:** TP, PP, DP, FSDP, CP, EP; modeli nereye ve nasıl böleriz?
6. **Uçtan uca:** Bir training step, depolama, checkpoint, arıza yönetimi ve MFU teşhisi.
7. **Gerçek performans kazanımı:** B200/GB300 üzerinde yayımlanmış MFU benchmark’ları, FP8, kernel fusion, overlap ve kendi kümenizde uygulayabileceğiniz ölçüm planı.
8. **Mimarinin bütününü anlamak:** Üretim inference, orkestrasyon, veri yönetişimi, güvenilirlik, SLO, enerji ve maliyetin birbiriyle ilişkisi.
9. **Mühendislik örnekleri:** GPU/serving kapasitesinin nasıl hesaplandığı; ağ testlerinde ne ölçüldüğü; VLM/VQA, RAG/agent ve güvenlik konularındaki tipik ödünleşimler.
10. **Bütünlüklü örnek:** 70B model üzerinden varsayımsal bir mimariyi okumak, operasyonel katmanları tartışmak ve küçük hesap deneyleri yapmak.

> **Sayısal doğruluk notu:** H100'le ilgili özellikler aksi belirtilmedikçe **H100 SXM** sürümü ve ilgili DGX H100 referans mimarisi içindir. “Teorik tavan”, “öğretici varsayım” ve “gerçek raporlanmış ölçüm” birbirinden farklıdır. Yazının sonunda [K1]–[K51] kaynaklarını bulacaksınız. Makale, [Understanding GPUs and LLM Scaling](https://muhendis.github.io/en/blog/post.html?slug=understanding-gpus-and-llm-scaling) bağlamıyla ve önceki Türkçe taslakla birleştirilerek yeniden kaleme alınmıştır.

---

### Bu yazının yöntemi: Ezber yerine zihinsel model

Büyük GPU kümeleri hakkında okurken insanın aklında birbirinden kopuk kelimeler kalabiliyor: *warp, SM, HBM, NCCL, MFU, FSDP, KV cache...* Bunları tek tek tarif etmek kolay; **neden birbirine ihtiyaç duyduklarını** anlamak daha değerli. O yüzden bu metinde aynı soruya sürekli geri döneceğiz: **Bir byte veri nereden geliyor, hangi işlemi mümkün kılıyor, sonra nereye gidiyor ve o yolculukta kim kimi bekliyor?**

İki ayrı hikâye izleyeceğiz. Birincisinde model eğitiyoruz: veri okunuyor, forward hesaplanıyor, hata bulunuyor, backward ile gradyanlar hesaplanıyor, GPU'lar senkronize oluyor ve model güncelleniyor. İkincisinde model bir kullanıcıya yanıt veriyor: istek sıraya giriyor, prompt *prefill* aşamasından geçiyor, ardından *decode* sırasında token'lar sırayla üretiliyor. Aynı GPU, bu iki hikâyede birbirinden çok farklı sınırlara çarpabiliyor.

Bu nedenle her önemli konuyu dört basamakla ele alacağız:

1. **Sezgi:** Tanıdık bir olay üzerinden problemi hissedelim.
2. **Mekanizma:** Donanımda veya yazılımda gerçekte hangi bileşenler çalışıyor?
3. **Matematik:** İlgili darboğazı hangi basit denklem görünür hâle getiriyor?
4. **Sınır:** Denklemin ve örneğin nerede yetersiz kaldığını konuşalım.

Makalede kullanılan **üretici tepe değerleri**, **araştırma makalelerindeki raporlanmış deneyler** ve **öğretici hesap örnekleri** birbirine karıştırılmıyor. Bir teknoloji farklı donanım neslinde veya farklı modelde başka sonuç verebilir. Anlatının amacı tek bir doğru mimari dayatmak değil, sistemin davranışını anlayacak ortak bir dil kurmak.

### Hikâyenin tek cümlelik haritası

<svg viewBox="0 0 720 240" role="img" aria-label="Neden-sonuç haritası. Bir byte depolamadan CPU ve host RAM üzerinden HBM&#x27;e, oradan SM ve Tensor Core&#x27;lara gider ve loss ya da sonraki token&#x27;ı üretir. Depolama ile CPU arasında I/O duvarı, HBM ile SM arasında bellek duvarı vardır. Birden çok GPU varsa sonuçlar NVLink veya InfiniBand kolektifleriyle tekrar HBM&#x27;e döner: bu da ağ duvarıdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="mp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="mp-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker><marker id="mp-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir byte&#x27;ın yolculuğu ve çarpabileceği üç duvar</text>
<rect x="16" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="76.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Depolama</text>
<text x="76.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">veri, checkpoint</text>
<rect x="158" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="218.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU + host RAM</text>
<text x="218.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">DataLoader, PCIe</text>
<rect x="300" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">HBM3</text>
<text x="360.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80 GB · 3,35 TB/s</text>
<rect x="442" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="502.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">SM + Tensor Core</text>
<text x="502.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">~990 TFLOP/s BF16</text>
<rect x="584" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="644.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Sonuç</text>
<text x="644.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">loss / sonraki token</text>
<line x1="136" y1="71" x2="156" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="278" y1="71" x2="298" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="420" y1="71" x2="440" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="562" y1="71" x2="582" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="107.12" y="104" width="79.76" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="147.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">I/O duvarı</text>
<rect x="380.956" y="104" width="100.088" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="431.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Bellek duvarı</text>
<line x1="522" y1="98" x2="522" y2="134" marker-end="url(#mp-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="330" y="136" width="210" height="46" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="435.0" y="156.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">NVLink / InfiniBand</text>
<text x="435.0" y="171.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kolektifler (AllReduce …)</text>
<line x1="360" y1="136" x2="360" y2="100" marker-end="url(#mp-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="530" y="122" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">birden çok GPU mu?</text>
<rect x="398.508" y="190" width="72.984" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="435.0" y="202.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Ağ duvarı</text>
<text x="16" y="228" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her bölüm aynı soruyu sorar: bu byte nerede bekliyor ve onu kim bekliyor?</text>
</svg>

Bu şema eksiksiz bir donanım topolojisi değil, okurken başvurabileceğimiz bir **neden–sonuç haritası**. Metnin sonunda her okun teknik karşılığını birlikte çözebilir duruma gelmiş olacağız.




---

**KISIM I — GPU neden hızlıdır, neden bekler?**

*Önce hesaplamanın en küçük birimini, sonra verinin GPU içinde dolaşmasını anlamaya çalışacağız.*


## 1. Üç fiziksel duvar: Compute, Network, Storage

Bir restoran zinciri düşünelim. Yönetim, üretimi artırmak için her şubeye iki kat fazla aşçı gönderiyor. Ama şubelerdeki malzeme depoları aynı; dağıtım kamyonları aynı; mutfak kapılarından malzeme geçişi de aynı. Acaba iki kat fazla yemek üretilebilir mi?

Muhtemelen hayır. Çünkü bu sistemde darboğaz artık aşçı sayısı değil. Malzeme ulaşana kadar dünyanın en iyi aşçısı bile bekler. İşte dev bir GPU kümesindeki sorun da çoğu zaman bu kadar gündeliktir.

Bir yapay zekâ kümesini restoran zinciri gibi düşünün:

- **Streaming Multiprocessor (SM):** Yemek hazırlayan küçük mutfak istasyonu.
- **Tensor Core:** Çok sayıda benzer matematiksel işi aynı anda yapan özel makine.
- **Register / Shared Memory:** Aşçının elinin altındaki tezgâh.
- **HBM:** Restoranın hızlı fakat sınırlı ortak deposu.
- **NVLink / NVSwitch:** Aynı binadaki mutfaklar arasındaki geniş servis koridoru.
- **InfiniBand / RoCE:** Farklı binalar arasındaki lojistik yolları.
- **Dağıtık depolama:** Hammaddenin ve işletmenin kayıtlarının tutulduğu merkez depo.

Bu benzetme bize çok önemli bir şey söylüyor: GPU sayısı, tek başına sistem performansının belirleyicisi değildir. Üretimin farklı katmanları farklı hızlarda çalışır ve bu hızlar arasında uyumsuzluk varsa bir taraf diğerini bekler.

Mutfak sayısını artırmak yemek sayısını artırmayabilir. Malzeme kapıdan geçemiyorsa **bellek duvarına**, mutfaklar sürekli birbirine danışıyorsa **ağ duvarına**, malzeme veya kayıt sistemi yetişemiyorsa **I/O duvarına** çarparsınız.

| Duvar | Sorulan asıl soru | Tipik belirti | İlk müdahale |
|---|---|---|---|
| **Memory Wall** | Veriyi taşımak mı hesaplamaktan pahalı? | HBM trafiği yüksek, FLOP/s düşük | Daha çok veri yeniden kullanımı; batching, tiling, fusion, düşük-bit ağırlık |
| **Fabric Wall** | GPU'lar konuşmak için ne kadar zaman harcıyor? | Uzun NCCL collective, idle pipeline evreleri | Topolojiyle uyumlu TP/PP/DP, iletişim örtüştürme |
| **I/O Wall** | Token'lar veya model kayıtları yetişiyor mu? | Dataloader beklemeleri, checkpoint durmaları | Prefetch, shard edilmiş I/O, async checkpoint, uygun GDS |

### 1.1. Fiziksel ölçek merdiveni

**Önce doğru ölçeğe bakalım.**

Bir CUDA kernel'i register sıkıntısından yavaşlıyorsa yeni InfiniBand switch'i almak o kernel'i hızlandırmaz. Tersine, iki sunucu arasındaki ağ darboğazı yüzünden eğitim bekliyorsa SM'deki birkaç register'ı azaltmak da çözüm olmayabilir. O yüzden önce sistemin *hangi büyüklükteki parçasının* ne yaptığını ayıracağız.

<svg viewBox="0 0 720 388" role="img" aria-label="Aşağıdan yukarı ölçek merdiveni: thread; 32 thread&#x27;lik warp; en fazla 64 warp ya da 2.048 thread barındıran SM; 132 SM, 80 GB HBM3 ve 3,35 TB/s&#x27;lik H100 SXM; NVSwitch&#x27;li 8 GPU&#x27;luk DGX H100; 32 DGX yani 256 GPU&#x27;luk SU; 4 SU ve 127 DGX yani 1.016 GPU&#x27;luk SuperPOD; Llama 3 405B&#x27;deki gibi 16.384 H100&#x27;lük çoklu pod kümesi. Her basamağın yanında tipik darboğazı yazar: dal ayrışması, gecikme gizleme, register ve shared memory, HBM bant genişliği, NVLink&#x27;te TP kolektifleri, leaf switch ve rail&#x27;ler, InfiniBand&#x27;de DP ve PP, en üstte depolama, arıza ve güç." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Ölçek merdiveni: her basamak alttakini çoğaltır</text>
<text x="704" y="24" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">tipik darboğaz burada yaşar</text>
<rect x="70" y="44" width="386" height="32" rx="6" style="fill:var(--c-danger);fill-opacity:.10;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="82" y="64" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Çoklu pod kümesi</text>
<text x="210.4" y="64" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">16.384 H100 (ör. Llama 3 405B, RoCE)</text>
<text x="60" y="64" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)"></text>
<line x1="462" y1="60" x2="494" y2="60" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12px">depolama, arıza, güç</text>
<rect x="80" y="84" width="376" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="92" y="104" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">SuperPOD</text>
<text x="163.2" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">4 SU · 127 DGX = 1.016 GPU</text>
<text x="70" y="104" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×4</text>
<line x1="462" y1="100" x2="494" y2="100" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="104" text-anchor="start" style="fill:var(--c-text);font-size:12px">InfiniBand&#x27;de DP/PP</text>
<rect x="90" y="124" width="366" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="102" y="144" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Scalable Unit</text>
<text x="208.95" y="144" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">32 DGX = 256 GPU</text>
<text x="80" y="144" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×32</text>
<line x1="462" y1="140" x2="494" y2="140" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="144" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf switch, rail&#x27;ler</text>
<rect x="100" y="164" width="356" height="32" rx="6" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="112" y="184" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX H100</text>
<text x="183.2" y="184" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">8 GPU · NVSwitch · 900 GB/s NVLink</text>
<text x="90" y="184" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×8</text>
<line x1="462" y1="180" x2="494" y2="180" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="184" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP kolektifleri (NVLink)</text>
<rect x="110" y="204" width="346" height="32" rx="6" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="122" y="224" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">H100 SXM</text>
<text x="193.2" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">132 SM · 80 GB HBM3 · 3,35 TB/s</text>
<text x="100" y="224" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×132</text>
<line x1="462" y1="220" x2="494" y2="220" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="224" text-anchor="start" style="fill:var(--c-text);font-size:12px">HBM bant genişliği (Roofline)</text>
<rect x="120" y="244" width="336" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="132" y="264" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">SM</text>
<text x="160.3" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">≤ 64 warp = 2.048 thread</text>
<text x="110" y="264" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×64</text>
<line x1="462" y1="260" x2="494" y2="260" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="264" text-anchor="start" style="fill:var(--c-text);font-size:12px">register, shared memory</text>
<rect x="130" y="284" width="326" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="142" y="304" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Warp</text>
<text x="184.6" y="304" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">32 thread, ortak komut akışı</text>
<text x="120" y="304" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×32</text>
<line x1="462" y1="300" x2="494" y2="300" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="304" text-anchor="start" style="fill:var(--c-text);font-size:12px">gecikme gizleme</text>
<rect x="140" y="324" width="316" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152" y="344" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Thread</text>
<text x="208.9" y="344" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">tek yürütme örneği</text>
<text x="130" y="344" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">1</text>
<line x1="462" y1="340" x2="494" y2="340" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="344" text-anchor="start" style="fill:var(--c-text);font-size:12px">dal ayrışması</text>
<text x="16" y="376" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">1.024 = 128 × 8, 4 SU&#x27;nun tasarım kapasitesidir; referans SuperPOD bir yuvayı UFM&#x27;e verir: 127 DGX = 1.016 GPU.</text>
</svg>

**1.024 mü, 1.016 mı?** Dört adet 32 sunuculuk SU, 128 × 8 = **1.024 GPU'luk tasarım kapasitesi** ifade eder. NVIDIA'nın belirli DGX H100 SuperPOD referansında ağ yönetimi için bir sunucu konumunun ayrılmasıyla **127 DGX = 1.016 GPU** bulunur. Bu iki rakamı aynı kurulumun gerçekleşmiş GPU sayısı gibi kullanmak yanlıştır. [K4]

Bu hiyerarşiyi öğrendiğimizde 'hangi sorun hangi seviyede çözülür?' sorusunu da yanıtlayabiliriz. Register baskısı tek SM'de, TP maliyeti NVLink alanında, gradyan senkronizasyonu çoğu zaman sunucular arası ağda, checkpoint krizi ise depolama sisteminde yaşanabilir.

---


## 2. GPU'nun içinde ne var? Thread, Warp, SM ve Tensor Core

### 2.1. CPU neden GPU gibi çalışmaz?

Bir GPU'ya “çok güçlü işlemci” demek yanlış değil ama eksik. GPU'nun gücü tek bir işi olağanüstü hızlı bitirmesinde değil; **çok sayıda benzer işi aynı anda yürütebilmesinde**. İşte burada CPU'yla arasındaki fark belirginleşiyor.

CPU, nispeten az sayıdaki güçlü çekirdekle dallanması bol, değişken ve gecikmeye duyarlı görevleri yürütmek için tasarlanmıştır. GPU ise binlerce benzer aritmetik işlemini yüksek throughput ile gerçekleştirmeye odaklanır.

Bu farkı aynı problemi iki farklı ekibe vermek gibi düşünün. CPU'da birkaç deneyimli uzman, görevleri aralarında sıralar. GPU'da ise çok sayıda işçi, benzer parçaları aynı anda işler. İkincisi özellikle büyük matris çarpımında etkilidir; fakat iş küçük veya düzensizse koordinasyon maliyeti kazanımı azaltabilir.

Aynı karşılaştırmayı Google'ın TPU'suyla yapmak da öğreticidir. Bir TPU çipinde az sayıda büyük TensorCore ve bunların içinde 128 × 128'lik sistolik matris birimleri (MXU) bulunur: iş büyük ve düzenli matrisler hâlinde geldiğinde bu dev tezgâhlar çok verimlidir, küçük ya da düzensiz işte boş kalırlar. H100 ise kendi warp'larını kendisi planlayan 132 bağımsız SM'den oluşan bir ekiptir. Çipler arasında da TPU'lar sabit 2D/3D torus bağlantılarıyla çalışırken GPU kümesi NVLink alanı, leaf–spine ağı ve depolama ağından oluşan **katmanlı** bir hiyerarşidir; yazının geri kalanı hangi katmanda neyin beklediğini anlatıyor. [K11]

### 2.2. CUDA thread, warp ve SIMT

Önce en küçük mantıksal işçiyi tanıyalım. Büyük bir görüntünün her pikseline aynı filtreyi uyguladığınızı düşünün. İşi tek bir görev gibi yazarsınız; GPU bunu çok sayıda küçük yürütme örneğine ayırabilir. Her yürütme örneği kendi pikseli üzerinde çalışır. Bir işçi, tek bir piksel; 32 işçi, bir warp…

**Thread**, GPU'da bir programın tek mantıksal yürütme örneğidir. **Warp**, NVIDIA mimarisinde birlikte planlanan 32 thread'lik gruptur. **SIMT (Single Instruction, Multiple Threads)**, thread'lerin kendi veri ve durumlarına sahip olabildiği, ancak warp düzeyinde komut yürütme mantığının koordine edildiği modeldir.

<svg viewBox="0 0 720 392" role="img" aria-label="Üstte: 128 thread&#x27;lik bir block, 32&#x27;şer thread&#x27;lik dört warp&#x27;a (warp 0–3) bölünür. Block tek bir SM&#x27;de çalışır; SM, register, shared memory ve warp yuvası yettiği kadar birden çok block barındırabilir. Altta: warp divergence. &#x27;x 0&#x27;dan küçükse A, değilse B&#x27; kodunda aynı warp önce 16 şerit etkin, 16 şerit maskeli olarak A yolunu, sonra diğer 16 şeritle B yolunu yürütür; bu eşit maliyetli örnekte her geçişte şeritlerin yarısı boştadır, ama bu evrensel bir %50 kayıp değildir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="bw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="bw-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Thread block, 32&#x27;lik warp&#x27;lara bölünür</text>
<rect x="16" y="38" width="430" height="176" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">thread block (128 thread)</text>
<text x="28" y="87" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 0</text>
<rect x="92" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="102" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="112" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="122" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="132" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="142" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="152" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="162" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="172" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="182" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="192" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="202" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="212" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="222" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="232" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="242" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="252" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="272" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="282" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="292" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="302" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="312" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="322" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="332" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="342" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="352" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="362" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="372" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="382" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="402" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="28" y="121" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 1</text>
<rect x="92" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="102" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="112" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="122" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="132" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="142" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="152" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="162" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="172" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="182" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="192" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="202" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="212" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="222" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="232" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="242" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="252" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="262" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="272" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="282" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="292" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="302" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="312" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="322" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="332" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="342" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="352" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="362" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="372" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="382" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="392" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="402" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="28" y="155" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 2</text>
<rect x="92" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="102" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="112" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="122" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="132" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="142" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="152" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="162" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="172" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="182" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="192" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="202" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="212" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="222" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="232" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="242" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="252" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="262" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="272" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="282" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="292" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="302" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="312" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="322" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="332" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="342" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="352" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="362" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="372" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="382" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="392" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="402" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="28" y="189" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 3</text>
<rect x="92" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="102" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="112" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="122" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="132" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="142" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="152" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="162" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="172" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="182" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="192" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="202" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="212" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="222" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="232" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="242" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="252" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="262" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="272" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="282" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="292" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="302" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="312" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="322" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="332" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="342" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="352" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="362" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="372" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="382" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="392" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="402" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<text x="92" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her kare = bir thread · warp başına 32</text>
<line x1="448" y1="126" x2="478" y2="126" marker-end="url(#bw-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="480" y="38" width="224" height="176" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="492" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">tek SM</text>
<rect x="492" y="70" width="200" height="34" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="592" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">block 0</text>
<rect x="492" y="112" width="200" height="34" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="592" y="133" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">block 1</text>
<rect x="492" y="154" width="200" height="34" rx="5" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="592" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">+ kaynak yettikçe fazlası</text>
<text x="592" y="204" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">register · shared memory · warp yuvası</text>
<text x="16" y="252" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Warp divergence: tek warp, iki yol, sırayla</text>
<text x="16" y="276" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">if (x &lt; 0) { A } else { B }</text>
<text x="16" y="307" text-anchor="start" style="fill:var(--c-text);font-size:12px">1. geçiş: if → A</text>
<rect x="176" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="188" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="200" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="212" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="224" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="236" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="248" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="260" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="272" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="284" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="296" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="308" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="320" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="332" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="344" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="356" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="368" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="380" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="392" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="404" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="416" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="428" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="440" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="452" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="464" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="476" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="488" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="500" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="512" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="524" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="536" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="548" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<text x="16" y="341" text-anchor="start" style="fill:var(--c-text);font-size:12px">2. geçiş: else → B</text>
<rect x="176" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="188" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="200" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="212" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="224" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="236" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="248" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="260" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="272" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="284" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="296" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="308" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="320" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="332" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="344" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="356" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="368" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="380" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="392" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="404" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="416" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="428" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="440" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="452" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="464" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="476" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="488" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="500" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="512" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="524" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="536" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="548" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<text x="176" y="362" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">her geçişte 16 şerit maskeli → bu eşit maliyetli örnekte şeritlerin yarısı boşta</text>
<text x="176" y="378" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(evrensel %50 kayıp değil: dal maliyetine ve zamanlamaya bağlı)</text>
</svg>

**Örnek:** 32 kişilik bir ekip aynı Excel formülünü kendi satırına uygulasın. Aynı komut, farklı veri: SIMT'nin sezgisi budur.

**Warp divergence (yol ayrışması)** neden maliyetlidir? Warp içindeki 16 thread `if` dalına, diğer 16 thread `else` dalına gidiyorsa donanım iki yolu maskeleyerek sırayla yürütmek zorunda kalabilir. İki kolun eşit maliyetli olduğu basit senaryoda etkinlik ciddi düşer. Ancak 'her divergence tam %50 performans kaybıdır' demek doğru değildir; branch davranışı, bağımsız iş ve zamanlama belirleyicidir.

### 2.3. Streaming Multiprocessor: GPU'nun çalışma istasyonu

Peki 32 thread'lik bu ekipler nerede çalışıyor? Hepsinin aynı anda devreye girip istediği kaynağı kullanabildiği sınırsız bir alan yok. GPU, hesaplama kaynaklarını **Streaming Multiprocessor (SM)** adını verdiğimiz birimlerde düzenliyor.

SM'yi fabrikanın tek bir üretim istasyonu gibi düşünün. İçinde işi sıraya koyan planlayıcı, aritmetik işlemleri yapan birimler, matris işlemlerini hızlandıran Tensor Core'lar ve çalışanların hemen yanındaki küçük ama hızlı veri alanları var. “GPU'da 132 SM var” dediğimizde, bu istasyonlardan 132 tane bulunduğunu ifade ediyoruz; her birinin içindeki thread sayısının sabit 132 olduğu gibi bir anlam çıkarmıyoruz.

H100 SXM'de **132 SM**, her SM'de **4 SM subpartition** bulunur. Bu nedenle 528 adet dördüncü nesil Tensor Core, 16.896 FP32 CUDA core ve diğer fonksiyonel birimler içeren bir mimari söz konusudur. Bunlar PCIe H100 için aynı sayılar değildir. [K2]

Tanıdık “989 TFLOP/s” sayısı doğrudan bu yerleşimden çıkar. Tepe değerin hesabında kullanılan 1,83 GHz saatte her Tensor Core döngü başına 1.024 yoğun BF16 FLOP yapar (8 × 8 × 8'lik bir çarpma-toplama bloğu):

$$
C_{\mathrm{peak}}=528\times1024\times1{,}83\times10^{9}\approx989\ \mathrm{TFLOP/s}
$$

Şöyle okuyun: 528 Tensor Core × döngü başına 1.024 FLOP × saniyede 1,83 milyar döngü. Aynı çarpım, gerçek bir kernel'ın neden tepeye ulaşamadığını da açıklar: bu 528 birimin her biri her döngüde beslenmelidir. Blackwell B200'de yoğun BF16 tepe değeri 2.250 TFLOP/s'ye çıkar. [K2][K16]

Bir SM içinde dört tür rolü ayıralım:

| Birim | Neyi yapar? | Benzetme |
|---|---|---|
| **Warp scheduler** | Hazır warp'ların komutlarını planlar | İş dağıtan şef |
| **CUDA core / ALU** | Toplama, çarpma, elementwise ve çeşitli sayısal işlemler | Çok yönlü iş tezgâhı |
| **Tensor Core** | Matris çarpımı-birikimi (MMA) hızlandırır | Seri üretim makinesi |
| **Register / shared memory** | İşlem sırasında küçük, hızlı verileri tutar | Tezgâh üstündeki malzeme |

<svg viewBox="0 0 720 374" role="img" aria-label="Bir H100 SM&#x27;nin içinde dört alt bölüm vardır. Her birinin kendi warp scheduler&#x27;ı, 64 KB register dosyası, 32 FP32, 16 INT32 ve 16 FP64 birimi ve bir Tensor Core&#x27;u bulunur. Dördü 256 KB&#x27;lık L1 veri önbelleğini ve shared memory&#x27;yi paylaşır; TMA birimi, tüm SM&#x27;lerin ortak kullandığı 50 MB&#x27;lık L2 ve HBM ile asenkron blok kopyalama yapar. 132 SM ile H100 SXM&#x27;de 528 Tensor Core ve 16.896 FP32 CUDA core bulunur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir H100 SM&#x27;nin içi: dört alt bölüm + shared memory</text>
<rect x="16" y="38" width="688" height="300" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="28" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="108.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">alt bölüm 0</text>
<rect x="36" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="108.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="36" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="108.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register dosyası 64 KB</text>
<text x="36" y="150" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">32 FP32 · 16 INT32 · 16 FP64</text>
<rect x="36" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="45" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="54" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="63" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="72" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="81" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="90" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="99" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="108" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="117" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="126" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="135" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="144" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="153" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="162" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="171" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="36" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="45" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="54" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="63" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="72" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="81" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="90" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="99" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="108" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="117" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="126" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="135" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="144" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="153" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="162" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="171" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="36" y="182" width="144" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="108.0" y="199.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">Tensor Core</text>
<text x="108.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">alt bölüm başına 1</text>
<rect x="196" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="276.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">alt bölüm 1</text>
<rect x="204" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="276.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="204" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="276.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register dosyası 64 KB</text>
<text x="204" y="150" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">32 FP32 · 16 INT32 · 16 FP64</text>
<rect x="204" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="213" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="222" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="231" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="240" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="249" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="258" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="267" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="276" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="285" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="294" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="303" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="312" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="321" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="330" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="339" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="204" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="213" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="222" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="231" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="240" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="249" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="258" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="267" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="276" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="285" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="294" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="303" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="312" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="321" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="330" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="339" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="204" y="182" width="144" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="276.0" y="199.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">Tensor Core</text>
<text x="276.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">alt bölüm başına 1</text>
<rect x="364" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="444.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">alt bölüm 2</text>
<rect x="372" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="444.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="372" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="444.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register dosyası 64 KB</text>
<text x="372" y="150" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">32 FP32 · 16 INT32 · 16 FP64</text>
<rect x="372" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="381" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="390" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="399" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="408" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="417" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="426" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="435" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="444" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="453" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="462" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="471" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="480" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="489" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="498" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="507" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="372" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="381" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="390" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="399" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="408" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="417" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="426" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="435" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="444" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="453" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="462" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="471" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="480" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="489" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="498" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="507" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="372" y="182" width="144" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="444.0" y="199.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">Tensor Core</text>
<text x="444.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">alt bölüm başına 1</text>
<rect x="532" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="612.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">alt bölüm 3</text>
<rect x="540" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="612.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="540" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="612.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register dosyası 64 KB</text>
<text x="540" y="150" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">32 FP32 · 16 INT32 · 16 FP64</text>
<rect x="540" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="549" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="558" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="567" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="576" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="585" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="594" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="603" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="612" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="621" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="630" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="639" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="648" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="657" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="666" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="675" y="156" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="540" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="549" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="558" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="567" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="576" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="585" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="594" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="603" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="612" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="621" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="630" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="639" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="648" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="657" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="666" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="675" y="166" width="7" height="8" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="540" y="182" width="144" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="612.0" y="199.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">Tensor Core</text>
<text x="612.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">alt bölüm başına 1</text>
<rect x="28" y="256" width="664" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="275.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">L1 veri önbelleği / shared memory · SM başına 256 KB</text>
<rect x="28" y="292" width="328" height="30" rx="4" style="fill:var(--c-border);fill-opacity:.08;stroke:var(--c-border);stroke-width:1.2"/>
<text x="192.0" y="311.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">TMA (asenkron blok kopyalama)</text>
<rect x="364" y="292" width="328" height="30" rx="4" style="fill:var(--c-border);fill-opacity:.08;stroke:var(--c-border);stroke-width:1.2"/>
<text x="528.0" y="311.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">→ L2 (50 MB, tüm SM&#x27;ler ortak) → HBM</text>
<text x="16" y="360" text-anchor="start" style="fill:var(--c-text);font-size:12px">H100 SXM&#x27;de × 132 SM = 528 Tensor Core ve 16.896 FP32 CUDA core.</text>
</svg>

Tensor Core'un rolü özellikle önemlidir. Transformer'ın lineer katmanlarında çok sayıda matris çarpımı vardır. `A × B + C` gibi işlemleri çok yüksek paralellikte yürütebilen özel donanım, H100'ün BF16 teorik throughput'unun büyük bölümünü açıklar.

**Teknik nüans:** 'CUDA core' ve 'Tensor Core' aynı işlem birimi değildir. `nvidia-smi` üzerinde GPU yüksek kullanımda görünse bile Tensor Core'ların verimli çalıştığını garanti etmiş olmayız.

### 2.4. Warp scheduler bellek gecikmesini nasıl gizler?

Şimdi ilginç bir soruya geldik. Bir warp, gereken veriyi HBM'den istiyor ve veri henüz gelmiyor. Bu 32 thread'in tamamı beklerken SM boş mu kalacak?

Mutlaka değil. Warp scheduler'ın kıymetini burada anlarız. Hazır başka bir warp varsa onun komutları yürütülebilir. Böylece ilk grubun bekleme süresine faydalı iş sığdırabiliriz. Buna “belleğin gecikmesini ortadan kaldırmak” değil, **gecikmeyi başka işle örtmek** diyoruz.

Bir warp'ın HBM'den veri istediğini düşünün. Veri hazır olmayana kadar o warp'ın sonraki komutu ilerleyemeyebilir. Bir SM'de başka hazır warp'lar bulunduğunda scheduler onlara geçebilir.

<svg viewBox="0 0 720 252" role="img" aria-label="Üç warp için 8 zaman dilimlik çizelge. Warp A 1–2 ve 7–8. dilimlerde hesaplar, 3–6&#x27;da bekler. Warp B 3–4&#x27;te hesaplar, diğer dilimlerde bekler. Warp C 5–6&#x27;da hesaplar, diğerlerinde bekler. SM&#x27;nin yürütme satırı A A B B C C A A&#x27;dır; hiçbir warp zamanın yarısından fazla çalışmasa da SM 8 dilimin hepsinde meşguldür. Bellek beklemesi kısalmaz, başka hazır warp&#x27;ların arkasına gizlenir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ws-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Gecikme gizleme: bir warp beklerken diğeri hesaplar</text>
<text x="183.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">1</text>
<text x="249.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">2</text>
<text x="315.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">3</text>
<text x="381.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">4</text>
<text x="447.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">5</text>
<text x="513.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">6</text>
<text x="579.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">7</text>
<text x="645.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">8</text>
<text x="138" y="46" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">zaman →</text>
<text x="16" y="73" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Warp A</text>
<rect x="152" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="183.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="218" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="249.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="284" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="315.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="350" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="381.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="416" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="447.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="482" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="513.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="548" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="579.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="614" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="645.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<text x="16" y="107" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Warp B</text>
<rect x="152" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="183.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="218" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="249.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="284" y="90" width="62" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="315.0" y="106.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="350" y="90" width="62" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="381.0" y="106.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="416" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="447.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="482" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="513.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="548" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="579.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="614" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="645.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<text x="16" y="141" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Warp C</text>
<rect x="152" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="183.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="218" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="249.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="284" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="315.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="350" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="381.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="416" y="124" width="62" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="447.0" y="140.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="482" y="124" width="62" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="513.0" y="140.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="548" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="579.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<rect x="614" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="645.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bekle</text>
<line x1="16" y1="162" x2="704" y2="162" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="16" y="189" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">SM yürütür</text>
<rect x="152" y="172" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="183.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A</text>
<rect x="218" y="172" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="249.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A</text>
<rect x="284" y="172" width="62" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<text x="315.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B</text>
<rect x="350" y="172" width="62" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<text x="381.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B</text>
<rect x="416" y="172" width="62" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="447.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">C</text>
<rect x="482" y="172" width="62" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="513.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">C</text>
<rect x="548" y="172" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="579.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A</text>
<rect x="614" y="172" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="645.0" y="189.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A</text>
<text x="16" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Hiçbir warp zamanın yarısından fazla meşgul değil; yine de SM 8 dilimin hepsinde iş yürütüyor.</text>
<text x="16" y="240" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Bellek beklemesi kısalmadı; hazır başka warp&#x27;ların işiyle örtüldü.</text>
</svg>

Buna **latency hiding (gecikme gizleme)** denir. Dikkat: Belleğin fiziksel gecikmesi sihirli biçimde azalmaz. Örtüştürülen diğer iş sayesinde **gözlenen toplam çalışma süresi** daha az etkilenir.

Çok kaba bir sezgisel model:

$$
\text{Gerekli bağımsız iş miktarı}\propto
\frac{\text{Bekleme süresi}}{\text{İki bağımsız komut arasındaki hesaplama süresi}}
$$

Bu bir donanım kuralı veya kesin warp formülü değildir; neden yüksek eşzamanlılık gerektiğini anlamaya yarar.

Bu sezgiye ölçülmüş sayılar koyabiliriz. Luo ve diğerleri (2024), Hopper'da yükleme gecikmelerini mikrobenchmark'larla ölçtü: shared memory için yaklaşık **29**, L1 için **41**, L2 için **263**, HBM için **479** döngü [K52]. Şimdi bir warp'ın 2 döngü hesap yapıp 6 döngü beklediğini düşünelim. Scheduler'ı her döngüde meşgul tutmak için gereken warp sayısı kabaca:

$$
N_{\mathrm{warp}}\approx1+\frac{T_{\mathrm{bekleme}}}{T_{\mathrm{hesap}}}=1+\frac{6}{2}=4
$$

Şöyle okuyun: bir warp hesap yaparken onun bekleme süresini dolduracak kadar başka warp hazır olmalıdır. Böyle iki warp'la scheduler zamanın yarısında boş kalır; dört warp'la hiç boş kalmaz. ~479 döngülük bir HBM yüklemesi çok daha fazla bağımsız iş ister. FlashAttention-3 gibi hızlı Hopper kernel'larının yalnızca warp sayısına güvenmeyip veri taşımayı TMA kopyaları ve asenkron WGMMA ile hesabın arkasına saklamasının nedeni budur. [K8]

### 2.5. Occupancy ve register pressure: Daha fazla thread her zaman daha hızlı değildir

Bu aşamada doğal bir fikir ortaya çıkıyor: “Öyleyse çok daha fazla warp açalım, GPU hiç boş kalmasın.” Mantıklı görünüyor, değil mi? Fakat her warp'ın bir maliyeti var. Çalışanların oturacağı yer ve kullanacağı tezgâh sonsuz değil.

Her thread'in kullandığı register sayısı arttıkça aynı SM'de barındırılabilecek thread sayısı azalabilir. Bir başka deyişle, bir işçinin çok büyük tezgâh istemesi, aynı anda çalışabilecek işçi sayısını azaltır.

**Occupancy**, SM üzerinde aynı anda ikamet eden aktif warp sayısının donanımın izin verdiği azami warp sayısına oranıdır. **Register pressure**, kernel'in thread başına ne kadar register kaynağı istediğini anlatır.

SM'nin register dosyası sınırlıdır. Bir thread çok sayıda register tüketirse aynı anda daha az thread/warp tutulabilir. Mesela H100 SM başına 65.536 adet 32-bit register varsayalım:

$$
R_{\text{warp}}=32\times R_{\text{thread}}
$$

<svg viewBox="0 0 720 288" role="img" aria-label="Bir SM&#x27;nin aynı 65.536 register&#x27;lık dosyası dört kernel için warp&#x27;lara bölünür. Thread başına 32 register&#x27;da en fazla 64 warp sığar (%100); SM warp üst sınırı ve blok düzeni de sınırlar. 64 register&#x27;da 32 warp (%50). 128 register&#x27;da 16 warp (%25); shared memory ve block boyutu da etkilidir. 255 register&#x27;da 8 warp (%12); tahsis granülerliği ve compiler limitleri devreye girer. Bunlar gerçek occupancy değil, yalnızca register bütçesinden türetilmiş üst sınırlardır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="oc-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Tek register dosyası, dört kernel: thread büyüdükçe warp azalır</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">SM başına 65.536 × 32-bit register · R_warp = 32 × R_thread · yalnızca register bütçesinden üst sınır</text>
<text x="16" y="78" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">32 register/thread</text>
<text x="16" y="95" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SM warp üst sınırı, blok düzeni</text>
<rect x="236" y="64" width="416" height="26" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="236.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="243.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="256.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="269.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="275.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="282.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="288.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="295.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="301.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="308.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="314.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="321.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="327.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="340.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="347.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="353.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="360.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="366.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="373.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="379.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="386.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="399.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="405.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="412.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="418.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="425.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="431.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="438.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="444.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="451.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="457.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="464.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="470.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="477.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="483.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="490.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="496.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="503.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="509.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="516.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="522.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="529.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="535.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="542.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="548.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="555.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="561.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="568.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="574.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="581.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="587.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="594.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="600.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="607.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="613.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="620.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="626.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="633.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="639.6" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="646.1" y="66" width="5.3" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="660" y="82" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">64 warp</text>
<text x="660" y="97" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">100%</text>
<text x="16" y="130" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">64 register/thread</text>
<text x="16" y="147" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">aynı etkenler</text>
<rect x="236" y="116" width="416" height="26" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="236.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="275.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="288.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="301.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="314.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="327.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="340.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="353.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="366.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="379.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="405.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="418.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="431.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="444.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="457.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="470.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="483.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="496.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="509.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="522.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="535.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="548.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="561.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="574.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="587.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="600.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="613.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="626.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="639.6" y="118" width="11.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="660" y="134" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">32 warp</text>
<text x="660" y="149" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">50%</text>
<text x="16" y="182" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">128 register/thread</text>
<text x="16" y="199" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">shared memory ve block boyutu</text>
<rect x="236" y="168" width="416" height="26" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="236.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="288.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="314.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="340.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="366.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="418.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="444.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="470.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="496.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="522.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="548.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="574.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="600.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="626.6" y="170" width="24.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="660" y="186" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">16 warp</text>
<text x="660" y="201" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">25%</text>
<text x="16" y="234" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">255 register/thread</text>
<text x="16" y="251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tahsis granülerliği, compiler limitleri</text>
<rect x="236" y="220" width="416" height="26" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="236.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="288.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="340.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="444.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="496.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="548.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="600.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="660" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">8 warp</text>
<text x="660" y="253" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">12%</text>
<text x="16" y="276" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Çubuktaki her parça = bir aktif warp. Gerçek occupancy daha düşük olabilir; yüksek occupancy kendiliğinden hız değildir.</text>
</svg>

Şekildeki warp sayıları yalnızca **register bütçesinden türetilmiş üst sınırlar**dır; gerçek occupancy değildir.

Peki her zaman occupancy'yi %100 yapmak mı gerekir? Hayır. Bazı compute-bound kernel'lar daha az warp'la daha yüksek Tensor Core verimi sağlar. Hopper'daki FlashAttention-3 gibi uygulamalarda asenkron veri taşıma ve warp specialization, düşük occupancy ile de işlem birimlerini dolu tutabilir. [K8]

**Öğrenme kontrolü:** Bir kernel'i hızlandırmak için 'daha çok thread' açmak isteyebilirsiniz. Önce register spill, shared memory kullanımı, warp divergence ve bellek erişim düzenini ölçmeden bunu yapmak performansı kötüleştirebilir.

---


## 3. Bellek hiyerarşisi: Veriye ulaşmak neden matematik kadar önemli?

Bir GPU'nun çok sayıda aritmetik birimi olduğunu öğrendik. Ama hepsi matematik yapabilmek için verilere ihtiyaç duyuyor. Peki veriler nerede?

Her şeyi HBM'de tutup her işlem öncesinde oradan okumak mümkün olsa da çoğu iş için verimsiz olur. Çünkü HBM çok yüksek bant genişliği sunsa bile register veya shared memory kadar yakından erişilebilir değildir. Bu nedenle GPU'nun içinde farklı boyutlarda ve farklı maliyetlerde bellek katmanları vardır.

Bunu bir aşçının elindeki bıçak, mutfak tezgâhı, mutfağın dolabı ve binanın deposu olarak düşünün. Mutfak tezgâhı çok hızlıdır ama sınırlı alan sunar; ana depo büyük miktarda malzeme tutar ama her şey için oraya gitmek maliyetlidir.

Modern GPU'nun içinde **hız–kapasite–erişim maliyeti** dengesi olan bir bellek merdiveni bulunur.

<svg viewBox="0 0 720 300" role="img" aria-label="Aşağı doğru genişleyen piramit biçiminde bellek merdiveni, yukarıdan aşağıya: register, SM başına 256 KB, tek thread&#x27;e özel; shared memory ve L1, SM başına 256 KB, ölçülmüş yükleme gecikmesi yaklaşık 29 ve 41 döngü, bir thread block&#x27;a ortak; L2, 50 MB, yaklaşık 263 döngü, 132 SM&#x27;nin ortak önbelleği; HBM3, 80 GB, 3,35 TB/s ve yaklaşık 479 döngü, ağırlık, aktivasyon ve KV cache burada; NVLink, PCIe ve NIC, yön başına 450, 63 ve 50 GB/s, başka GPU ve sunuculardaki veriye yol; CPU RAM ve depolama, DGX başına 2 TB ve petabayt ölçeği, veri kümeleri ve checkpoint&#x27;ler. Her basamak aşağısı daha büyük, daha uzak ve erişimi daha pahalıdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mh-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bellek merdiveni: her basamak aşağısı daha büyük, daha uzak, daha pahalı</text>
<rect x="210" y="40" width="120" height="36" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="270" y="62" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Register</text>
<text x="490" y="55" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">SM başına 256 KB</text>
<text x="490" y="70" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tek thread&#x27;e özel</text>
<line x1="334" y1="58" x2="482" y2="58" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="182" y="82" width="176" height="36" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="270" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Shared memory / L1</text>
<text x="490" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">SM başına 256 KB · ~29 / ~41 döngü</text>
<text x="490" y="112" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">bir thread block&#x27;a ortak</text>
<line x1="362" y1="100" x2="482" y2="100" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="154" y="124" width="232" height="36" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="270" y="146" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">L2</text>
<text x="490" y="139" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">50 MB · ~263 döngü</text>
<text x="490" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">132 SM&#x27;nin ortak önbelleği</text>
<line x1="390" y1="142" x2="482" y2="142" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="126" y="166" width="288" height="36" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="270" y="188" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">HBM3</text>
<text x="490" y="181" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">80 GB · 3,35 TB/s · ~479 döngü</text>
<text x="490" y="196" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağırlık, aktivasyon, KV cache</text>
<line x1="418" y1="184" x2="482" y2="184" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="98" y="208" width="344" height="36" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="270" y="230" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">NVLink / PCIe / NIC</text>
<text x="490" y="223" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">450 · 63 · 50 GB/s/yön</text>
<text x="490" y="238" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">başka GPU ve sunuculardaki veri</text>
<line x1="446" y1="226" x2="482" y2="226" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="70" y="250" width="400" height="36" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="270" y="272" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU RAM / depolama</text>
<text x="490" y="265" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">DGX başına 2 TB · PB ölçeği</text>
<text x="490" y="280" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">veri kümeleri, checkpoint</text>
<line x1="474" y1="268" x2="482" y2="268" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<path d="M20 44 V282" marker-start="url(#mh-arr)" marker-end="url(#mh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="28" y="54" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yakın</text>
<text x="28" y="278" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">uzak</text>
</svg>

**Register**, tek thread'in değişkenlerini tuttuğu çok hızlı alan; **shared memory**, aynı thread block içindeki thread'lerin açıkça yönetebildiği hızlı SRAM alanıdır. **L1/L2 cache**, sık kullanılan verilerin daha yakın tutulmasını sağlar. **HBM (High Bandwidth Memory)**, GPU üzerindeki büyük ağırlık ve aktivasyonları depolar.

H100 SXM için 80 GB HBM3, yaklaşık **3,35 TB/s** tepe bant genişliği ve yaklaşık 50 MB L2 belleği referans alabiliriz. `GB` ile `GiB` ve tepe bant genişliği ile sürdürülebilir uygulama bant genişliğini karıştırmamak gerekir. [K2]

### 3.1. Bandwidth ile latency aynı şey değildir

İki farklı soruyu ayıralım. “Bir kamyon aynı anda kaç koli taşıyabilir?” ve “Kamyonun depoya gidip gelmesi kaç dakika sürer?” Birincisi kapasiteyle, ikincisi bekleme süresiyle ilgilidir. Bellek sisteminde bandwidth ve latency arasındaki fark da böyledir.

**Bandwidth:** Bir saniyede kaç byte taşınabilir? Birimi GB/s veya TB/s.

**Latency:** Bir veri isteğinin sonucunu görmek için ne kadar bekleriz? Birimi ns veya clock cycle olabilir.

Bir otoyolun sekiz şeritli olması yüksek bandwidth, o yolun 500 kilometre olması yüksek latency sezgisidir. Her ikisi de performansı etkiler. Büyük blok taşımada bandwidth, küçük ve bağımlı isteklerde latency özellikle önemlidir.

**Memory coalescing**, bir warp içindeki thread'lerin bellek adreslerinin donanımın daha az sayıda verimli memory transaction yapmasına izin verecek biçimde düzenlenmesidir. Dağınık erişimler, taşınan baytların faydalı iş oranını azaltabilir.

### 3.2. Roofline: İşlemci gücü ile veri hızı arasındaki matematiksel sınır

Diyelim ki H100'ün ne kadar hızlı işlem yapabildiğini biliyoruz ve HBM'nin kaç byte taşıyabildiğini de biliyoruz. Bu iki sayıdan, bir algoritmanın hangi durumda bellek yüzünden yavaşlayacağını tahmin edebilir miyiz?

Evet. Bunun için kullanılan en faydalı sezgisel araçlardan biri **Roofline Modeli**. Karmaşık bir isme sahip ama sorduğu soru basit: “Bellekten getirdiğim her byte'la yeterince çok işlem yapabiliyor muyum?”

Bir kernel'in aritmetik yoğunluğu (*arithmetic intensity*):

$$
\alpha=\frac{\text{İşlem sayısı (FLOP)}}{\text{İlgili bellek düzeyinde taşınan byte}}
$$

Roofline modelinin basitleştirilmiş üst sınırı:

$$
P_{\mathrm{achievable}}\leq\min(C_{\mathrm{peak}},\alpha W_{\mathrm{mem}})
$$

Bu formülde $C_{\mathrm{peak}}$ donanımın belirtilen veri tipindeki teorik hesaplama kapasitesi, $W_{\mathrm{mem}}$ ölçtüğümüz bellek düzeyinin bant genişliği, $P_{\mathrm{achievable}}$ erişilebilir işlem hızıdır.

- $\alpha W_{\mathrm{mem}}<C_{\mathrm{peak}}$ ise **memory-bound** bölgesindeyiz.
- $\alpha W_{\mathrm{mem}}\ge C_{\mathrm{peak}}$ ise teorik **compute roof** bölgesine ulaşma imkânımız doğar.

H100 SXM ve **sparsity kullanılmayan BF16 Tensor Core** varsayımıyla:

$$
C_{\mathrm{peak}}\simeq990\ \mathrm{TFLOP/s},\quad
W_{\mathrm{HBM}}\simeq3.35\ \mathrm{TB/s}
$$

$$
\boxed{\alpha_{\mathrm{crit}}=\frac{990}{3.35}\simeq295.5\ \mathrm{FLOP/byte}}
$$

Bu sayı 'H100 her işte 295 işlem yapmalı' kuralı değildir. **BF16 Tensor Core** ve **HBM** için teorik roofline kesişimidir. CUDA core FP32 veya başka veri tipiyle hesaplamada tavan değişir.

### 3.3. Neden batch büyüklüğü her şeyi değiştiriyor?

Roofline'ı bir formül olarak öğrendik. Şimdi onu gerçekten hissedebileceğimiz bir örnekle kullanalım. Çok büyük bir ağırlık matrisiniz olduğunu ve sıradaki token için bu matrisi okuduğunuzu düşünün. Eğer bir kez okuduğunuz ağırlıkla yalnızca tek token üzerinde işlem yapıyorsanız, o veri taşımaya çok az hesaplama sığar.

Ama aynı ağırlığı *birden fazla token* üzerinde kullanabilirseniz, belleğe aynı maliyetle gidip çok daha fazla işlem gerçekleştirebilirsiniz. İşte batch büyüklüğünün etkisi buradan geliyor.

Bir lineer katmana bakalım:

$$
X\in\mathbb{R}^{B\times D},\quad
W\in\mathbb{R}^{D\times F},\quad
Y=XW
$$

BF16/FP16 veri ve tüm matrislerin HBM'den bir defa okunup sonucun bir defa yazıldığı *ideal basitleştirme* altında:

$$
FLOP\simeq2BDF
$$

$$
Byte\simeq2(BD+DF+BF)
$$

$$
\boxed{\alpha(B)\simeq\frac{BDF}{BD+DF+BF}}
$$

**Terimler:** $B$, birlikte işlenen token satırı sayısı; $D$, giriş gizil boyutu (*hidden dimension*); $F$, çıkış veya FFN boyutu.

Ağırlık matrisi çok büyük ve $B$ küçükse $DF$ baskın olur ve $\alpha\approx B$ çıkar. Çünkü ağırlıkları HBM'den bir kez okuyup yalnızca $B$ farklı girdi için kullanırsınız.

Llama 3 70B MLP boyutlarını öğretici örnek olarak alalım: $D=8192, F=28672$. Yukarıdaki formüle göre:

| Birlikte işlenen token B | Yaklaşık aritmetik yoğunluk (FLOP/byte) | H100 Roofline yorumu |
|---:|---:|---|
| 1 | 1,0 | Güçlü biçimde memory-bound |
| 8 | 8,0 | Memory-bound |
| 32 | 31,8 | Memory-bound |
| 128 | 125,5 | HBM tavanı hâlâ etkili |
| 310 | 295,6 | Teorik eşik civarı |
| 512 | 473,9 | Compute roof mümkün |

<svg viewBox="0 0 720 364" role="img" aria-label="H100 SXM için BF16 yoğun Tensor Core ve HBM&#x27;e göre log-log roofline. Bellek tavanı 3,35 TB/s çarpı aritmetik yoğunluk olarak yükselir ve yaklaşık 295,5 FLOP/byte&#x27;ta 990 TFLOP/s&#x27;lik hesap tavanıyla buluşur. Llama 3 70B MLP katmanı için noktalar: B=1 (α 1,0), B=8 (8,0), B=32 (31,8) ve B=128 (125,5) bellek tavanı üzerindedir; B=310 (295,6) kırılma noktasına ulaşır; B=512 (473,9) hesap tavanının altındadır. İdealize veri hareketi modelidir, ölçülmüş kernel değildir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rf-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">H100 SXM roofline&#x27;ı (BF16 yoğun, HBM) ve tablodaki batch değerleri</text>
<rect x="84" y="50" width="466.3" height="250" style="fill:var(--c-warn);fill-opacity:.06"/>
<rect x="550.3" y="50" width="139.7" height="250" style="fill:var(--c-success);fill-opacity:.06"/>
<line x1="84" y1="300" x2="690" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="304.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="84" y1="224.266" x2="690" y2="224.266" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="228.26606231134411" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<line x1="84" y1="148.532" x2="690" y2="148.532" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="152.5321246226882" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<line x1="84" y1="72.7982" x2="690" y2="72.7982" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="76.79818693403232" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.000</text>
<line x1="134.644" y1="50" x2="134.644" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="134.64440287267917" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="302.881" y1="50" x2="302.881" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="302.8814676242264" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<line x1="471.119" y1="50" x2="471.119" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="471.11853237577355" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<line x1="639.356" y1="50" x2="639.356" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="639.3555971273208" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.000</text>
<line x1="84" y1="300" x2="690" y2="300" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="84" y1="50" x2="84" y2="300" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="387.0" y="334" text-anchor="middle" style="fill:var(--c-text);font-size:12px">aritmetik yoğunluk α (FLOP/byte, log)</text>
<text x="22" y="175.0" text-anchor="middle" style="fill:var(--c-text);font-size:12px" transform="rotate(-90 22 175.0)">TFLOP/s (log)</text>
<path d="M84.0 283.0 L550.3 73.1 L690 73.1" style="fill:none;stroke:var(--c-accent);stroke-width:2.4"/>
<text x="214.91388229729048" y="214.10201792524697" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600" transform="rotate(-24.5 215 214)">bellek tavanı: 3,35 TB/s × α</text>
<line x1="84" y1="73.1288" x2="550.289" y2="73.1288" style="stroke:var(--c-accent);stroke-width:1;stroke-dasharray:4 3;stroke-opacity:.5"/>
<text x="538.2892778792796" y="66.1287508344046" text-anchor="end" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">hesap tavanı: 990 TFLOP/s</text>
<line x1="550.289" y1="73.1288" x2="550.289" y2="300" style="stroke:var(--c-accent);stroke-width:1;stroke-dasharray:4 3"/>
<text x="556.2892778792796" y="292" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">α* ≈ 295,5</text>
<text x="97.32122041842246" y="66" text-anchor="start" style="fill:var(--c-warn);font-size:12px;font-weight:600">memory-bound</text>
<text x="620.1446389396398" y="240" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:600">compute-bound</text>
<text x="620.1446389396398" y="256" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:600">mümkün</text>
<circle cx="134.633" cy="260.241" r="5" style="fill:var(--c-warn);fill-opacity:.95"/>
<text x="142.6329364897536" y="278.24145103027234" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">B=1</text>
<circle cx="286.486" cy="191.883" r="5" style="fill:var(--c-warn);fill-opacity:.95"/>
<text x="294.4859307732348" y="209.8829996754247" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">B=8</text>
<circle cx="387.5" cy="146.41" r="5" style="fill:var(--c-warn);fill-opacity:.95"/>
<text x="395.500382593623" y="164.4101295256936" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">B=32</text>
<circle cx="487.702" cy="101.303" r="5" style="fill:var(--c-warn);fill-opacity:.95"/>
<text x="495.70195991195186" y="119.30318460759065" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">B=128</text>
<circle cx="550.313" cy="73.1288" r="5" style="fill:var(--c-success);fill-opacity:.95"/>
<text x="554.3126959176745" y="95.1287508344046" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">B=310</text>
<circle cx="584.797" cy="73.1288" r="5" style="fill:var(--c-success);fill-opacity:.95"/>
<text x="588.796756687289" y="64.1287508344046" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">B=512</text>
<text x="16" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Noktalar: α(B) = B·D·F / (B·D + D·F + B·F), D = 8192, F = 28672 (Llama 3 70B MLP); idealize veri hareketi modeli, ölçülmüş kernel değil.</text>
</svg>

Bu sayıların **gerçek ölçülmüş kernel FLOP/s değerleri** değil, idealize edilen matris için veri-hareketi hesabı olduğunu unutmayın. Tiling, cache reuse, dönüşümler ve diğer katmanlar tabloyu değiştirir.

### 3.4. Tensor Core doluluğu, occupancy ve MFU farklıdır

GPU Utilization, SM activity, achieved occupancy, Tensor Core utilization, DRAM throughput ve MFU birbirinin yerine kullanılamaz. Örneğin SM yoğun çalışırken kötü memory locality yüzünden az FLOP üretiyor olabilir. Bir diğer sistemde Tensor Core kernel'ı çok verimli çalışır fakat her training step'in yarısı collective bekleyerek geçebilir. Birinci durumda kernel, ikinci durumda sistem düzeyi sorun vardır.

### 3.5. Nesiller boyunca: Kırılma noktası neden yukarı kayıyor?

Roofline'ın kırılma noktası $\alpha_{\mathrm{crit}}=C_{\mathrm{peak}}/W_{\mathrm{HBM}}$ sabit bir doğa yasası değildir. Her GPU neslinde hesap gücü bellek bant genişliğinden hızlı büyüdüğü için yukarı kayar. Tablo, üreticilerin yoğun (sparsity'siz) tepe değerlerini bir araya getiriyor:

| Nesil | Tensor Core | HBM kapasite · bant genişliği | BF16/FP16 TFLOP/s | FP8 TFLOP/s | FP4 TFLOP/s | $\alpha^{*}=C/W$ (FLOP/byte) |
|---|---|---|---:|---:|---:|---:|
| V100 SXM2 | 1. nesil | 32 GB · 0,9 TB/s | 125 (FP16) | — | — | ≈139 |
| A100 SXM | 3. nesil | 80 GB · 2,0 TB/s | 312 | — (INT8: 624 TOPS) | — | ≈156 |
| H100 SXM | 4. nesil | 80 GB · 3,35 TB/s | 989 | 1.979 | — | ≈295 |
| H200 SXM | 4. nesil | 141 GB · 4,8 TB/s | 989 | 1.979 | — | ≈206 |
| B200 (DGX B200) | 5. nesil | 180 GB · 8 TB/s | 2.250 | 4.500 | 9.000 | ≈281 |

İki ders çıkıyor. Birincisi, H200 aynı hesap gücüne %43 daha fazla bellek bant genişliği ekler ve kırılma noktasını 295'ten 206 FLOP/byte'a indirir: TFLOP/s'si aynı olsa da memory-bound decode için H100'den hızlı bir çiptir. İkincisi, B200 iki tarafı da büyütür ama oran yine 280 FLOP/byte civarında kalır: matris çarpımı başına yaklaşık 300 token'ın altında Tensor Core'lar hâlâ aç kalır. Blackwell ayrıca Tensor Core birikimlerini tutmak için SM başına ayrı, 256 KB'lık bir Tensor Memory (TMEM) ekler. [K16][K56]

---



---

### Ara durak: GPU'yu doğru anlamak için üç ayrı "paralellik" düşüncesi

GPU mimarisinde sıkça birbirine karıştırılan üç kavramı netleştirelim:

1. **Concurrency (Eşzamanlılık):** Birden fazla işin aynı zaman penceresinde ilerlemesi.
2. **Parallel execution (Paralel yürütme):** Farklı donanım birimlerinde (örneğin birden çok SM veya Tensor Core) aynı anda fiziksel hesaplama yapılması.
3. **Latency hiding (Gecikme gizleme):** Bir warp bellekten veri beklerken scheduler'ın hazır olan başka bir warp'ı yürütmesi.

Yüksek occupancy değeri, scheduler'ın elinde geçiş yapabileceği daha fazla warp olduğunu gösterir; ancak bu durum Tensor Core'ların faydalı FLOP ürettiğini tek başına garanti etmez. Register baskısı ve bellek bağımlılıkları çözülmeden yalnızca occupancy artırmak performansı iyileştirmez.

**Düşünce deneyi:** Bir kernel 256 thread'lik bloklarla çalışırken aynı SM'de yalnızca bir blok yerleşebiliyorsa, blok büyüklüğünü 128'e indirmek neden bazen daha fazla aktif warp sağlayabilir? Çünkü blok başına register/shared memory tahsisi ve SM başına kaynak sınırları birlikte belirleyicidir. Fakat bu değişiklik kodun daha hızlı çalışacağını garanti etmez: Yeni düzen daha fazla koordinasyon veya daha düşük veri yeniden kullanımı da doğurabilir.


### Bellek "yakınlığı" neden sadece fiziksel uzaklık değildir?

HBM, GPU'ya bağlıdır; bu nedenle “GPU'nun yanındaki bellek” demek sezgisel olarak doğrudur. Fakat bir warp için verinin register'da bulunması ile HBM'den getirilmesi aynı şey değildir. Buradaki *yakınlık* yalnızca santimetrelerle ilgili değildir: erişim protokolü, bank çakışmaları, cache hit oranı, transfer şekli ve bir erişim tamamlanırken başka işin sürüp sürmediği de belirleyicidir.

O nedenle **latency** ile **bandwidth** birbirinin yerine kullanılmamalı. Tek bir küçük verinin gelmesini beklemek gecikmeyle; binlerce byte'ı uzun süre ardışık taşımak bant genişliğiyle daha yakından ilişkilidir. Bir yolun az trafikle hızlı gidilebilir olması, aynı yoldan aynı anda çok kamyon geçebileceği anlamına gelmez.

### Roofline'ı tabloya değil, iş yüküne uygulayalım

H100 SXM için BF16 yoğun Tensor Core tepe kapasitesini yaklaşık 990 TFLOP/s, HBM bant genişliğini 3,35 TB/s alırsak kırılma noktası yaklaşık 296 FLOP/byte olur. Bu oran bir *kanun* değil, belirli precision ve hesaplama yolu için teorik sınırdır.

Şimdi üç farklı işi ele alalım:

| İşin şekli | Ağırlık tekrar kullanımı | Beklenen eğilim |
|---|---|---|
| Tek vektörle çok büyük matris çarpımı | Düşük | HBM trafiği belirleyici olabilir |
| Aynı matrise karşı büyük token batch'i | Daha yüksek | Tensor Core kapasitesi önem kazanabilir |
| Küçük, art arda pointwise kernel'lar | Düşük veya parçalı | Kernel launch ve gereksiz HBM round-trip'leri öne çıkabilir |

Bu ayrım GPU seçimi kadar **algoritma ve kernel tasarımının** neden önemli olduğunu açıklar. Daha hızlı HBM bazen yardımcı olur; bazen en iyi kazanım aynı HBM verisini daha çok hesapta tekrar kullanmaktır. Kernel fusion ve FlashAttention gibi tekniklerin anlamı burada ortaya çıkar.




---

**KISIM II — Modelin yaptığı iş: training, prefill, decode ve KV cache**

*Aynı Transformer neden farklı görevlerde farklı hız sınırlarına takılıyor? İstekten kullanıcıya uzanan yolu izleyeceğiz.*


## 4. Transformer iş yükü: Training, prefill ve decode neden aynı GPU'yu farklı kullanıyor?

Şimdiye kadar GPU'ya ve belleğe baktık. Peki GPU tam olarak neyi hesaplıyor? Bu soruya cevap vermeden “LLM inference yavaş”, “training compute-bound” gibi cümleler biraz havada kalır.

Bir Transformer'ı tek bir dev ve anlaşılmaz matematiksel nesne olarak düşünmeyin. İçinde sürekli tekrarlanan birkaç temel iş var: token'ları vektörlere dönüştürmek, bu vektörlerden birbirleriyle ilgili bilgiler üretmek, büyük ağırlık matrisleriyle çarpmak ve sonucu bir sonraki katmana aktarmak. Bu işlerin nasıl paketlendiği değiştiğinde GPU'nun darboğazı da değişiyor.

### 4.1. Önce Transformer'ın yaptığı işi görelim

Çok kabaca bir Transformer bloğu şu hesaplamaları yapar:

<svg viewBox="0 0 720 428" role="img" aria-label="Bir Transformer bloğunun yukarıdan aşağı akışı: token ID&#x27;leri; N × d boyutunda X aktivasyonlarına embedding okuması; GEMM olan Q, K, V lineer projeksiyonları; maliyeti dizi uzunluğuyla büyüyen attention, yani softmax(Q·Kᵀ/√d)·V; çıkış projeksiyonu, residual ve normalizasyon; ve yine GEMM olan MLP, W2 · act(W1 · X). Projeksiyonlardan MLP&#x27;ye kadar olan kısım L kez tekrarlanır (70B modelde 80); son norm logits ve sonraki token olasılıklarını üretir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tb-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir Transformer bloğu ne hesaplar?</text>
<rect x="150" y="40" width="360" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="162" y="57" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Token ID&#x27;leri</text>
<text x="162" y="73" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[t1, t2, …, tN]</text>
<line x1="330.0" y1="80" x2="330.0" y2="92" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="92" width="360" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="162" y="109" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Embedding</text>
<text x="162" y="125" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">tablo okuma → X  (N × d)</text>
<rect x="524" y="103" width="93.312" height="18" rx="9" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="570.656" y="115.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">bellek okuma</text>
<line x1="330.0" y1="132" x2="330.0" y2="144" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="144" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="161" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Q, K, V projeksiyonları</text>
<text x="162" y="177" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Q = XW_Q,  K = XW_K,  V = XW_V</text>
<rect x="524" y="155" width="39.104" height="18" rx="9" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="543.552" y="167.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GEMM</text>
<line x1="330.0" y1="184" x2="330.0" y2="196" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="196" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="162" y="213" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Attention</text>
<text x="162" y="229" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">softmax(Q·Kᵀ / √d) · V</text>
<rect x="524" y="207" width="161.072" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="604.5360000000001" y="219.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">dizi uzunluğuyla büyür</text>
<line x1="330.0" y1="236" x2="330.0" y2="248" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="248" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="265" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Çıkış proj. + residual + norm</text>
<text x="162" y="281" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">X + Attn·W_O → norm</text>
<rect x="524" y="259" width="120.416" height="18" rx="9" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="584.208" y="271.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GEMM + pointwise</text>
<line x1="330.0" y1="288" x2="330.0" y2="300" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="300" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="317" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">MLP / Feed-Forward</text>
<text x="162" y="333" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">W2 · act(W1 · X)</text>
<rect x="524" y="311" width="39.104" height="18" rx="9" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="543.552" y="323.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GEMM</text>
<path d="M140 144 H128 V340 H140" style="fill:none;stroke:var(--c-accent);stroke-width:1.6"/>
<text x="118" y="238.0" text-anchor="end" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">× L blok</text>
<text x="118" y="254.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">(70B&#x27;de 80)</text>
<path d="M330.0 340 V358" marker-end="url(#tb-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="150" y="360" width="360" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="330.0" y="382.0" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Son norm → logits → sonraki token olasılıkları</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Teal = büyük matris çarpımları (Tensor Core). Norm, softmax ve veri taşıma da gerçek zaman alır.</text>
</svg>

**Embedding:** Token kimliğini sayısal vektöre dönüştürür. **Activation:** Bir katmanın ürettiği geçici tensör. **Attention:** Farklı token konumlarının birbirlerinden bilgi almasını sağlar. **MLP:** Her token'ın temsilini öğrenilmiş matrislerle dönüştürür. **Logits:** Bir sonraki token olasılıkları hesaplanmadan önceki skorlar.

Modelin hesaplamasının önemli bölümü matris çarpımlarıdır; ama norm, softmax, veri taşımaları ve senkronizasyon da toplam zamanı etkiler.

### 4.2. Training: Büyük GEMM ve backward

Eğitim sırasında amaç yalnızca metnin devamını üretmek değil, modelin ağırlıklarını değiştirmektir. Bu yüzden iki yönlü bir süreç gerekir. Önce model mevcut ağırlıklarıyla tahmin üretir; sonra tahminin ne kadar hatalı olduğuna bakılarak parametreleri hangi yönde değiştirmemiz gerektiği hesaplanır.

Örneğin bir öğretmen öğrencinin çözümüne not verip yalnızca “yanlış” demekle kalmaz; hangi adımın nasıl düzeltilmesi gerektiğini de anlatır. Eğitimde *loss* ve *backward* bu geri bildirim mekanizmasının matematiksel karşılığıdır.

Eğitimde çok sayıda token birlikte işlenir. İleri geçiş (*forward pass*) aktivasyonları hesaplar; kayıp (*loss*) oluşturulur; geri yayılım (*backward pass*) gradyanları hesaplar; optimizer ağırlıkları günceller.

Çok sayıda satır içeren matrislerin çarpılması **GEMM (General Matrix–Matrix Multiplication)** iş yükü oluşturur. Ağırlıklar tek veri transferinde birçok token için yeniden kullanılabilir. Bu nedenle yoğun eğitim iş yükleri çoğu zaman compute-bound olmaya daha yakındır.

### 4.3. Prefill: Prompt'un topluca işlenmesi

Bir sohbet botuna 2.000 token'lık uzun bir soru gönderdiğinizi düşünün. Model henüz size tek kelime cevap vermeden önce bu 2.000 token'ın bağlamını hesaplamalı. Prefill dediğimiz aşama bu ilk hazırlıktır.

Prefill'in güzel yanı, çok sayıda token'ın aynı anda ele alınabilmesidir. Bu, büyük matris çarpmalarına ve ağırlıkların yeniden kullanılmasına imkân verir. Ancak attention'ın uzun dizilerde oluşturduğu ayrı maliyetleri de unutmamalıyız.

LLM servisine 2.000 token'lık bir prompt gönderildiğinde sistem önce bu prompt'u işleyerek attention durumunu ve **KV cache** verisini oluşturur. Buna **prefill** denir.

Prefill çoğunlukla daha büyük matrisler üzerinde çalışabildiğinden yüksek aritmetik yoğunluk potansiyeline sahiptir. Bununla birlikte attention boyutu, sequence length, kullanılan kernel ve model mimarisi yüzünden her prefill'in koşulsuz compute-bound olduğu iddia edilemez.

### 4.4. Decode: Tek tek token üretimi

İlk token üretildikten sonra işin şekli değişir. Dil modelleri genellikle yanıtı soldan sağa, token token ilerleterek üretir: bir sonraki token için mevcut bağlama bakar, token'ı üretir, sonra yeni bağlamla tekrar çalışır.

Yani 2.000 token'lık prompt'u topluca okumakla, bundan sonra gelen her yeni token için model ağırlıklarını yeniden kullanmak aynı hesaplama örüntüsü değildir. Inference performansının neden tek bir “token/s” sayısına indirgenemeyeceği burada ortaya çıkar.

Sonraki token üretiminde model, geçmiş konumların key/value verilerini KV cache'ten kullanırken mevcut token'ın katmanlardan geçişini tamamlar. Batch küçükken büyük ağırlık matrislerini tekrar okumak gerekir.

**B=1 dense BF16 model** için ağırlık trafiği üzerinden yapılan basit çıkarım $\alpha\approx1\ \mathrm{FLOP/byte}$ verir. Burada ağırlıklar GPU HBM'e sığdığı ve aynı anda işlenen istek sayısı az olduğu sürece token üretimi sıkça memory-bound'dur.

Fakat bu, 'decode daima memory-bound' veya 'MFU kesin %1' demek değildir. Çoklu eşzamanlı istekler, batched decode, attention süresi ve cache davranışı rejimi değiştirebilir.

### 4.5. KV cache neyi saklar?

Her yeni kelimede konuşmanın başından itibaren her ara bilgiyi tekrar hesaplamak istemeyiz. Bir toplantıda konuşulanları her cümlede baştan dinlemeniz gerekseydi sohbet çok yavaş ilerlerdi. Bir not defteri tuttuğunuzda ise geçmişin önemli bilgilerine geri dönebilirsiniz.

KV cache'in sezgisel rolü böyle bir not defterine benzer. Fakat sakladığı şey kelimelerin kendisi değil, attention hesaplamasında kullanılan **key** ve **value** tensörleridir.

**KV cache**, önceki token konumlarına ait key ve value aktivasyonlarını saklayarak her yeni token'da geçmişin yeniden projeksiyonunu önler.

Basitleştirilmiş BF16 belleği:

$$
S_{\mathrm{KV}}\approx
2\times L\times T\times H_{\mathrm{KV}}\times d_{\mathrm{head}}\times b
$$

Burada $L$ katman sayısı, $T$ saklanan token sayısı, $H_{\mathrm{KV}}$ KV head sayısı, $d_{\mathrm{head}}$ head boyutu ve $b$ eleman başına byte sayısıdır. Baştaki 2, key ve value için.

Bu formül toplam istek sayısıyla çarpıldığında GPU belleğinin eşzamanlı kullanım altında neden hızla dolabildiğini görürüz. **GQA (Grouped-Query Attention)** birden çok query head'in aynı KV head'i paylaşmasını sağlayarak cache kapasitesi ihtiyacını azaltabilir.

### 4.6. Quantization: Daha az byte taşıyarak hızlanmak

Bellek duvarına çarptığımızı düşünün. Bir çözüm, aynı yoldan daha az veri taşımaktır. Örneğin bir modelin her ağırlığını 2 byte yerine 1 byte ile temsil edebilirsek, teorik olarak ağırlık verisinin kapladığı alan yarıya iner.

Bunun adı **quantization (kuantizasyon)**. Ancak burada akılda tutulması gereken ince bir nokta var: Sayıyı daha küçük temsil etmek ile modelin aynı hızda, aynı doğrulukla çalışmasını sağlamak farklı hedeflerdir.

BF16 ağırlıklar 2 byte/parametre, INT8 ağırlıklar yaklaşık 1 byte/parametre, INT4 ise yaklaşık 0,5 byte/parametre yer kaplar. Ölçek ve sıfır noktası gibi metadata hariç kaba karşılaştırma:

<svg viewBox="0 0 720 311" role="img" aria-label="Ham ağırlık belleğini gösteren yatay çubuklar. 8B model: BF16&#x27;da 16 GB, INT8&#x27;de 8 GB, INT4&#x27;te 4 GB. 70B: 140, 70 ve 35 GB. 405B: 810, 405 ve 202,5 GB. Kesikli referans çizgileri tek H100&#x27;ü (80 GB) ve 8 GPU&#x27;lu DGX&#x27;i (640 GB) gösterir: 70B model tek H100&#x27;e yalnızca INT4&#x27;te sığar, 405B model BF16&#x27;da tek DGX&#x27;e sığmaz. Ölçek, sıfır noktası, KV cache ve çalışma alanı dahil değildir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="qb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Hassasiyete göre ağırlık belleği ve GPU belleği</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca ham ağırlık (ölçek, sıfır noktası, KV cache, çalışma alanı hariç)</text>
<text x="16" y="110" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">8B</text>
<text x="142" y="93" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16</text>
<rect x="150" y="82" width="9.6" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<text x="165.6" y="94" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">16 GB</text>
<text x="142" y="112" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT8</text>
<rect x="150" y="101" width="4.8" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.75"/>
<text x="160.8" y="113" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">8 GB</text>
<text x="142" y="131" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT4</text>
<rect x="150" y="120" width="2.4" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.75"/>
<text x="158.4" y="132" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">4 GB</text>
<text x="16" y="181" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">70B</text>
<text x="142" y="164" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16</text>
<rect x="150" y="153" width="84" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<text x="240.0" y="165" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">140 GB</text>
<text x="142" y="183" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT8</text>
<rect x="150" y="172" width="42" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.75"/>
<text x="198.0" y="184" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">70 GB</text>
<text x="142" y="202" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT4</text>
<rect x="150" y="191" width="21" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.75"/>
<text x="177.0" y="203" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">35 GB</text>
<text x="16" y="252" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">405B</text>
<text x="142" y="235" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16</text>
<rect x="150" y="224" width="486" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<text x="642.0" y="236" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">810 GB</text>
<text x="142" y="254" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT8</text>
<rect x="150" y="243" width="243" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.75"/>
<text x="399.0" y="255" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">405 GB</text>
<text x="142" y="273" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT4</text>
<rect x="150" y="262" width="121.5" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.75"/>
<text x="277.5" y="274" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">202,5 GB</text>
<line x1="198" y1="74" x2="198" y2="285" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="202.0" y="299" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">1 × H100 = 80 GB</text>
<line x1="534" y1="74" x2="534" y2="285" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="538.0" y="299" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">8 × H100 (DGX) = 640 GB</text>
<text x="150.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="270.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">200</text>
<text x="390.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">400</text>
<text x="510.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">600</text>
<text x="630.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">800</text>
</svg>

**Neden her INT4 model daha hızlı değil?** Dequantization, kernel desteği, Tensor Core sayısal formatı, KV cache maliyeti ve bellek erişim düzeni nedeniyle gerçek performans değişebilir. Ayrıca quantization kaliteye zarar verebilir; perplexity ve görev bazlı doğruluk ölçümü gerekir.

**Pratik ilke:** Training için MFU ve step time; online inference için TTFT (*time to first token*), TPOT (*time per output token*), throughput, KV cache kapasitesi ve p95/p99 latency birlikte raporlanmalıdır.

### 4.7. FlashAttention, TMA ve WGMMA: Aynı matematiği daha akıllı taşımak

Şimdi bellek kullanımını yalnızca sayıları küçülterek değil, *algoritmanın veri hareketini değiştirerek* azaltabileceğimizi düşünelim.

Bir yemek yaparken her malzeme için binanın ana deposuna inmek yerine gerekli malzemeleri tezgâha toplamak işinizi hızlandırır. GPU'da matrisleri küçük bloklara ayırıp shared memory ve register'larda yeniden kullanmak da benzer bir fikirdir. İşte FlashAttention ve Hopper'ın asenkron mekanizmaları bu sezgiyi ileri taşır.

**FlashAttention**, büyük attention ara matrisini HBM'e bütünüyle yazmak yerine bloklar halinde hesaplayıp daha küçük ve hızlı belleği kullanır. Bu işleme **tiling (döşeme/bloklama)** denir. Temel fikir, matematiksel attention'ı değiştirmeden gereksiz veri okuma-yazma miktarını azaltmaktır. [K8]

Hopper özelinde:

- **TMA (Tensor Memory Accelerator):** Büyük blok veri hareketlerini asenkron başlatabilen donanım yolu.
- **WGMMA (Warpgroup Matrix Multiply-Accumulate):** Warp grubu ölçeğinde matris işlemlerini asenkron gerçekleştirebilen komut ailesi.
- **Warp specialization:** Bazı warp'ların veri getirmeye, diğerlerinin matris hesaplamaya ağırlık vermesi.
- **Double buffering / overlap:** Bir blok üzerinde hesap yaparken sonraki bloğu hazırlamak.

<svg viewBox="0 0 720 264" role="img" aria-label="Dört şeritli zaman çizelgesi: HBM, TMA, shared memory ve WGMMA. K/V blok 1, 2 ve 3 HBM&#x27;de durur. TMA blok 1&#x27;i tampon A&#x27;ya, blok 2&#x27;yi tampon B&#x27;ye, blok 3&#x27;ü yeniden tampon A&#x27;ya yükler. WGMMA her bloğu yüklemeden bir dilim sonra hesaplar; blok 1&#x27;i hesaplarken TMA blok 2&#x27;yi, blok 2&#x27;yi hesaplarken blok 3&#x27;ü yükler. Gölgeli örtüşme pencereleri, k+1. bloğun veri taşımasının k. bloğun hesabının arkasına gizlendiğini gösterir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tm-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tm-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Hopper&#x27;da çift tampon: WGMMA k. bloğu hesaplarken TMA k+1&#x27;i yükler</text>
<text x="16" y="70" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">HBM</text>
<line x1="144" y1="82" x2="704" y2="82" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="120" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">TMA</text>
<line x1="144" y1="132" x2="704" y2="132" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">Shared</text>
<line x1="144" y1="182" x2="704" y2="182" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="220" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">WGMMA</text>
<line x1="144" y1="232" x2="704" y2="232" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="254" text-anchor="start" style="fill:var(--c-text);font-size:1px"></text>
<rect x="150" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="214.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V blok 1</text>
<rect x="286" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="350.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V blok 2</text>
<rect x="422" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="486.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V blok 3</text>
<rect x="150" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="214.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">yükle 1</text>
<rect x="154" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="214.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">tampon A</text>
<line x1="214.0" y1="128" x2="214.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="286" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="350.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">hesapla 1</text>
<path d="M272 178 L296 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="286" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="350.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">yükle 2</text>
<rect x="290" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="350.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">tampon B</text>
<line x1="350.0" y1="128" x2="350.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="422" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="486.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">hesapla 2</text>
<path d="M408 178 L432 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="422" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="486.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">yükle 3</text>
<rect x="426" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="486.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">tampon A</text>
<line x1="486.0" y1="128" x2="486.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="558" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="622.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">hesapla 3</text>
<path d="M544 178 L568 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="286" y="94" width="128" height="140" rx="6" style="fill:var(--c-accent-2);fill-opacity:.06;stroke:var(--c-accent-2);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="422" y="94" width="128" height="140" rx="6" style="fill:var(--c-accent-2);fill-opacity:.06;stroke:var(--c-accent-2);stroke-width:1;stroke-dasharray:4 3"/>
<text x="418.0" y="252" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">örtüşme: k+1&#x27;in veri taşıması k&#x27;nın hesabının arkasına gizlenir</text>
<text x="704" y="44" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">zaman →</text>
</svg>

**Sezgi:** Mutfakta ikinci aşçı, birinci aşçı mevcut yemeği pişirirken yeni malzemeyi hazırlar. Böylece mutfak daha az boş kalır. Ama tezgâh küçükse veya koordinasyon kötüyse daha fazla aşçı işe yaramayabilir.

### 4.8. Uçtan uca bir inference isteğini izleyelim

Training'i konuşurken çok sayıda token'ı aynı anda işlediğimizi, inference sırasında ise prefill ve decode davranışlarının ayrıştığını öğrendik. Şimdi bir kullanıcı isteğini servisin girişinden çıkışına kadar takip edelim. Böylece hangi darboğazın hangi kullanıcı metriğini etkilediği daha net görünecek.

**Senaryo:** Bir kullanıcı servise 2.000 token'lık bir soru gönderiyor ve 200 token'lık yanıt alıyor. Model 70B sınıfında olsun; BF16 ağırlıkları kabaca 140 GB. Sunucu birden fazla GPU üzerinde modeli çalıştırıyor ve istekleri birlikte zamanlayan bir inference engine kullanıyor.

<svg viewBox="0 0 720 314" role="img" aria-label="70B modelde 2.000 token&#x27;lık prompt ve 200 token&#x27;lık yanıt. CPU ve servis tarafında istek tokenizer&#x27;dan ve isteği kuyruğa alıp sürekli batch&#x27;e ekleyen scheduler&#x27;dan geçer. GPU&#x27;da prefill 2.000 token&#x27;ı büyük GEMM&#x27;lerle birlikte işler ve KV cache&#x27;i doldurur. Ardından decode her adımda bir token üretir, bunu 200 kez yapar; ağırlıkları ve büyüyen KV cache&#x27;i okuduğu için çoğunlukla memory-bound&#x27;dur. Her yeni token decode döngüsüne geri döner, detokenize edilip akıtılır. Altta kullanıcının zaman çizgisi: kuyruk, tokenize, prefill ve ilk decode TTFT&#x27;yi oluşturur; sonraki token&#x27;lar arasındaki süre TPOT&#x27;tur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="il-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="il-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="il-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Tek istek: 2.000 token prompt, 200 token yanıt (70B, BF16 ≈ 140 GB)</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">CPU / servis</text>
<text x="16" y="148" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">GPU</text>
<line x1="16" y1="120" x2="704" y2="120" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<rect x="16" y="60" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="71.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">İstek</text>
<text x="71.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">prompt metni</text>
<rect x="140" y="60" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="195.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Tokenizer</text>
<text x="195.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">metin → ID</text>
<rect x="264" y="60" width="140" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="334.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Scheduler</text>
<text x="334.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kuyruk + sürekli batch</text>
<line x1="126" y1="83" x2="138" y2="83" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="250" y1="83" x2="262" y2="83" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="250" y="132" width="164" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="332.0" y="151.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Prefill</text>
<text x="332.0" y="166.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2.000 token birlikte</text>
<text x="332.0" y="181.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">büyük GEMM, compute ağırlıklı</text>
<line x1="334" y1="106" x2="334" y2="130" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="132" width="92" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="476.0" y="151.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">KV cache</text>
<text x="476.0" y="166.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">token başına</text>
<text x="476.0" y="181.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">büyür</text>
<line x1="414" y1="161" x2="428" y2="161" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="538" y="132" width="166" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="621.0" y="151.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Decode</text>
<text x="621.0" y="166.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım başına 1 token × 200</text>
<text x="621.0" y="181.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">ağırlık + KV → memory-bound</text>
<line x1="522" y1="161" x2="536" y2="161" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M690 190 V212 H560 V192" marker-end="url(#il-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="625" y="226" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px">sonraki token</text>
<rect x="538" y="60" width="166" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="621.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Detokenize + akış</text>
<text x="621.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">token → metin</text>
<line x1="621" y1="130" x2="621" y2="108" marker-end="url(#il-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="16" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">kullanıcının hissettiği</text>
<rect x="16" y="252" width="130" height="24" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="267.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">kuyruk + tokenize</text>
<rect x="146" y="252" width="210" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="251.0" y="267.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">prefill + ilk decode</text>
<rect x="360" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="394.4" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="428.8" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="463.2" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="497.6" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="532" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="566.4" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="600.8" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="635.2" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="669.6" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="700.0" y="268" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">…</text>
<path d="M16 286 H356" marker-start="url(#il-arr)" marker-end="url(#il-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="186" y="302" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">TTFT</text>
<path d="M360 286 H390" marker-start="url(#il-arr)" marker-end="url(#il-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="392" y="302" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">TPOT = token&#x27;lar arası süre</text>
</svg>

**Birinci aşama — Tokenizer:** Kullanıcının metni model sözlüğündeki sayısal kimliklere dönüştürülür. Bu aşama çoğu serviste CPU üzerinde de gerçekleştirilebilir. Özellikle çok küçük isteklerde, tokenization ve sunucu tarafındaki kuyruk yönetimi toplam gecikmede hissedilebilir.

**İkinci aşama — Scheduler:** Sistemde yalnızca bir istek yoksa engine farklı kullanıcıları bir araya getirir. Burada **continuous batching**, yani her decode iterasyonunda yeni istekleri uygun şekilde batch'e alma yöntemi önem kazanır. Amaç GPU'nun ağırlıkları okurken yaptığı işi artırmaktır. Buna karşılık daha fazla kullanıcı aynı anda hizmet aldığında KV cache alanı ve kuyruk gecikmesi de büyüyebilir.

**Üçüncü aşama — Prefill:** Kullanıcının 2.000 token'ı modelin Transformer katmanlarından geçirilir. Matrisler görece büyük olduğu için Tensor Core kapasitesinden daha iyi yararlanma fırsatı vardır. Bu aşama özellikle **TTFT (Time to First Token)** üzerinde etkili olabilir. TTFT yalnızca prefill süresi değildir; kuyruk, veri hazırlama ve ilk decode adımını da içerir.

**Dördüncü aşama — Decode:** İlk token oluşturulduktan sonra sonraki token üretilir. Geçmiş token'ların key/value tensörlerini yeniden hesaplamak yerine KV cache'ten okuruz. Batch küçükse ağırlıkları defalarca HBM'den okumak baskın maliyet olabilir. Batch veya context büyüdükçe attention ve KV cache erişimi de belirleyici hale gelebilir.

**Beşinci aşama — Yanıtın akışı:** Her token kullanıcıya aktarıldıkça algılanan hız **TPOT (Time per Output Token)** veya token/s olarak ölçülebilir. Ama ortalamalar tek başına yeterli değildir: Sistem yük altında p95/p99 gecikmeyi nasıl koruyor? Birkaç kullanıcı çok uzun context gönderdiğinde kısa istekler gereksiz bekliyor mu?

Bu yolculuğun sonunda üç farklı optimizasyon problemimiz vardır. **İlk yanıtın gecikmesini** azaltmak istiyorsak prefill, kuyruk ve ilk token yolunu inceleriz. **Üretim hızını** artırmak istiyorsak decode, HBM, KV cache ve batching'e bakarız. **Aynı anda daha çok kullanıcı** desteklemek istiyorsak kapasiteyi, KV cache yerleşimini, gerektiğinde quantization'ı ve tail latency'yi birlikte değerlendiririz.

> **Akılda kalacak ayrım:** Training'de hedef çoğu zaman birim zamanda daha çok *eğitim token'ı* işlemektir. Online inference'ta ise throughput kadar kullanıcının ilk token'a ne zaman ulaştığı ve yanıtı ne kadar akıcı aldığı da önemlidir.

---

### 4.9. KV cache'in hesabını yapalım: Aynı model neden on kullanıcıda rahat, yüz kullanıcıda zorlanır?

Bir çağrı merkezinde aynı anda birden çok müşteriyle konuştuğunuzu düşünün. Her müşteri için konuşmanın geçmişini hatırlamanız gerekiyor. İlk müşterinin geçmişi birkaç cümle; diğerininki yüzlerce sayfa. Yeni müşteri geldikçe yalnızca konuşmayı işlemek değil, **geçmiş konuşmaların işlenmiş temsilini saklamak** da maliyetli hale gelir.

Transformer'da **KV cache (Key–Value cache)** bunu yapar: Decode sırasında önceki token'ların attention için gerekli key ve value tensörlerini yeniden hesaplamamak üzere saklar. Bu bir metin veritabanı değildir; modele ve katmana özgü sayısal tensörlerden oluşur. Model ağırlıklarından farklı olarak **isteğe ve bağlam uzunluğuna göre büyür**.

Bir modelin her Transformer katmanı için token başına tuttuğu KV verisini yaklaşık şöyle hesaplayabiliriz:

$$
M_{\mathrm{KV/token}}=2\times L\times H_{\mathrm{KV}}\times d_{\mathrm{head}}\times b
$$

- $2$: Key ve Value olmak üzere iki tensör.
- $L$: Transformer katmanı sayısı.
- $H_{\mathrm{KV}}$: KV head sayısı; attention head sayısıyla mutlaka aynı değildir.
- $d_{\mathrm{head}}$: Her head'in boyutu.
- $b$: Saklanan her elemanın byte sayısı; örneğin BF16 için 2 byte.

**Somut örnek:** Llama 3.1 70B benzeri, 80 katmanlı, 8 KV head'li, head başına 128 boyutlu ve BF16 KV cache kullanan bir model için:

$$
M_{\mathrm{KV/token}}=2\times80\times8\times128\times2
=327\,680\;\mathrm{byte}\approx320\;\mathrm{KiB}
$$

8.192 token'lık bir konuşmanın yalnızca KV cache'i yaklaşık $320\;\mathrm{KiB}\times8192=2.5\;\mathrm{GiB}$ eder. 32.768 token'da yaklaşık **10 GiB** olur. Bu değerler **mantıksal toplam KV verisi** içindir; TP/CP arasında parçalama, blok tahsisi, quantization, padding ve uygulama overhead'i fiziksel GPU başına tüketimi değiştirir. [K21]

Düşünün: 16 eşzamanlı uzun konuşma varsa yalnızca KV cache için onlarca veya yüzlerce GiB gerekir. Üstelik aynı GPU'da model ağırlıkları, çalışma tamponları ve geçici aktivasyonlar da yer kaplar. Yani `nvidia-smi` ekranında GPU hesaplaması düşük olsa bile yeni isteğe yer kalmayabilir.

<svg viewBox="0 0 720 244" role="img" aria-label="8.192 token bağlamlı 70B model için yığılmış çubuklar. Token başına KV 2 × 80 × 8 × 128 × 2 byte = 320 KiB; istek başına 2,5 GiB yani 2,68 GB. 10 kullanıcıda 140 GB ağırlık + 26,8 GB KV ≈ 167 GB; çalışma alanı hariç, 4 × H100 = 320 GB&#x27;lık öğretici havuza sığar. 100 kullanıcıda 140 + 268,4 GB ≈ 408 GB; sığmaz." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="kg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Aynı 70B model: 8.192 token bağlamlı 10 ve 100 kullanıcı</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Token başına KV = 2 × 80 × 8 × 128 × 2 B = 320 KiB → 8K istek başına 2,5 GiB (2,68 GB) · öğretici havuz: 4 × H100 = 320 GB</text>
<text x="16" y="94" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">10 kullanıcı</text>
<rect x="130" y="76" width="161.778" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="210.88888888888889" y="95.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">ağırlık 140</text>
<rect x="291.778" y="76" width="30.9689" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="330.74666666666667" y="88" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">167 GB</text>
<text x="330.74666666666667" y="104" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">sığar (+ çalışma alanı?)</text>
<text x="324.74666666666667" y="122" text-anchor="start" style="fill:var(--c-warn);font-size:11px">KV 26,8</text>
<text x="16" y="156" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">100 kullanıcı</text>
<rect x="130" y="138" width="161.778" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="210.88888888888889" y="157.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">ağırlık 140</text>
<rect x="291.778" y="138" width="310.151" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="446.85333333333335" y="157.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">KV 268,4</text>
<text x="609.9288888888889" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">408 GB</text>
<text x="609.9288888888889" y="166" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">sığmaz</text>
<line x1="499.778" y1="60" x2="499.778" y2="202" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="499.77777777777777" y="216" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">4 × 80 GB = 320 GB</text>
<text x="130.0" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="245.55555555555554" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100</text>
<text x="361.1111111111111" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">200</text>
<text x="476.66666666666663" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">300</text>
<text x="592.2222222222222" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">400</text>
<text x="690" y="232" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">GB</text>
</svg>

**GQA (Grouped-Query Attention)** bu hesabı neden değiştirir? Klasik multi-head attention'da query head sayısına yakın sayıda ayrı key/value head bulunabilir. GQA ise birden fazla query head'in daha az sayıdaki KV head'i paylaşmasını sağlar. Sezgisel olarak farklı çalışanlar aynı dosya dolabını kullanır. Böylece KV cache boyutu azalır; fakat attention davranışı ve kalite etkisi modelin eğitimiyle birlikte ele alınmalıdır. [K21]

> **Mimari karar:** Inference GPU kapasitesini yalnızca `model size / GPU memory` hesabıyla tahmin etmeyin. En azından **ağırlıklar + KV cache + geçici çalışma belleği + güvenlik payı** hesabını, gerçek prompt/output uzunluğu dağılımıyla yapın.

### 4.10. PagedAttention ve prefix caching: KV belleğini daha akıllı nasıl yönetiriz?

Bir kütüphanede her okuyucuya boşlukları önceden ayrılmış, devasa ve bitişik bir raf tahsis ettiğinizi düşünün. Okuyucuların çoğu o kadar kitap istemez; fakat rafların büyük bölümü yine de rezerve edilmiş olur. **Bellek parçalanması (fragmentation)** ve boşa ayrılan kapasite de buna benzer.

**PagedAttention**, KV cache'i küçük bloklar hâlinde ele alarak fiziksel GPU belleğinde ardışık olmayan alanlarda saklanmasına olanak tanır. Model mantıksal olarak bir token dizisi görürken, bellek yöneticisi hangi blokların nerede olduğuna karar verir. Bu yaklaşım özellikle çok sayıda, farklı uzunlukta isteğin birlikte sunulmasında işe yarar. [K22]

İkinci yöntem **prefix caching**: Birden fazla isteğin başlangıcı aynıysa, ortak prefix için hesaplanmış KV blokları yeniden kullanılabilir. Örneğin her isteğin başında aynı uzun sistem talimatı veya aynı belge bulunuyorsa, o bölümü defalarca prefill etmek yerine önbellekten faydalanılabilir.

İki fikri aynı resimde görelim:

<svg viewBox="0 0 720 258" role="img" aria-label="İki istek ortak bir prefix paylaşır. İstek A&#x27;nın mantıksal KV blokları P0, P1, A2 ve A3; istek B&#x27;ninkiler P0, P1 ve B2&#x27;dir. Blok tablosu her mantıksal bloğu HBM&#x27;deki bir fiziksel yuvaya eşler: P0 satır 0 sütun 3&#x27;e, P1 satır 2 sütun 2&#x27;ye, A2 satır 0 sütun 1&#x27;e, A3 satır 1 sütun 4&#x27;e, B2 satır 1 sütun 0&#x27;a; diğer yuvalar boştur. İki istek de aynı fiziksel P0 ve P1 bloklarına işaret ettiği için ortak prefix bir kez prefill edilir. Blokların bitişik olması gerekmez, bu da parçalanmayı azaltır. Tenant&#x27;lar arası paylaşım çitlenmelidir (ör. vLLM cache_salt); decode işi ve token başına K/V yok olmaz." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="pg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="pg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">PagedAttention + prefix caching: mantıksal dizi ≠ fiziksel yerleşim</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">mantıksal KV blokları (istek başına)</text>
<text x="430" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">fiziksel GPU belleği (HBM)</text>
<text x="16" y="86" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">istek A</text>
<rect x="100" y="70" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="156" y="70" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="182.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="212" y="70" width="52" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="238.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A2</text>
<rect x="268" y="70" width="52" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A3</text>
<text x="16" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">istek B</text>
<rect x="100" y="134" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="156" y="134" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="182.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="212" y="134" width="52" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="238.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B2</text>
<rect x="430" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="456.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="486" y="62" width="52" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="512.0" y="81.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A2</text>
<rect x="542" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="568.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="598" y="62" width="52" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="624.0" y="81.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="654" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="680.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="430" y="98" width="52" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="456.0" y="117.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B2</text>
<rect x="486" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="512.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="542" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="568.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="598" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="624.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="654" y="98" width="52" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="680.0" y="117.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A3</text>
<rect x="430" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="456.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="486" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="512.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="542" y="134" width="52" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="568.0" y="153.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="598" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="624.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="654" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="680.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="330" y="64" width="84" height="100" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="372" y="82" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">blok tablosu</text>
<text x="372" y="98" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">P0 → (0,3)</text>
<text x="372" y="111" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">P1 → (2,2)</text>
<text x="372" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">A2 → (0,1)</text>
<text x="372" y="137" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">A3 → (1,4)</text>
<text x="372" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">B2 → (1,0)</text>
<line x1="316" y1="114" x2="328" y2="114" marker-end="url(#pg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="414" y1="114" x2="428" y2="114" marker-end="url(#pg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="182" width="688" height="64" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="202" text-anchor="start" style="fill:var(--c-text);font-size:12px">Ortak prefix (P0, P1): iki istek aynı fiziksel bloklara işaret eder → prefill bir kez yapılır.</text>
<text x="28" y="220" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bloklar bitişik olmak zorunda değil → önden dev rezervasyon yok, daha az parçalanma.</text>
<text x="28" y="238" text-anchor="start" style="fill:var(--c-danger);font-size:12px">Tenant&#x27;lar arası paylaşım çitlenmeli (ör. vLLM cache_salt); decode işi ve token başına K/V yok olmaz.</text>
</svg>

Fakat iki sınırı unutmayalım:

1. Prefix caching esas olarak **ortak prompt bölümünün prefill hesaplamasını** azaltır; yeni çıktı token'larının decode işini ortadan kaldırmaz.
2. Kurumsal çok kullanıcılı sistemde cache paylaşımı yalnızca performans kararı değildir. **Tenant isolation (kiracı yalıtımı)** gerektirir. Ayrı müşterilerin gizli bağlamları istemeden birbirine açılmamalı; önbellek anahtarları ve izin sınırları bu gereksinime göre tasarlanmalıdır. vLLM, örneğin `cache_salt` ile farklı tenant’ların aynı prefix bloklarını paylaşmasını sınırlandırmayı belgeliyor. Bu ayrım yalnızca verinin yanlış kullanıcıya gösterilmesi riskinden ibaret değildir; önbellek isabetine bağlı zamanlama farkları da bazı koşullarda yan kanal bilgisi sızdırabilir. [K22]

**Continuous batching**, farklı kullanıcı isteklerinden hazır token üretim adımlarını dinamik olarak aynı GPU yürütmesine toplar. Bu, sabit bir batch'in tamamlanmasını beklemek zorunda kalmamak anlamına gelir. Ama aşırı agresif batching, tek kullanıcının gecikmesini artırabilir. İyi scheduler **throughput ile kuyruk gecikmesini beraber** optimize eder.

### 4.11. Prefill ve decode'u ayırmak: Disaggregated serving ne zaman mantıklı?

Önceki bölümde prefill ile decode'un farklı karakterde olduğunu gördük. Bir restoranda siparişleri hazırlayan mutfakla yemekleri masalara sırayla çıkaran servis ekibini düşünün: Biri toplu ve yoğun iş yapar, diğeri sürekli ve düzenli çalışmak ister. Tek ekip her ikisini yaptığında yoğun sipariş hazırlığı servis akışını aksatabilir.

**Disaggregated serving**, prefill ve decode için ayrı GPU worker havuzları kurar. İstek önce prefill havuzunda işlenir, ardından oluşturulan KV durumu decode havuzunun erişimine sunulur veya aktarılır. Böylece iki havuzun kapasitesini farklı iş yüklerine göre ayrı ayrı ölçeklemek mümkün olur. NVIDIA Dynamo ve TensorRT-LLM bu topolojiyi belgeliyor. [K23]

<svg viewBox="0 0 720 324" role="img" aria-label="İki panel. A, birleşik: router yükü, her biri hem prefill hem decode çalıştıran worker&#x27;lara dağıtır; KV transferi yoktur ama evreler birbirini etkiler. B, ayrıştırılmış: KV farkındalıklı gateway istekleri uzun girişlere uygun, hesap ağırlıklı prefill havuzuna gönderir; oluşan KV cache RDMA veya NVLink üzerinden uzun çıktılara uygun, HBM sınırlı ayrı decode havuzuna aktarılır; decode havuzu router&#x27;a boş KV kapasitesini, önbellekteki prefix&#x27;leri ve kuyruk derinliğini bildirir. Ayrıştırma ancak KV transferi ortadan kaldırdığı etkileşimden ucuzsa kazandırır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker><marker id="dg-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Birleşik ve ayrıştırılmış (disaggregated) serving</text>
<rect x="16" y="38" width="688" height="92" rx="6" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">A · birleşik: her worker prefill + decode yapar</text>
<rect x="28" y="70" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83.0" y="90.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Router</text>
<text x="83.0" y="105.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yük dağıtımı</text>
<rect x="176" y="68" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="184" y="76" width="66" height="34" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="217.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="256" y="76" width="72" height="34" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="292.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">decode</text>
<rect x="348" y="68" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="356" y="76" width="66" height="34" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="389.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="428" y="76" width="72" height="34" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="464.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">decode</text>
<rect x="520" y="68" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="528" y="76" width="66" height="34" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="561.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="600" y="76" width="72" height="34" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="636.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">decode</text>
<line x1="138" y1="93" x2="174" y2="93" marker-end="url(#dg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="704" y="58" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">+ KV transferi yok  − evreler birbirini etkiler</text>
<rect x="16" y="142" width="688" height="150" rx="6" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">B · ayrıştırılmış: ayrı prefill ve decode havuzları</text>
<rect x="28" y="186" width="110" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83.0" y="211.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Gateway / router</text>
<text x="83.0" y="226.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">KV farkındalıklı</text>
<rect x="176" y="176" width="170" height="76" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="261" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Prefill havuzu</text>
<text x="261" y="212" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">compute ağırlıklı · uzun ISL</text>
<rect x="190" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="227" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="264" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="301" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="496" y="176" width="170" height="76" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="581" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Decode havuzu</text>
<text x="581" y="212" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">HBM sınırlı · uzun OSL</text>
<rect x="510" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="547" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="584" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="621" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="138" y1="214" x2="174" y2="214" marker-end="url(#dg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="346" y1="214" x2="494" y2="214" marker-end="url(#dg-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<text x="420" y="206" text-anchor="middle" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">KV transferi</text>
<text x="420" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">RDMA / NVLink</text>
<path d="M581 252 V272 H83 V244" marker-end="url(#dg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="332" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">geri bildirim: boş KV kapasitesi, önbellekteki prefix&#x27;ler, kuyruk</text>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca KV transferi, ortadan kaldırdığı etkileşimden ucuzsa kazandırır; ikisini kendi ISL/OSL karışımınızla ölçün.</text>
</svg>

Bu tasarım her zaman daha hızlı değildir. Çünkü prefill ile decode arasındaki **KV transferi** yeni bir ağ bağımlılığı yaratır. Eğer transfer yavaşsa, tasarımın kazandırdığı kaynak uzmanlaşması, taşıma maliyetinde kaybolur. Bu yüzden karşılaştırılması gereken iki alternatif vardır:

| Topoloji | Güçlü tarafı | Zayıf tarafı | Hangi koşulda düşünülür? |
|---|---|---|---|
| **Aggregated**: Her worker prefill + decode | Daha az KV transfer karmaşıklığı | Aşamalar birbirini etkileyebilir | Küçük-orta ölçek, basit operasyon |
| **Disaggregated**: Ayrı prefill/decode havuzları | Bağımsız ölçekleme ve ayrışmış kaynak planı | KV transferi, ek ağ, orkestrasyon | Büyük concurrency, farklı ISL/OSL profilleri |

Burada **ISL (Input Sequence Length)** istek başına giriş token sayısı, **OSL (Output Sequence Length)** ise üretilen çıkış token sayısıdır. Uzun doküman analizi yapan sistemin ISL'si yüksek; uzun sohbet yanıtları üreten sistemin OSL'si yüksek olabilir. Bu iki iş yükünü aynı benchmark altında değerlendirmek yanıltıcıdır.

**KV-aware routing**, yalnızca en boş GPU'yu seçmek yerine hangi worker'ın ilgili prefix'i zaten belleğinde tuttuğunu ve anlık yükünü dikkate alır. Ama cache hit uğruna aşırı dolu worker'a bütün trafiği yönlendirmek de doğru değildir. [K23]

### 4.12. Speculative decoding, admission control ve kalite: Her token aynı maliyette gelmez

Bir asistanın uzun metni taslak olarak hazırlayıp editörün hızlıca kontrol ettiğini düşünün. **Speculative decoding** buna benzer: Küçük bir draft model veya alternatif bir öneri mekanizması, gelecek token'ları önceden tahmin eder; hedef model bunları uygun algoritmayla doğrular. Kabul edilen token'lar sayesinde hedef modelin seri decode adımlarının sayısı azaltılabilir.

Fakat draft model de hesaplama tüketir. Kabul oranı düşükse veya doğrulama maliyeti yüksekse hızlanma kaybolabilir. Ayrıca kullanılan doğrulama algoritmasının hedef dağılımı koruyup korumadığı açıklığa kavuşturulmalıdır. Bu, performans kazanımını kalite ve doğrulukla birlikte ölçmemiz gereken bir başka yerdir.

vLLM dokümantasyonu, speculative decoding yöntemlerinin kazancının trafik ve model koşullarına göre değiştiğini açıklıyor. [K19]

**Admission control** ise bir sistemin kaldırabileceğinden fazla işi körlemesine kabul etmemesidir. Örneğin şu anda 50 uzun context isteği çalışırken 51'incisini hemen almak, herkesin KV cache'ini sıkıştırıp p99 gecikmesini bozabilir. Bunun yerine istek kuyruğa alınabilir, ayrı worker'a yönlendirilebilir veya önceden ilan edilmiş kapasite politikası uygulanabilir.

> **AI Architect kontrolü:** Prefill ve decode için **ayrı throughput eğrileri**, KV cache doluluk ve tahliye (eviction) oranları, cache hit ratio, TTFT/TPOT p95-p99, kuyrukta bekleme, başarısız/iptal edilen istekler ve tenant bazlı maliyet görünürlüğü gerekir. Sadece "GPU utilization %95" yeterli değildir.

---



---

### İki istek, aynı model, bambaşka davranış

Bir kullanıcı “Merhaba!” yazıyor ve 40 token'lık yanıt istiyor. Başka bir kullanıcı 50 sayfalık bir sözleşme yüklüyor ve 1.000 token'lık yorum bekliyor. İkisi de aynı 70B modele gidiyor. Buna rağmen GPU açısından iki istek aynı problem değildir.

İlk isteğin **prefill** işi küçüktür ama decode boyunca onlarca ardışık üretim adımı gerekir. İkinci isteğin prompt'u çok uzundur; ilk token'dan önceki prefill çalışması ağırlaşır, KV cache büyür, sonra uzun decode aşaması başlar. Üstelik platform ikisini aynı anda sunuyorsa **scheduler** hangi isteğe ne kadar kaynak vereceğini seçmek zorundadır.

<svg viewBox="0 0 720 326" role="img" aria-label="Aynı 70B modelde iki istek için şematik zaman çizgileri. Kısa “Merhaba!” istemi kısa süre kuyrukta bekler, küçük bir prefill&#x27;le ilk token&#x27;a erken ulaşır, sonra 40 decode adımı çalışır; KV cache&#x27;i küçük ve kısa ömürlüdür. 50 sayfalık sözleşme uzun bir prefill gerektirir, ilk token&#x27;ı daha geç gelir, sonra 1.000 decode adımı boyunca büyük bir KV cache tutar. Renkler mantıksal prefill ve decode iş yüklerini ayırır; bunlar aynı GPU&#x27;yu paylaşabilir ya da ayrı havuzlarda çalışabilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tq-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tq-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tq-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Aynı 70B modelde iki istek (şematik, ölçekli değil)</text>
<text x="16" y="70" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">“Merhaba!”</text>
<text x="16" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kısa prompt · 40 token yanıt</text>
<rect x="196" y="56" width="46" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="219.0" y="74.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">kuyruk</text>
<rect x="246" y="56" width="30" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="280" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="292" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="304" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="316" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="328" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="340" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="352" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="364" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="376" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="388" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="400" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="412" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="424" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="436" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<text x="452" y="76" text-anchor="start" style="fill:var(--c-text-mute);font-size:13px"></text>
<text x="452" y="76" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">40 decode adımı</text>
<line x1="278" y1="50" x2="278" y2="96" style="stroke:var(--c-text);stroke-width:1.2"/>
<text x="278" y="110" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TTFT</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">50 sayfalık sözleşme</text>
<text x="16" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">uzun prompt · 1.000 token yanıt</text>
<rect x="196" y="148" width="46" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="219.0" y="166.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">kuyruk</text>
<rect x="246" y="148" width="170" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="331.0" y="167.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="420" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="432" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="444" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="456" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="468" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="480" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="492" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="504" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="516" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="528" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="540" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="552" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<text x="568" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:13px">…</text>
<text x="580" y="168" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">1.000 decode adımı</text>
<line x1="418" y1="142" x2="418" y2="188" style="stroke:var(--c-text);stroke-width:1.2"/>
<text x="418" y="202" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TTFT</text>
<text x="16" y="248" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">tutulan KV cache</text>
<rect x="196" y="238" width="40" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="242" y="251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">“Merhaba!”: küçük, kısa ömürlü</text>
<rect x="196" y="262" width="364" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="566" y="275" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sözleşme: büyük, 1.000</text>
<text x="566" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">adım boyunca tutulur</text>
<text x="16" y="314" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Renkler iki mantıksal iş yükünü (prefill, decode) ayırır; aynı GPU&#x27;da ya da ayrı havuzlarda çalışabilirler.</text>
</svg>

Burada prefill ve decode **ayrı fiziksel GPU'larda olmak zorunda değildir**. Şekil iki mantıksal iş yükünü ayırmak için iki renk kullanıyor; tek GPU'da birleşik servis de mümkündür. Ayrıştırılmış serviste ise KV transferinin kendi ağ ve zaman maliyeti ortaya çıkar.

### KV cache hesabını adım adım türetelim

Transformer decode aşamasında her yeni token için geçmişin Key ve Value temsilleri yeniden üretilmez; HBM'de ayrılan dinamik bir tamponda (*KV cache*) saklanır:

$$
M_{\mathrm{KV}}=2\times L\times H_{\mathrm{KV}}\times d_{\mathrm{head}}\times S\times b
$$

Bu formüldeki her çarpanın doğrudan fiziksel bir karşılığı vardır:
- Baştaki **2**, her katmanda bağımsız Key ve Value tensörlerinin tutulmasından gelir.
- $H_{\mathrm{KV}}$ (KV head sayısı), Grouped-Query Attention (GQA) sayesinde toplam Query head sayısından ($H_Q$) daha küçük seçilerek bu alanı 4 ila 8 kat daraltır.
- $d_{\mathrm{head}}$ ve $L$ katman sayısı modelin mimari boyutlarını belirler; $b$ ise sayısal hassasiyettir (BF16 için 2 byte, FP8 için 1 byte).
- $S$ (bağlam uzunluğu) arttıkça her kullanıcı oturumunun bellek ayak izi doğrusal olarak büyür.

Bölüm 4.9'daki 70B örneğinde gördüğümüz gibi, 8K bağlamlı tek bir isteğin $2{,}5\ \mathrm{GiB}$ tutması kâğıt üzerinde yönetilebilir görünür. Fakat model ağırlıkları statik kalırken KV cache'in eşzamanlı istek sayısıyla doğrusal katlanması ($100 \times 2{,}5 \approx 250\ \mathrm{GiB}$), bellek duvarının asıl belirleyicisinin ağırlıklar değil, canlı trafik olduğunu gösterir. [K21], [K49]


### PagedAttention neden sanal bellek fikrini ödünç aldı?

Bir otelde her misafire tahmini kalışının en uzun süresi kadar *kesintisiz* oda bloğu ayırdığınızı düşünün. Misafirler ne zaman çıkacağını bilmiyorsanız boş odalar arada parçalanır ve otel dolu görünürken kullanılamayan boşluklar oluşur. Geleneksel KV cache rezervasyonunda da uzun ve dinamik token dizileri yüzünden benzer bir problem görülür.

PagedAttention, KV cache'i sabit token bloklarına ayırarak mantıksal ardışıklığı fiziksel ardışıklık zorunluluğundan ayırır. Böylece cache, farklı fiziksel yerlerdeki blokları bir isteğin tek bir mantıksal geçmişi gibi kullanabilir. Bu, bellek parçalanmasını azaltır ve belirli koşullarda prefix paylaşımını kolaylaştırır. **Ama token başına gereken gerçek K/V bilgisini matematiksel olarak yok etmez.** [K49]

### Neden yüksek token/s, kullanıcıya hızlı hissedilen servis demek değil?

Diyelim ki bir sistem toplam 10.000 token/s üretiyor. Harika görünüyor. Fakat 500 kullanıcı aynı anda istekte bulunuyorsa, bazıları ilk token için uzun kuyruk bekleyebilir. Toplam throughput iyi iken **TTFT p99** kötü olabilir. Bu nedenle üç farklı şeyi birlikte düşünmek gerekir:

- **TTFT:** İstek geldiğinden ilk token'a kadar geçen süre; router ve queue süreleri de buna girer.
- **TPOT / ITL:** Token'lar arasındaki üretim gecikmesi; stream deneyimini belirler.
- **Goodput:** Belirlenmiş hizmet gecikmesi ve kalite sınırını karşılayan *faydalı* çıktı miktarı.

Yani en yüksek throughput'u veren batch boyutu, bazen en iyi kullanıcı deneyimini vermez. Bu çelişki bir hata değil, kuyruklanan sistemlerin doğal davranışıdır.


## 5. Kapasite neden yalnızca GPU sayısı değildir? HBM, KV cache, trafik ve güç

GPU satın alma kararını sezgisel bir tahminle değil, sınırlayıcı kaynakları ayrı ayrı modelleyerek verelim. Bir arabanın ağırlığını taşıyacak motor kadar, yakıt deposu ve soğutma kapasitesi de önemlidir. LLM servisinde bunun karşılığı **ağırlık belleği, KV cache, hesaplama, bellek bant genişliği ve trafik dalgalanmasıdır**.

### 5.1. Model belleği: İlk alt sınır

Yoğun bir model için kaba ağırlık hesabı:

$$
M_{\mathrm{weights}}\approx N_{\mathrm{params}}\times\frac{b}{8}
$$

Burada $b$, parametre başına bit sayısıdır. Örnek: 70 milyar parametre, BF16 (16 bit) ise yaklaşık **140 GB** ham ağırlık gerekir. Aynı model INT8'de yaklaşık 70 GB, dört bitte yaklaşık 35 GB olabilir. Ancak quantization ölçekleri, sıfır noktaları, padding, katmanların farklı hassasiyetleri ve runtime buffer'ları bu değerleri artırabilir.

Model 2×80 GB GPU belleğine teorik olarak sığsa da inference için gerekli çalışma alanını, CUDA graph'ları, aktivasyon tamponlarını ve KV cache'i de ayırmalıyız. Bu yüzden $M_{weights}<M_{GPU}$ şartı **gerekli ama yeterli değildir**.

### 5.2. KV cache: Görünmeyen kapasite kalemi

Bölüm 4'te token başına yaklaşık $320\ \mathrm{KiB}$ ve 8K bağlam için istek başına $2{,}5\ \mathrm{GiB}$ olarak hesapladığımız mantıksal KV ayak izi, sunucu kapasite planlamasında en değişken sınırlayıcıdır:

$$
M_{\mathrm{KV,total}} \approx N_{\mathrm{concurrent}} \times M_{\mathrm{KV/request}}
$$

Kapasite planlamasındaki en yaygın yanılgı, "Ağırlıkları belleğe sığdırdım, kalan alan serbesttir" varsayımıdır. 140 GB ağırlık barındıran 4 adet H100 (320 GB toplam HBM) havuzunda kâğıt üzerinde 180 GB boşluk görünür. Ancak 8K bağlamlı yalnızca 70 eşzamanlı oturum ($70 \times 2{,}5 \approx 175\ \mathrm{GB}$) bu payı tamamen tüketir.

Üstelik gerçek bir üretim ortamında:
1. **Bellek parçalanması (fragmentation):** PagedAttention kullanılsa dahi kısmen dolan bloklar nedeniyle %10–15 ek tampon gerekir.
2. **Aktivasyon ve çalışma payı:** Prefill sırasında matris çarpımlarının ihtiyaç duyduğu geçici aktivasyon tensörleri için güvenlik marjı ayrılmalıdır.
3. **Allocator rezervi:** PyTorch veya vLLM bellek yöneticilerinin OOM yaşamaması için taban bir boşluk bırakılmalıdır.

Dolayısıyla model ağırlıkları sabit bir zemin, KV cache ise trafiğe bağlı olarak hızla yükselen dinamik bir tavandır.


### 5.3. Serving kapasitesi: Token/sn ölçümünü isteğe dönüştürmek

Diyelim ki belirli bir model-gpu konfigürasyonunda **ölçtüğünüz** başarılı çıktı kapasitesi 6.000 output token/sn. Bir istek ortalama 300 token üretiyorsa, ilk bakışta:

$$
\lambda_{\mathrm{max}}\approx\frac{6000}{300}=20\ \text{istek/sn}
$$

diyebilirsiniz. Bu yalnızca **kaba üst sınır**dır: giriş token'larının prefill maliyeti, uzun prompt'lar, batch değişimi, routing, güvenlik kontrolleri ve p99 SLO henüz dahil değildir. Bu hesap size benchmark tasarlayacağınız noktayı verir; satın alma kararını tek başına vermez.

**Little's Law** ise ortalama eşzamanlılık için güçlü bir kontrol sağlar:

$$
L=\lambda W
$$

Saniyede 10 istek geliyorsa ve bir istek sistemde ortalama 4 saniye kalıyorsa, ortalama eşzamanlı iş sayısı yaklaşık **40** olur. Fakat kuyruk teorisinde ortalamalar, p99 patlamalarını garanti etmez. Üretimde hem gerçek zamanlı geliş eğrisini hem p50/p95 input-output token dağılımlarını test etmeliyiz.

### 5.4. Kapasite = en küçük sınır

Ulaşılabilir kapasiteyi kabaca şu sınırlardan en düşük olanı belirler:

$$
C_{\mathrm{serving}} \lesssim \min(C_{\mathrm{compute}},C_{\mathrm{HBM}},C_{\mathrm{KV}},C_{\mathrm{network}},C_{\mathrm{SLO}})
$$

Örneğin GPU yüksek token/s verebiliyor olabilir; fakat p95 TTFT sınırını aştığı için bu throughput'un önemli kısmı hizmet açısından işe yaramayabilir. Bu durumda hedef **maksimum çıktı** değil, **SLO içinde tamamlanan doğru yanıt**tır.

### 5.5. Kapasite rezervi neden lüks değildir?

On GPU'lu bir havuzda on GPU'nun tamamını tepe yükte kullanıyorsanız, tek GPU arızası veya kontrollü sürüm yükseltmesi hizmeti zor duruma sokabilir. Rezerv oranı konusunda evrensel yüzde yoktur; n+1 replika, fault-domain çeşitlendirme, gelen trafik patlamaları, açılış süresi ve SLO'dan hareket edilmelidir. Özellikle büyük modelin HBM'e yüklenme süresi, "gerektiğinde hemen bir pod açarız" varsayımını boşa çıkarabilir.

### 5.6. Güç ve soğutma: İkinci bir kapasite sınırı

128 adet sekiz GPU'lu sunucunun her birinin **varsayımsal** 10 kW çektiğini düşünürsek, yalnızca sunucuların güç bütçesi 1,28 MW'dır. NIC/switch, storage ve tesis kayıpları dahil değildir. PUE=1,3 varsayarsak toplam tesis enerji yükünün yaklaşık 1,66 MW mertebesine çıktığını tahmin edebiliriz. Bu **bir tasarım örneğidir; herhangi bir belirli veri merkezinin ölçümü değildir**.

PUE (Power Usage Effectiveness), tüm tesis enerjisinin IT ekipman enerjisine oranıdır. PUE düşük olabilir ama bir modelin token başına enerji maliyeti yine de yüksek olabilir; bu nedenle **kWh / başarılı milyon token** gibi hizmet metriği ayrıca tutulmalıdır.

---


## 6. Production inference: Bir isteğin kuyruktan son token’a yolculuğu

Bir API çağrısının başarılı dönmesi, kullanıcının iyi bir deneyim yaşadığı anlamına gelmez. Kullanıcı için iki ayrı bekleme vardır: **İlk token ne zaman geliyor?** ve **geldikten sonra metin ne hızda akıyor?** Teknik terimlerle bunlar TTFT ve TPOT/ITL'dir.

### 6.1. TTFT'yi parçalara ayıralım

$$
T_{\mathrm{TTFT}}\approx T_{\mathrm{auth}}+T_{\mathrm{safety}}+T_{\mathrm{data}}+T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{first\ decode}}+T_{\mathrm{network}}
$$

Bu terimler uygulamaya göre kısmen üst üste binebilir. `T_data` yalnızca veri erişimi gereken isteklerde ortaya çıkar. Ayrıca kullanıcı ilk token'ı görmeden önce sunucunun streaming tamponlarını boşaltması veya proxy'nin cevabı tutması da gecikme yaratabilir.

Örnek: Auth 20ms, güvenlik 30ms, opsiyonel veri servisi 150ms, queue 500ms, prefill 650ms, ilk decode 50ms, ağ/proxy 50ms ise toplam yaklaşık 1,45 saniyedir. Tamamı temsili olup **bu waterfall'ın her satırı gerçek tracing ile ölçülmelidir**.

<svg viewBox="0 0 720 298" role="img" aria-label="Toplam 1,45 saniyelik temsili TTFT şelalesi: kimlik doğrulama 20 ms, güvenlik 30 ms, opsiyonel veri servisi 150 ms, kuyruk 500 ms, prefill 650 ms, ilk decode 50 ms ve ağ/proxy 50 ms. Kuyruk ve prefill toplamın %79&#x27;udur. Her satır gerçek tracing ile ölçülmelidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">1,45 sn&#x27;lik TTFT nereye gidiyor? (temsili şelale)</text>
<text x="180" y="59" text-anchor="end" style="fill:var(--c-text);font-size:12px">auth</text>
<rect x="190" y="47" width="6.34483" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="202.3448275862069" y="60" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">20 ms</text>
<text x="180" y="87" text-anchor="end" style="fill:var(--c-text);font-size:12px">güvenlik</text>
<rect x="196.345" y="75" width="9.51724" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="211.86206896551724" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">30 ms</text>
<line x1="196.345" y1="65" x2="196.345" y2="75" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="115" text-anchor="end" style="fill:var(--c-text);font-size:12px">veri servisi (opsiyonel)</text>
<rect x="205.862" y="103" width="47.5862" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="259.44827586206895" y="116" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">150 ms</text>
<line x1="205.862" y1="93" x2="205.862" y2="103" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="143" text-anchor="end" style="fill:var(--c-text);font-size:12px">kuyruk</text>
<rect x="253.448" y="131" width="158.621" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.6"/>
<text x="418.0689655172414" y="144" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">500 ms</text>
<line x1="253.448" y1="121" x2="253.448" y2="131" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="171" text-anchor="end" style="fill:var(--c-text);font-size:12px">prefill</text>
<rect x="412.069" y="159" width="206.207" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="624.2758620689656" y="172" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">650 ms</text>
<line x1="412.069" y1="149" x2="412.069" y2="159" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="199" text-anchor="end" style="fill:var(--c-text);font-size:12px">ilk decode</text>
<rect x="618.276" y="187" width="15.8621" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="640.1379310344828" y="200" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">50 ms</text>
<line x1="618.276" y1="177" x2="618.276" y2="187" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="227" text-anchor="end" style="fill:var(--c-text);font-size:12px">ağ / proxy</text>
<rect x="634.138" y="215" width="15.8621" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="656.0" y="228" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">50 ms</text>
<line x1="634.138" y1="205" x2="634.138" y2="215" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<line x1="190" y1="246" x2="650" y2="246" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="190" y1="246" x2="190" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="190.0" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0 sn</text>
<line x1="348.621" y1="246" x2="348.621" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="348.62068965517244" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0,5 sn</text>
<line x1="507.241" y1="246" x2="507.241" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="507.2413793103448" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1,0 sn</text>
<line x1="650" y1="246" x2="650" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="650.0" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1,45 sn</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Burada kuyruk + prefill = %79. Her satır gerçek tracing&#x27;le ölçülmeli; terimler pratikte örtüşebilir.</text>
</svg>

### 6.2. Kuyruk neden doğrusal büyümez?

Servis kapasitesine yaklaşıldığında bekleme süresi hızla artabilir. On kasalı bir restoran mutfağı ortalama altı siparişi rahat işleyebilir; dokuz siparişte ise küçük aksaklıklar bile kuyruğu büyütür. Inference için de **yüksek utilization ile düşük p99 gecikme** arasında çelişki doğabilir.

Bu nedenle autoscaler'ı yalnızca GPU utilization'a göre yönetmek iyi sonuç vermeyebilir. Kuyruk süresi, bekleyen token miktarı, aktif decode sequence'leri, KV cache boşluğu ve TTFT SLO burn rate gibi sinyalleri birlikte değerlendirmek daha anlamlıdır.

### 6.3. Goodput ve admission control

**Throughput**, toplam çıktı hacmidir. **Goodput** ise belirlenen iş hedefini (örneğin doğru sonuç, son kullanıcı gecikme sınırı, hata koşulu) karşılayan faydalı üretimdir. Kuyruğa her isteği sınırsız kabul etmek, throughput grafiğini bir süre yükseltirken gecikmeyi herkes için bozabilir.

**Admission control**, sisteme yeni isteğin alınıp alınmayacağını; öncelik, tenant kotası ve kapasiteye göre belirlemektir. Örnek: Uzun context kullanan offline batch isteğini kritik gerçek zamanlı kullanıcı isteklerinin kuyruğuna önceliksiz eklemek yerine başka havuza yönlendirmek veya ertelemek.

### 6.4. Dört trafik profili neden farklı sonuç verir?

| Profil | Prompt/Output | Testin amacı |
|---|---|---|
| Kısa sohbet | 256 / 128 token | Küçük prompt TTFT, düşük gecikme |
| Uzun RAG | 8k / 512 token | Prefill, retrieval ve uzun context |
| Uzun üretim | 1k / 4k token | Decode ve KV residency |
| Çok turlu tekrar | Prefix'i yüksek oranda aynı | Prefix caching ve cache isolation |

Bunlara video/görsel girdili çok modlu istek profili de eklenmelidir. Her profil için istek geliş hızı ayrı artışlarla denenir: 1, 2, 5, 10... req/s gibi; sürdürülemez noktayı, p95/p99 SLO ihlalini ve hata oranını bulun. Testler warmup, farklı random seed, farklı cache durumları ve farklı sürümlerde yinelenmelidir. MLPerf Inference senaryoları ve SGLang benchmark dokümanları karşılaştırma metodolojisi için iyi referanstır. [K42][K43]

### 6.5. PagedAttention, prefix caching ve disaggregation hangi sorunlara yanıt verir?

**PagedAttention**, KV cache'i bloklar halinde yöneterek bellek israfını azaltmaya yardımcı olur; sıfır bellek maliyetine dönüşmez. **Prefix caching**, aynı prefix üzerinde tekrar prefill yapma ihtiyacını azaltabilir, ancak yalnızca gerçekten ortak ve yetkili biçimde yeniden kullanılabilecek prefix'ler için uygundur.

**Disaggregated serving** ise prefill ve decode işlerini ayrı GPU havuzlarına koyar. Özellikle uzun prompt'lar ve yüksek concurrency için faydalı olabilir; ama KV'yi taşımanın maliyetini getirir. Az concurrency, kısa prompt veya hızlı KV aktarım altyapısı olmayan koşullarda birleşik worker daha iyi olabilir. NVIDIA Dynamo dokümantasyonu da bu trade-off'u açıkça belirtiyor. [K40]

<svg viewBox="0 0 720 412" role="img" aria-label="Serving için karar ağacı. Trace ve profil sonrası hangi p95/p99 SLO&#x27;nun bozulduğundan başlayın. TTFT geç ise: kuyruk büyüyorsa admission control ve sıcak replika; prefill baskınsa chunked prefill ve prefix caching; hâlâ yavaşsa kendi giriş/çıkış uzunluklarınızla ayrı prefill ve decode havuzlarını A/B test edin ve yalnızca kazanç KV transfer maliyetini aşıyorsa ayrıştırın, değilse birleşik kalın. TPOT yavaşsa sorun decode, HBM ve KV baskısındadır: paged KV ve sürekli batch kullanın ya da quantization veya speculative decoding ile token başına byte&#x27;ı azaltın. Her değişiklikten sonra kaliteyi ve p99&#x27;u yeniden ölçün." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="st-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="st-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="st-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker><marker id="st-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Hangi serving çözümü? Bozulan SLO&#x27;dan başlayın</text>
<rect x="250" y="40" width="220" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="360.0" y="62.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Hangi p95/p99 SLO bozuluyor?</text>
<text x="360.0" y="77.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">önce trace + profil</text>
<path d="M300 90 V108 H140 V122" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M420 90 V108 H560 V122" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="124" width="200" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="140.0" y="146.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">TTFT (ilk token geç)</text>
<text x="140.0" y="161.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kuyruk mu, uzun prefill mi?</text>
<rect x="460" y="124" width="200" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="560.0" y="146.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">TPOT (akış yavaş)</text>
<text x="560.0" y="161.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">decode, HBM, KV baskısı</text>
<path d="M90 174 V192" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M190 174 V192" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="16" y="194" width="140" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">kuyruk büyüyor</text>
<text x="86.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">admission control,</text>
<text x="86.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">sıcak replika</text>
<rect x="164" y="194" width="150" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="239.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">prefill baskın</text>
<text x="239.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">chunked prefill,</text>
<text x="239.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">prefix caching</text>
<path d="M239 256 V274" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="150" y="276" width="178" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="239.0" y="298.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">hâlâ yavaş? ayrı havuz dene</text>
<text x="239.0" y="313.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kendi ISL/OSL&#x27;nizle A/B</text>
<path d="M190 326 V344 H100 V354" marker-end="url(#st-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<path d="M290 326 V344 H380 V354" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="20" y="356" width="160" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="100.0" y="375.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">kazanç &gt; KV maliyeti</text>
<text x="100.0" y="390.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">→ ayrıştır</text>
<rect x="300" y="356" width="160" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="380.0" y="375.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">kazanç ≤ KV maliyeti</text>
<text x="380.0" y="390.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">→ birleşik kal</text>
<path d="M510 174 V192" marker-end="url(#st-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<path d="M610 174 V192" marker-end="url(#st-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="470" y="194" width="112" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="526.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">batch / KV</text>
<text x="526.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">paged KV,</text>
<text x="526.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">sürekli batch</text>
<rect x="590" y="194" width="114" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="647.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">byte / token</text>
<text x="647.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">quantization,</text>
<text x="647.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">speculative</text>
<text x="704" y="290" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">her değişiklikte kaliteyi</text>
<text x="704" y="305" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">ve p99&#x27;u yeniden ölç</text>
</svg>

### 6.6. GPU autoscaling neden CPU autoscaling gibi değildir?

Bir GPU pod'unun ayağa kalkması; scheduler'ın kaynak bulması, image çekilmesi, model ağırlıklarının storage'dan okunması, NCCL/engine initialization, bellek tahsisi ve olası kernel warmup gibi aşamalardan geçebilir. Trafik 3 saniyede patlıyorsa kapasiteyi o anda sıfırdan açarak p99 hedefini korumak mümkün olmayabilir. **Warm replica**, model cache, proactive scale ve admission control birlikte tasarlanmalıdır.

### 6.7. Streaming güvenliği ve tekrar deneme

Bir streaming cevapta ilk 200 token kullanıcıya ulaştıktan sonra bağlantı koparsa, "retry" işlemi otomatik olarak aynı üretimi kaldığı yerden sürdürmez. Aynı request'in tekrar hesaplanması, yeniden ücretlendirilmesi veya kullanıcıya çift işlem yapılması riski vardır. Bu nedenle **request ID**, idempotency policy, cancellation propagation, timeout bütçesi ve kısmi çıktı işleme sözleşmesi önceden tanımlanmalıdır. Agent veya harici araç çalıştıran sistemlerde bu konu daha da kritiktir.

---


## 7. VLM, VQA, görsel ve video: Text-only kapasite hesabının bittiği yer

Sadece LLM çalıştıran bir kümede girdi genellikle token dizisidir. **Vision-Language Model (VLM)** ve **Visual Question Answering (VQA)** ise aynı çıktıya farklı bir yolculukla ulaşır: görsel ya da video açılır, yeniden boyutlandırılır, patch/crop'lara bölünebilir, vision encoder çalışır, görüntü temsilleri dil modeliyle birleştirilir ve cevap üretilir.

Bir hastane raporu veya teknik çizim üzerinde soru sorulduğunu hayal edin. Sistemin darboğazı yalnızca decoder'da oluşmayabilir. Yüksek çözünürlüklü görselin decode edilmesi, preprocessing, patch sayısı ve vision encoder hesaplama maliyeti de TTFT'yi artırabilir.

### 7.1. Bir görsel neden tek token değildir?

Bazı VLM'lerde görüntü birçok patch'e ayrılır; bazıları dinamik çözünürlük, çoklu crop veya image tiling kullanır. Örneğin 1024×1024 görüntüyü 16×16 patch boyutunda **tek düzlemli öğretici bir şemayla** düşünürseniz 64×64=4.096 patch elde edersiniz. Fakat gerçek mimariler patch merging, pooling, crop seçimi ve farklı tokenlaştırma yöntemleri kullanabileceği için bu sayı doğrudan "4.096 LLM token" demek değildir.

**Tasarım sonucu:** "Model 8k context destekliyor" ifadesi, 8k metin token + sınırsız görsel kullanılabilir anlamına gelmez. Modelin işlemcisi (processor), görüntü resolution sınırı, visual token bütçesi ve multimodal attention düzeni incelenmelidir. Hugging Face dokümantasyonu da metin tokenizer'ından farklı olarak multimodal `Processor` yapısını tanımlar. [K44]

### 7.2. VLM request waterfall

<svg viewBox="0 0 720 246" role="img" aria-label="VLM istek şelalesi: medya girdisi decode edilir, frame, crop veya tile&#x27;lara örneklenir, patch&#x27;ler üzerinde çalışan ViT vision encoder&#x27;dan geçer, dil modelinin uzayına projekte edilir; ardından dil modeli metin ve görsel token&#x27;ları prefill eder ve yanıtı decode eder. Decode&#x27;dan önceki her şey TTFT&#x27;ye yazılır. Görsel örneği: 1024 × 1024 piksel ve 16 × 16 patch, tek düzlemli öğretici modelde 64 × 64 = 4.096 patch eder; merging ve pooling gerçek LLM token sayısını değiştirir. Video örneği: 10 saniyelik klip 2 fps&#x27;te 20, 30 fps&#x27;te 300 frame&#x27;dir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="vl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="vl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="vl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">VLM istek şelalesi: ilk metin token&#x27;ından önceki iş</text>
<rect x="16" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="70.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Medya girdisi</text>
<text x="70.0" y="76.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">görsel / video</text>
<text x="70.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">decode, EXIF</text>
<rect x="132" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="186.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Örnekleme</text>
<text x="186.0" y="76.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">frame, crop,</text>
<text x="186.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tile</text>
<line x1="124" y1="71" x2="131" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="248" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="302.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Vision encoder</text>
<text x="302.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">patch&#x27;lerde ViT</text>
<line x1="240" y1="71" x2="247" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="418.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Projeksiyon</text>
<text x="418.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">görsel → LM uzayı</text>
<line x1="356" y1="71" x2="363" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="480" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="534.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LM prefill</text>
<text x="534.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">metin + görsel token</text>
<line x1="472" y1="71" x2="479" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="596" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="650.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LM decode</text>
<text x="650.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yanıt token&#x27;ları</text>
<line x1="588" y1="71" x2="595" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M16 112 H592" marker-start="url(#vl-arr)" marker-end="url(#vl-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="306.0" y="128" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">bunların hepsi TTFT&#x27;ye yazılır</text>
<rect x="16" y="142" width="340" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Görsel: “görsel başına 1 token” değil</text>
<text x="28" y="182" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">1024 × 1024 px ÷ 16 × 16 patch</text>
<text x="28" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">= 64 × 64 = 4.096 patch (tek düzlemli öğretici)</text>
<text x="28" y="220" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">merging/pooling LLM token sayısını değiştirir</text>
<rect x="364" y="142" width="340" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Video: “10 sn klip” yük tanımı değil</text>
<text x="376" y="189" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">2 fps</text>
<rect x="430" y="179" width="12" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="448.0" y="190" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">20 frame</text>
<text x="376" y="215" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">30 fps</text>
<rect x="430" y="205" width="180" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="616.0" y="216" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">300 frame</text>
</svg>

Bu akış her VLM ailesinde birebir aynı değildir; örneğin fusion noktası, cross-attention ve token entegrasyonu mimariye göre değişir. Fakat mimari kapasite incelemesi için doğru soruları hatırlatır.

### 7.3. Çok modlu serving için yeni metrikler

- Görüntü başına **piksel**, patch/crop sayısı ve görsel token eşdeğeri.
- Video için **fps, örneklenen frame sayısı, video uzunluğu** ve decode/IO gecikmesi.
- Image preprocessor CPU süresi, varsa hızlandırılmış decoder kullanımı, vision encoder süresi.
- Dil modeli prefill/decode ve KV cache; request başına toplam bellek.
- Çıktı kalitesi için **VQA accuracy**, OCR/readability, hallucination ve görev temelli skorlar.

Video için "10 saniyelik klip" yeterli yük tanımı değildir; 2 fps örnekleyerek 20 frame almakla 30 fps üzerinden 300 frame işlemek bambaşka operasyon maliyeti doğurabilir. Modelin eğitildiği frame ve temporal context sınırları da kaliteyi etkiler. [K44]

### 7.4. Multimodal quantization ne zaman fayda sağlar?

BF16→INT8/INT4 PTQ ile LLM decoder ağırlık belleği azalabilir. Ancak vision encoder, projection layer, normalization, attention ve KV cache ayrı dtypes kullanabilir. "Model 4-bit" ifadesi, bütün multimodal pipeline'ın aynı bit genişliğinde çalıştığı anlamına gelmez.

**Ölçülmesi gerekenler:** GPU başına weight belleği; görüntü başına toplam peak HBM; end-to-end TTFT; output TPOT; görüntü encoder süresi; VQA kalite kaybı; farklı görüntü çözünürlüklerinde stabilite. PTQ çalışmasında aynı soruyu hem text-only hem image+text koşullarında ölçmek, hızlanmanın hangi modülden geldiğini ayırmayı sağlar.

### 7.5. Güvenlik ve veri hakları

Görüntü dosyaları EXIF metadata taşıyabilir; gerçek kurum belgeleri kişisel veri içerebilir. Depolama süresi, görüntüden çıkarılan metin/OCR, ara embedding ve debug log'lar aynı veri sınıflandırma politikalarına tabi olmalıdır. VLM çıktısının iyi olması, dosyanın lisanslı ve yetkili olduğu anlamına gelmez.

---




---

**KISIM III — Birden çok GPU nasıl ortak çalışır?**

*Tek GPU iyi çalıştığında bile binlercesi neden bekleyebilir? Haberleşmenin fiziksel ve algoritmik tarafını ayıracağız.*


## 8. Bir H100'den sekiz GPU'lu DGX sunucusuna

Artık tek GPU içindeki hesaplama ile veri hareketini anlayabiliyoruz. Şimdi modeli büyütelim.

70 milyar parametreli bir modelin BF16 ağırlıkları yaklaşık 140 GB tutar. Bu ağırlıkları, yalnızca 80 GB HBM kapasitesi bulunan tek bir H100 SXM GPU'ya bütünüyle koyamayız. Demek ki ya modeli daha küçük bir sayısal temsille sunmalıyız ya da belleği farklı GPU'lar arasında paylaştırmalıyız.

İkinci yöntemi seçtiğimiz anda yeni bir mühendislik problemi doğar: **GPU'lar birbiriyle ne kadar sık ve hangi hızda konuşacak?**

Bir H100'ün HBM'ine sığmayan modeli nasıl çalıştırırız? Ağırlıkları farklı GPU'lara parçalayabiliriz. Fakat bu noktadan itibaren matematiğin yanına **GPU'lar arasında verinin hareket etmesi** eklenir.

### 8.1. PCIe, NVLink, NVSwitch: Üç farklı rol

Bu üç terim sık sık yan yana geçer, ama aynı bağlantının üç adı değildir. Bir sunucuyu bina olarak düşünürseniz PCIe farklı odaları ve cihazları bağlayan genel koridor sistemi, NVLink GPU'lar arasındaki özel yüksek hızlı geçit, NVSwitch ise bu geçitleri çok sayıda GPU arasında düzenleyen kavşak gibidir.

**PCI Express (PCIe)** genel amaçlı bir donanım bağlantı standardıdır; CPU, GPU, NIC ve NVMe aygıtları aynı platform içinde PCIe topolojisinden yararlanabilir. PCIe Gen5 ×16 teorik yön başına yaklaşık 63 GB/s düzeyindedir; gerçek uygulama hızı kodlama ve protokol maliyetleri yüzünden daha düşüktür.

**NVLink** GPU'lar arasındaki yüksek hızlı doğrudan iletişim teknolojisidir. H100 SXM'de GPU başına toplam **450 GB/s tek yönlü**, **900 GB/s çift yön toplam** NVLink kapasitesi vardır. Bu iki sayı birbirinin alternatifi değil, **yön tanımı farklı aynı bağlantı kümesinin iki gösterimidir**. [K2]

**NVSwitch**, birçok GPU'nun NVLink üzerinden haberleşmesini düzenleyen switch donanımıdır. Bir DGX H100 sunucusunda sekiz GPU, NVSwitch ağı içinde iletişim kurar.

Kablolamanın ayrıntısı önemlidir. Her H100, 18 NVLink'ini dört NVSwitch'e 5 + 4 + 4 + 5 olarak dağıtır; bu yüzden herhangi iki GPU aynı hızda konuşur ve dört GPU'yu diğer dördünden ayıran kesitten $4\times450\ \mathrm{GB/s}\times2=3{,}6\ \mathrm{TB/s}$ geçer (biseksiyon bant genişliği). NVLink ağı kendiliğinden açılmaz: NVSwitch'li sistemlerde GPU'lar arası yollar ancak **Fabric Manager** servisi çalıştığında kurulur [K53]. Blackwell'le bu alan kasanın dışına taşar: GB200/GB300 NVL72 rafı 72 GPU'yu 18 NVSwitch çipiyle tek bir NVLink alanında birleştirir [K51].

<svg viewBox="0 0 720 350" role="img" aria-label="DGX H100&#x27;ün mantıksal topolojisi. Üstte dört NVSwitch çipi vardır; 80&#x27;er GB&#x27;lık sekiz GPU&#x27;nun her biri dördüne 25 GB/s&#x27;lik 18 NVLink ile, GPU başına yön başına 450 GB/s bağlanır. Altta PCIe Gen5 ağacı her GPU&#x27;yu kendi 400 Gb/s (yön başına 50 GB/s) ConnectX-7 NIC&#x27;ine ve oradan compute ağına bağlar. CPU 0, GPU ve NIC 0–3&#x27;e NUMA 0 olarak, CPU 1 ise 4–7&#x27;ye NUMA 1 olarak hizmet eder; CPU&#x27;lar UPI ile bağlıdır. Sunucuda ayrıca depolama ağı için iki çift portlu ConnectX-7, sekiz 3,84 TB NVMe, 2 TB RAM bulunur ve en fazla 10,2 kW çeker." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dx-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dx-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dx-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">DGX H100: mantıksal topoloji</text>
<rect x="96" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="158.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 0</text>
<rect x="229" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="291.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 1</text>
<rect x="362" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="424.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 2</text>
<rect x="495" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="557.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 3</text>
<line x1="127" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="127" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="127" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="127" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<text x="704" y="24" text-anchor="end" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">her GPU: 18 NVLink × 25 GB/s = 450 GB/s/yön</text>
<rect x="96" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="127.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<text x="127.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="162" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="193.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<text x="193.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="228" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="259.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<text x="259.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="294" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="325.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<text x="325.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="360" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="391.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 4</text>
<text x="391.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="426" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="457.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 5</text>
<text x="457.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="492" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="523.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 6</text>
<text x="523.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="558" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="589.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 7</text>
<text x="589.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="16" y="154" width="70" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="51" y="175" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 0</text>
<text x="51" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">56 çekirdek</text>
<rect x="634" y="154" width="70" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="669" y="175" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="669" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">56 çekirdek</text>
<line x1="86" y1="180" x2="325" y2="180" style="stroke:var(--c-warn);stroke-width:2"/>
<line x1="391" y1="180" x2="634" y2="180" style="stroke:var(--c-warn);stroke-width:2"/>
<line x1="127" y1="144" x2="127" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="127" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="193" y1="144" x2="193" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="193" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="259" y1="144" x2="259" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="259" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="325" y1="144" x2="325" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="325" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="391" y1="144" x2="391" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="391" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="457" y1="144" x2="457" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="457" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="523" y1="144" x2="523" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="523" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="589" y1="144" x2="589" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="589" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<rect x="100" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="127.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0</text>
<line x1="127.0" y1="242" x2="127.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="166" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="193.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 1</text>
<line x1="193.0" y1="242" x2="193.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="232" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="259.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 2</text>
<line x1="259.0" y1="242" x2="259.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="298" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="325.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 3</text>
<line x1="325.0" y1="242" x2="325.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="391.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 4</text>
<line x1="391.0" y1="242" x2="391.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="457.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 5</text>
<line x1="457.0" y1="242" x2="457.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="496" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="523.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 6</text>
<line x1="523.0" y1="242" x2="523.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="562" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="589.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 7</text>
<line x1="589.0" y1="242" x2="589.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="358.0" y="278" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">compute ağı: 8 × ConnectX-7 · her biri 400 Gb/s (50 GB/s/yön), her GPU&#x27;ya bir NIC</text>
<rect x="92" y="148" width="268" height="100" rx="8" style="fill:none;stroke:var(--c-warn);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="356" y="148" width="268" height="100" rx="8" style="fill:none;stroke:var(--c-warn);stroke-width:1;stroke-dasharray:4 3"/>
<text x="16" y="222" text-anchor="start" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">PCIe Gen5</text>
<text x="16" y="238" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:600">NUMA 0</text>
<text x="16" y="254" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">UPI ⇄ CPU 1</text>
<text x="704" y="222" text-anchor="end" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">PCIe Gen5</text>
<text x="704" y="238" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:600">NUMA 1</text>
<rect x="16" y="292" width="688" height="46" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="311" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Sunucuda ayrıca: 2 × çift portlu ConnectX-7 → depolama ağı · 8 × 3,84 TB NVMe · 2 TB RAM · en fazla 10,2 kW</text>
<text x="28" y="328" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GPU, diğer GPU&#x27;larla NVSwitch üzerinden, başka sunucularla kendi NIC&#x27;i (GPUDirect RDMA), host ile PCIe üzerinden konuşur.</text>
</svg>

Bu çizim fiziksel PCIe switch bağlantılarının eksiksiz şeması değildir. Ama hangi yolun neden var olduğunu anlamak için yeterlidir.

### 8.2. NIC ve GPUDirect RDMA nedir?

**NIC (Network Interface Card)** sunucunun ağ arayüzüdür. **RDMA (Remote Direct Memory Access)** uyumlu ağ üzerinden uzak belleğe CPU'nun her veri parçasını kopyalamasını gerektirmeden erişim yöntemidir. **GPUDirect RDMA**, desteklenen NIC ile GPU belleği arasında doğrudan DMA iletişimi sağlayabilir.

**GDS ile RDMA'yı karıştırmayın:** GPUDirect RDMA esas olarak ağ cihazı ↔ GPU belleği yoluyla; GPUDirect Storage ise depolama ↔ GPU belleği I/O yoluyla ilgilidir. Bir depolama çözümü RDMA'dan yararlanabilir, ancak ikisi aynı kavram değildir. [K6]

### 8.3. NUMA ve 'aynı sunucudayız, neden yavaş?' sorusu

Bir sunucunun aynı kasada bulunan parçaları her zaman birbirine eşit uzaklıkta değildir. Özellikle iki CPU soketli makinelerde bazı bellek alanları ve PCIe aygıtları bir sokete daha yakındır. “Zaten aynı sunucudayız” diyerek verinin hangi yoldan geçtiğini göz ardı etmek bu yüzden risklidir.

**NUMA (Non-Uniform Memory Access)**, çok soketli CPU sisteminde belleğe erişim maliyetinin hangi işlemciye ve hangi RAM bölgesine erişildiğine göre değişmesi anlamına gelir. Bir NIC veya GPU'nun PCIe yolu bir CPU soketine daha yakın olabilir. Yanlış CPU affinity, pinned memory ve NUMA yerleşimi gereksiz uzak bellek erişimleri oluşturabilir.

Bu nedenle bir DGX'te iki CPU bulunması yalnızca 'yönetim işlemcisi' rolü değildir: veri yükleme, bellek tahsisi, NIC işlemleri, storage akışı ve uygulama süreçleri de CPU–PCIe–NUMA yerleşiminden etkilenir.

### 8.4. Sayılar niçin mimariyi belirler?

<svg viewBox="0 0 720 286" role="img" aria-label="Yön başına bant genişliği, log ölçek. HBM3 tepe değeri yaklaşık 3.350 GB/s; GPU&#x27;nun kendi ağırlık ve aktivasyonlarını okumasıdır. NVLink, NVLink alanında GPU başına yön başına 450 GB/s verir; HBM&#x27;den 7,4 kat azdır. PCIe 5.0 ×16 teorik olarak yön başına yaklaşık 63 GB/s&#x27;dir, host ve PCIe aygıtları içindir; 53 kat azdır. NDR 400 Gb/s port, sunucular arası trafik için yön başına 50 GB/s hat hızı verir; HBM&#x27;den 67 kat azdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="bl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bant genişliği uçurumu: HBM&#x27;den her uzaklaşmada yol birkaç kat daralır</text>
<text x="16" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">HBM3</text>
<text x="16" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">tepe</text>
<rect x="150" y="48" width="383.579" height="24" rx="3" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="541.5790576486218" y="65" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">3.350 GB/s</text>
<text x="150" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU&#x27;nun kendi ağırlık/aktivasyon okuması</text>
<text x="16" y="116" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NVLink</text>
<text x="16" y="132" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">yön / GPU</text>
<rect x="150" y="102" width="251.139" height="24" rx="3" style="fill:var(--c-success);fill-opacity:.6"/>
<text x="409.1391862669635" y="119" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">450 GB/s</text>
<text x="150" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink alanında GPU ↔ GPU</text>
<text x="704" y="119" text-anchor="end" style="fill:var(--c-success);font-size:12px;font-weight:700">÷7,4 (HBM&#x27;e göre)</text>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">PCIe 5.0 ×16</text>
<text x="16" y="186" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">yön, teorik</text>
<rect x="150" y="156" width="121.428" height="24" rx="3" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="279.42766490529937" y="173" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">63 GB/s</text>
<text x="150" y="196" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">host ve PCIe aygıtları</text>
<text x="704" y="173" text-anchor="end" style="fill:var(--c-warn);font-size:12px;font-weight:700">÷53 (HBM&#x27;e göre)</text>
<text x="16" y="224" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NDR 400 Gb/s</text>
<text x="16" y="240" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">yön, hat hızı</text>
<rect x="150" y="210" width="106.18" height="24" rx="3" style="fill:var(--c-danger);fill-opacity:.6"/>
<text x="264.18039523127993" y="227" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">50 GB/s</text>
<text x="150" y="250" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sunucu ↔ sunucu, GPU–NIC trafiği</text>
<text x="704" y="227" text-anchor="end" style="fill:var(--c-danger);font-size:12px;font-weight:700">÷67 (HBM&#x27;e göre)</text>
<line x1="150" y1="258" x2="560" y2="258" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="150" y1="258" x2="150" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="150.0" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10</text>
<line x1="301.91" y1="258" x2="301.91" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="301.9098023843601" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100</text>
<line x1="453.82" y1="258" x2="453.82" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="453.8196047687201" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1.000</text>
<text x="560" y="276" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">GB/s (log)</text>
</svg>

Şekil bir **fiziksel yol kapasitesi merdivenidir**. HBM, NVLink, PCIe ve NIC rakamlarını aynı işin gerçekten alacağı toplam süre olarak yorumlayamayız. Mesela sunucuda sekiz ayrı NIC varsa toplam sunucu çıkış bant genişliği ile tek GPU'nun port kapasitesi farklıdır.

**Sezgi:** Tezgâhınızdan malzeme almakla, yan mutfağa gitmekle ve başka şehirdeki depoya ulaşmak aynı fiyatta olamaz. Modeli GPU'lara bölerken veri taşıma sıklığını bağlantı mesafesine göre yerleştirmelisiniz.

Merdiveni süreye çevirelim. Tek bir GPU'nun 70B modelin 140 GB'lık BF16 ağırlığını her yoldan bir kez taşıdığını düşünün, $t=140\ \mathrm{GB}/W$: HBM'den yaklaşık **42 ms**, NVLink'ten **0,31 sn**, PCIe'den **2,2 sn**, tek bir NDR porttan **2,8 sn** ve 25 GB/s'lik paylaşılan bir depodan **5,6 sn**. Aynı hesap decode hızına da tavan koyar. Batch 1'de decode her token için ağırlıkların tamamını HBM'den okur. TP = 8 ile ağırlıklar sekiz GPU'ya bölündüğünde token başına en az $140/(8\times3{,}35)\approx5{,}2$ ms gerekir: KV cache ve diğer okumalar hiç sayılmadan **saniyede ~190 token** tavanı.

---


## 9. DGX'ten SuperPOD'a: Ağ topolojisi neden 'kablo sayısından' fazlasıdır?

Sekiz GPU'yu tek bir sunucuda yüksek hızlı NVLink alanıyla birleştirdik. Şimdi 128 sunucu düşünelim. Her birindeki GPU'lar kendi işini yaparken zaman zaman başka sunuculardaki GPU'larla sonuç paylaşmak zorunda.

Bu durumda mesele artık birkaç kabloyu birbirine bağlamak değildir. Hangi sunucu kiminle ne kadar konuşacak, hangi trafik aynı yolu paylaşacak, bir switch dolduğunda hangi işler bekleyecek? **Network fabric**, yani hesaplama ağı, bu soruların fiziksel yanıtıdır.

### 9.1. Scale-up ve scale-out ayrımı

**Scale-up**, aynı sistem veya yüksek hızlı paylaşımlı GPU alanının kapasitesini artırmaktır. **Scale-out**, farklı sunucuları ağ üzerinden birlikte çalıştırarak sistemi büyütmektir.

H100 için yaygın pratik sınır, sekiz GPU'lu sunucunun NVSwitch alanıdır. Bunun üzerine çıkıldığında başka sunucularla bir **compute fabric** üzerinden haberleşme gerekir. Ancak 'TP asla sekiz GPU'nun dışına çıkamaz' ifadesi **yanlıştır**: sekiz üzerindeki TP teknik olarak yapılabilir; yalnızca gecikme ve bant genişliği maliyeti artabilir. NVIDIA Megatron belgeleri TP'yi farklı paralellik boyutlarıyla birlikte destekler. [K3]

### 9.2. Leaf ve spine: Verinin kime uğradığını bilmek

Bir şehri düşünün. Mahallenizin içindeki iki sokak arasında dolaşırken otoyola çıkmanız gerekmez. Ama başka bir semte gidecekseniz ana arterleri kullanmanız gerekebilir. Leaf–spine ağı da kabaca böyle çalışır: sunucular alt katmandaki leaf switch'lere bağlanır; farklı leaf grupları arasındaki trafik üst katmandaki spine üzerinden gidebilir.

**Leaf switch**, sunucu/NIC bağlantılarının bağlandığı alt katman anahtardır. **Spine switch** ise leaf katmanlarını birbirine bağlayan üst katmandır.

<svg viewBox="0 0 720 340" role="img" aria-label="DGX H100 SuperPOD compute ağı. Üstte 16 spine switch vardır. Altta dört SU bulunur; her birinde rail başına bir tane olmak üzere 0–7 numaralı sekiz leaf switch ve 32 DGX H100 vardır. Bir SU&#x27;daki leaf r, 32 sunucunun NIC r&#x27;sini bağlar; böylece SU içinde aynı rail trafiği tek leaf atlamasıyla gider. Her leaf her spine&#x27;a iki 400 Gb/s kabloyla bağlıdır, 32 port yukarı ve 32 port aşağı; ağ tasarımda 1:1&#x27;dir. Vurgulu yol, farklı SU&#x27;lar arasındaki trafiğin leaf, spine, leaf yolunu izlediğini gösterir. Dört SU 128 yuva verir; biri UFM&#x27;e ayrılır, geriye 127 DGX yani 1.016 GPU kalır. Depolama ayrı bir ağda çalışır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ft-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ft-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="ft-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">DGX H100 SuperPOD compute ağı: 32 leaf + 16 spine, rail-optimize</text>
<rect x="27" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="45.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S0</text>
<rect x="69" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="87.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S1</text>
<rect x="111" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="129.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S2</text>
<rect x="153" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="171.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S3</text>
<rect x="195" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="213.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S4</text>
<rect x="237" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="255.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S5</text>
<rect x="279" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="297.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S6</text>
<rect x="321" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="339.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S7</text>
<rect x="363" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="381.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S8</text>
<rect x="405" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="423.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S9</text>
<rect x="447" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="465.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S10</text>
<rect x="489" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="507.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S11</text>
<rect x="531" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="549.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S12</text>
<rect x="573" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="591.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S13</text>
<rect x="615" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="633.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S14</text>
<rect x="657" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="675.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S15</text>
<rect x="17" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="99" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 1</text>
<rect x="23" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="31.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="42.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="51.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="62" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="70.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="81.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="90.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="101" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="109.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="120.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="129.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="140" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="148.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="159.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="168.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="23" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="99.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="31.5" y1="200" x2="29" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="51" y1="200" x2="49" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="70.5" y1="200" x2="69" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="90" y1="200" x2="89" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="109.5" y1="200" x2="109" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="129" y1="200" x2="129" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="148.5" y1="200" x2="149" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="168" y1="200" x2="169" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<rect x="191" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="273" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 2</text>
<rect x="197" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="216.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="225.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="236" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="244.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="255.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="264.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="275" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="283.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="294.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="303.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="314" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="322.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="333.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="342.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="197" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="273.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="205.5" y1="200" x2="203" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="225" y1="200" x2="223" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="244.5" y1="200" x2="243" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="264" y1="200" x2="263" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="283.5" y1="200" x2="283" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="303" y1="200" x2="303" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="322.5" y1="200" x2="323" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="342" y1="200" x2="343" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<rect x="365" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="447" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 3</text>
<rect x="371" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="379.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="390.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="399.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="410" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="418.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="429.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="449" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="457.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="468.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="477.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="488" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="496.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="507.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="516.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="371" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="447.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="379.5" y1="200" x2="377" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="399" y1="200" x2="397" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="418.5" y1="200" x2="417" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="438" y1="200" x2="437" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="457.5" y1="200" x2="457" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="477" y1="200" x2="477" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="496.5" y1="200" x2="497" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="516" y1="200" x2="517" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<rect x="539" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="621" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 4</text>
<rect x="545" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="553.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="564.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="573.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="584" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="592.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="603.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="612.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="623" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="631.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="642.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="651.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="662" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="670.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="681.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="690.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="545" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="621.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="553.5" y1="200" x2="551" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="573" y1="200" x2="571" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="592.5" y1="200" x2="591" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="612" y1="200" x2="611" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="631.5" y1="200" x2="631" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="651" y1="200" x2="651" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="670.5" y1="200" x2="671" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="690" y1="200" x2="691" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="31.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<path d="M31.5 176 L255 70 L379.5 176" style="fill:none;stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="704" y="104" text-anchor="end" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">farklı SU → leaf → spine → leaf</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Bir SU&#x27;daki leaf r, 32 DGX&#x27;in NIC r&#x27;sini bağlar (rail r) → SU içinde aynı rail = tek leaf atlaması.</text>
<text x="16" y="310" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Her leaf: 32 port aşağı + 32 port yukarı (16 spine&#x27;ın her birine 2 × 400 Gb/s) → tasarımda 1:1, engellemesiz.</text>
<text x="16" y="328" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">4 SU × 32 yuva = 128; bir yuva UFM&#x27;e ayrılır → 127 DGX = 1.016 GPU. Depolama ağı ayrı bir ağdır.</text>
</svg>

Aynı leaf üzerindeki iki sunucu daha az switch geçişiyle konuşabilir. Farklı leaf'ler arasındaki yol çoğunlukla **leaf → spine → leaf** biçimindedir.

**Hop**, paketin geçtiği ağ anahtarı geçişidir. **Fat-tree / Clos**, çok sayıda sunucu arasında yüksek paralel yol kapasitesi oluşturmak için düzenlenen çok katmanlı topoloji ailesidir.

**Bisection bandwidth**, ağı iki parçaya ayırdığımızda bu iki yarı arasından aktarılabilecek toplam bant genişliğidir. **Non-blocking** veya **full-bisection** yapı, tasarlanan trafik örüntüsünde ağın bölümlerinin dar bir üst bağlantıda boğulmaması hedefidir. Gerçek uygulamada congestion, yönlendirme ve aşırı yüklenme yine görülebilir.

### 9.3. Rail optimization nasıl çalışır?

Ağ topolojisini biraz daha somutlaştıralım. Sekiz GPU'lu her sunucunun sekiz ayrı ağ bağlantısı olduğunu düşünün. Bunları gelişigüzel bağlarsanız GPU'ların aynı tür konuşmaları birbirini gereksiz yere engelleyebilir. Rail optimization, benzer konumdaki bağlantıları düzenli hatlar halinde örgütleyerek belirli trafik örüntülerini daha verimli taşımayı amaçlar.

Her sunucunun sekiz NIC'i ve sekiz GPU'su olduğunu düşünün. Aynı indeksteki NIC'leri mantıksal bir 'rail' üzerinde buluşturabilirsiniz.

<svg viewBox="0 0 720 342" role="img" aria-label="Rail-optimize kablolama. A, B ve C sunucularının her birinde GPU başına bir tane olmak üzere 0–7 numaralı sekiz NIC vardır. Her sunucunun NIC i&#x27;si leaf switch i&#x27;ye bağlanır: leaf 0 rail 0&#x27;dır, leaf 1 rail 1&#x27;dir ve böyle devam eder; üstteki spine katmanı leaf&#x27;leri bağlar. Vurgulu: A sunucusunun GPU 0&#x27;ı, B&#x27;nin GPU 0&#x27;ına NIC 0 ve leaf 0 üzerinden aynı SU içinde tek leaf atlamasıyla ulaşır. A&#x27;nın GPU 0&#x27;ından C&#x27;nin GPU 1&#x27;ine gitmek rail değiştirmektir: ya spine&#x27;a çıkılır ya da önce NVLink ile A&#x27;nın GPU 1&#x27;ine geçilip rail 1 kullanılır; NCCL buna PXN der." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="rl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Rail optimizasyonu: her sunucunun NIC i&#x27;si leaf i&#x27;de buluşur</text>
<rect x="96" y="40" width="600" height="24" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="396.0" y="56.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">spine katmanı (rail&#x27;ler / SU&#x27;lar arası trafik)</text>
<rect x="100" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="132.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 0</text>
<line x1="132" y1="86" x2="132" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="176" y="86" width="64" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="208.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 1</text>
<line x1="208" y1="86" x2="208" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="252" y="86" width="64" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="284.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 2</text>
<line x1="284" y1="86" x2="284" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="328" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="360.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 3</text>
<line x1="360" y1="86" x2="360" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="404" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="436.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 4</text>
<line x1="436" y1="86" x2="436" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="480" y="86" width="64" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="512.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 5</text>
<line x1="512" y1="86" x2="512" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="556" y="86" width="64" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="588.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 6</text>
<line x1="588" y1="86" x2="588" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="632" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="664.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 7</text>
<line x1="664" y1="86" x2="664" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="16" y="150" width="688" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="24" y="172" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">sunucu A</text>
<rect x="108" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="116.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">0</text>
<line x1="116" y1="156" x2="116" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="184" y="156" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="192.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">1</text>
<line x1="192" y1="156" x2="192" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="260" y="156" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="268.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">2</text>
<line x1="268" y1="156" x2="268" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="336" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="344.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">3</text>
<line x1="344" y1="156" x2="344" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="412" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="420.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">4</text>
<line x1="420" y1="156" x2="420" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="488" y="156" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="496.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">5</text>
<line x1="496" y1="156" x2="496" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="564" y="156" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="572.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">6</text>
<line x1="572" y1="156" x2="572" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="640" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="648.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">7</text>
<line x1="648" y1="156" x2="648" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="16" y="196" width="688" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="24" y="218" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">sunucu B</text>
<rect x="126" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="134.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">0</text>
<line x1="134" y1="202" x2="134" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="202" y="202" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="210.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">1</text>
<line x1="210" y1="202" x2="210" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="278" y="202" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="286.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">2</text>
<line x1="286" y1="202" x2="286" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="354" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="362.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">3</text>
<line x1="362" y1="202" x2="362" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="430" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">4</text>
<line x1="438" y1="202" x2="438" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="506" y="202" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="514.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">5</text>
<line x1="514" y1="202" x2="514" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="582" y="202" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="590.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">6</text>
<line x1="590" y1="202" x2="590" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="658" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="666.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">7</text>
<line x1="666" y1="202" x2="666" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="16" y="242" width="688" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="24" y="264" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">sunucu C</text>
<rect x="144" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">0</text>
<line x1="152" y1="248" x2="152" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="220" y="248" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="228.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">1</text>
<line x1="228" y1="248" x2="228" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="296" y="248" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="304.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">2</text>
<line x1="304" y1="248" x2="304" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="372" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="380.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">3</text>
<line x1="380" y1="248" x2="380" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="448" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="456.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">4</text>
<line x1="456" y1="248" x2="456" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="524" y="248" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="532.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">5</text>
<line x1="532" y1="248" x2="532" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="600" y="248" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="608.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">6</text>
<line x1="608" y1="248" x2="608" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="676" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="684.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">7</text>
<line x1="684" y1="248" x2="684" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<path d="M116 150 V112" style="fill:none;stroke:var(--c-accent);stroke-width:2.6"/>
<path d="M134 196 V112" style="fill:none;stroke:var(--c-accent);stroke-width:2.6"/>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">A·GPU0 → NIC0 → leaf 0 → NIC0 → B·GPU0: aynı rail, tek leaf atlaması (aynı SU içinde).</text>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">A·GPU0 → C·GPU1: farklı rail → spine&#x27;a çıkar ya da önce NVLink ile A·GPU1&#x27;e geçip rail 1&#x27;i kullanır (NCCL PXN).</text>
</svg>

Böylece bazı aynı-rail iletişimleri kendi scalable unit'inde tek bir leaf switch üzerinden gerçekleşebilir. Ancak farklı SU'lar arasında veya farklı rail'ler arasında trafik spine katmanına çıkabilir. **Bütün rail-optimize ağ trafiğinin tek hop olduğu** iddiası doğru değildir. NVIDIA DGX H100 referansı, SU içindeki rail trafiği ile SU'lar arası yolları açıkça ayırır. [K4]

### 9.4. Süper bilgisayarda kaç ayrı ağ bulunur?

Büyük bir AI sistemi yalnızca tek bir InfiniBand ağına sahip olmak zorunda değildir. NVIDIA'nın DGX H100 SuperPOD referansında aşağıdaki işlevler ayrıştırılır:

<svg viewBox="0 0 720 304" role="img" aria-label="Tek bir DGX H100 dört ayrı ağa bağlanır. Compute ağı, 400 Gb/s InfiniBand&#x27;li sekiz ConnectX-7, GPU kolektiflerini ve dağıtık eğitimi taşır; AllReduce ve AllToAll depolamayla boğuşmasın diye ayrıdır. Depolama ağı, iki çift portlu ConnectX-7, veri kümelerini ve checkpoint&#x27;leri taşır; yüksek hacimli burst I/O ayrı planlanır. 100 GbE in-band yönetim ağı işletim sistemi, servis ve scheduler trafiğini taşır. 1 GbE out-of-band yönetim ağı BMC&#x27;ye ulaşır; açma-kapama, konsol ve arıza müdahalesi içindir ve işletim sistemi bozulsa bile çalışır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="fn-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Tek DGX, dört ayrı ağ (NVIDIA DGX H100 SuperPOD referansı)</text>
<rect x="16" y="44" width="120" height="228" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="76" y="150" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">DGX H100</text>
<text x="76" y="168" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPU</text>
<line x1="136" y1="69" x2="168" y2="69" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="44" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="182" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Compute ağı</text>
<text x="182" y="82" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × CX-7 · 400 Gb/s IB</text>
<text x="380" y="64" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU kolektifleri, dağıtık eğitim</text>
<text x="380" y="82" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">AllReduce/AllToAll depolamayla boğuşmasın</text>
<line x1="136" y1="127" x2="168" y2="127" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="102" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="182" y="122" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Depolama ağı</text>
<text x="182" y="140" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">2 × çift portlu CX-7</text>
<text x="380" y="122" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">veri kümeleri, checkpoint</text>
<text x="380" y="140" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">yüksek hacimli burst I/O ayrı planlansın</text>
<line x1="136" y1="185" x2="168" y2="185" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="160" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="182" y="180" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">In-band yönetim</text>
<text x="182" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100 GbE</text>
<text x="380" y="180" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">işletim sistemi, servisler, scheduler</text>
<text x="380" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">günlük operasyon trafiği ayrı yönetilsin</text>
<line x1="136" y1="243" x2="168" y2="243" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="218" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="182" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Out-of-band (BMC)</text>
<text x="182" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1 GbE</text>
<text x="380" y="238" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">açma-kapama, konsol, arıza müdahalesi</text>
<text x="380" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">işletim sistemi bozulsa bile erişim kalsın</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BMC = anakart yönetim denetleyicisi; UFM InfiniBand ağını yönetir ve izler; Slurm gibi bir scheduler işleri yerleştirir.</text>
</svg>

Referans tasarımda bu ağların her birinin kendi switch'leri vardır: compute ağında 48 Quantum-2 QM9700 (32 leaf + 16 spine), depolama ağında 16 QM9700, in-band yönetimde 8 SN4600C ve out-of-band yönetimde 8 SN2201 [K4]. Tek bir QM9700, 64 adet 400 Gb/s port sunar [K54].

**BMC (Baseboard Management Controller)**, sunucunun işletim sisteminden bağımsız donanım yönetim bileşenidir. **UFM (Unified Fabric Manager)**, InfiniBand fabric yönetimi ve gözlemlenmesi için kullanılan NVIDIA çözümüdür. **Scheduler**, Slurm gibi, işleri uygun sunucu/GPU kaynaklarına yerleştirir.

Bunların yanında imaj yönetimi, konteyner, sürücüler, NCCL, monitoring, güç ve soğutma da sistemin parçalarıdır. Bir node'un 10 kW mertebesinde tüketimi olan referans tasarımda elektrik ve termal dağıtım yalnızca tesisat ayrıntısı değildir: frekans düşmesi ve kararsızlık, eğitim performansını etkileyebilir. [K4]

### 9.5. SuperPOD sayısal örneği

Belirli NVIDIA DGX H100 SuperPOD referansı için:

- 1 SU: 32 sunuculuk kapasiteyle tasarlanan tekrar eden blok.
- 4 SU: fiziksel tasarımda 128 sunucu konumu.
- Tam sistem: UFM yerleşimi nedeniyle **127 DGX, 1.016 H100**.
- Compute fabric: **32 leaf + 16 spine = 48 InfiniBand switch**.
- Depolama ağı: compute fabric'den ayrı düzenlenir.
- **Leaf başına port hesabı**: 32 port aşağı (SU'daki her DGX'in NIC $i$'si) + 32 port yukarı (16 spine'ın her birine 2 kablo) = 64 port. Spine katmanı $16\times64\times50\ \mathrm{GB/s}=51{,}2\ \mathrm{TB/s}$ taşır; bu, bütün GPU'ların aynı anda basabileceği $1016\times50=50{,}8\ \mathrm{TB/s}$'den fazladır, yani tasarım engellemesizdir.
- **İki katmanın ötesi**: leaf–spine 8 SU'ya kadar (64 leaf × 32 = 2.048 GPU) yeter; 16 SU'dan itibaren üçüncü bir **core** katmanı eklenir. 2.048 DGX'lik (16.384 GPU) bir kurulum 512 leaf, 512 spine ve 256 core switch kullanır ve en uzak yol 3'ten 5 switch'e uzar [K55].

Bu sayılar evrensel 'AI kümesi standardı' değildir; referans mimarisinin bileşenleridir. [K4]

### 9.6. Ethernet mi InfiniBand mi?

Burada sık sorulan bir soru var: “Büyük model eğitimi yapacaksak illa InfiniBand mi almalıyız?” Ağ teknolojisini seçmek, yalnızca marka veya kablo seçmek değildir. İhtiyaç duyulan collective performansını, yönetilebilirliği, hata davranışını ve uygun yazılım desteğini birlikte ölçmeliyiz.

**InfiniBand**, yüksek performanslı hesaplama için geliştirilmiş ağ teknolojisidir. **RoCE (RDMA over Converged Ethernet)** ise RDMA iletişimini Ethernet üzerinde sağlar.

'Büyük LLM yalnızca InfiniBand ile eğitilebilir' düşüncesi de yanlıştır. Meta'nın Llama 3 405B eğitim altyapısı RoCE tabanlı tasarım kullanmıştır. Önemli olan fiziksel bağlantı adı kadar RDMA davranışı, congestion control, collectives, topoloji, hata yönetimi ve ölçülmüş uçtan uca throughput'tur. [K5]

---


### 9.7. Dördüncü fiziksel duvar: Elektrik, soğutma ve tesis kapasitesi

Önceki bölümlerde veri taşıma yollarını hesapladık. Peki aynı anda 1.024 H100'ü çalıştıracak **elektrik ve soğutma kapasitesi** veri merkezinde yoksa ne olur? GPU'ların teknik özellik tablosunda gördüğümüz TFLOP/s değeri, hiç kullanamayacağımız bir potansiyelden ibaret kalır.

Bir DGX H100/H200 sunucusunun dokümante edilen **azami sistem gücü 10,2 kW** düzeyindedir. 1.024 GPU'yu sekizer GPU'lu 128 DGX sunucusuna yerleştirdiğimiz **öğretici referans senaryoda**:

$$
P_{\mathrm{DGX,max}}=128\times10.2\ \mathrm{kW}
=1305.6\ \mathrm{kW}\approx1.31\ \mathrm{MW}
$$

Bu, yalnızca 128 sunucunun **isim plakası (nameplate) azami güçlerinin toplamıdır**; gerçek sürekli tüketim ölçümü değildir. Switch, NIC, CPU, depolama, güç dönüşüm ve soğutma tüketimi bu sayıya dahil değildir. Her sunucunun aynı anda maksimumda çalışacağı da varsayılamaz. [K26]

**PUE (Power Usage Effectiveness)** veri merkezinin toplam enerji tüketiminin IT cihazlarının tüketimine oranıdır:

$$
\mathrm{PUE}=\frac{E_{\mathrm{facility}}}{E_{\mathrm{IT}}}
$$

Örneğin IT yükü 1 MW ve PUE 1,3 ise tesis genelinde aynı süre için yaklaşık 1,3 MW eşdeğer güç tüketimi söz konusu olur. Bu, örnek bir PUE değeridir; belirli bir tesis ölçümü değildir.

Bir başka terim **rack density**: Tek bir rack içinde kaç kW ısı/güç yoğunluğu oluştuğunu anlatır. GPU sistemleri yoğunlaştıkça hava akışı yeterli olmayabilir ve sıvı soğutma veya farklı güç dağıtım tasarımı gerekebilir. Uptime Institute, yoğun AI altyapılarının güç/soğutma stratejisini geleneksel kurumsal IT'den farklılaştırdığını vurguluyor. [K27]

**Power capping**, sistemin güç sınırını yazılım/firmware üzerinden belirlemektir. Bu yöntem tesis kapasitesini korumak veya sıcaklığı yönetmek için yararlı olabilir; ancak aktif kısıtlama performansı ve MFU ölçümlerinin yorumunu etkileyebilir.

**Tasarım kontrol listesi:** GPU/sunucu gücü, PDU/UPS/yedeklilik, arıza senaryosu, rack başına güç, soğutma yöntemi, NIC/switch güçleri, kablo/optik modül bütçesi ve genişleme için fiziksel boşluk birlikte onaylanmalıdır. Elektrik mühendisliği kararı, GPU satın alma kararından sonra verilmemelidir.

---


## 10. Collective communication: GPU'lar neden birbirini bekler?

Dört kişinin ortak bir rapor yazdığını düşünün. Herkes kendi bölümünü hazırlamakta çok hızlı. Ama raporun sonunda toplam rakamın herkes tarafından aynı bilinmesi gerekiyor. O rakamı birleştirmeden sonraki aşamaya geçilemiyor.

Dağıtık eğitimde buna benzeyen çok sayıda zorunlu buluşma noktası vardır. GPU'lar tek başlarına hızlı hesap yapsalar bile, bazı sonuçları ortaklaştırmak zorundadır. **Collective communication**, birçok katılımcının birlikte yürüttüğü bu veri iletişim işlemlerinin genel adıdır.

Yapay zekâ eğitiminde GPU'ların kendi sonuçlarını birleştirmesi gerekir. Bunlar **collective communication** işlemleridir.

### 10.1. Dört temel collective işlemi

Elimizde dört GPU ve her birinde bir sayı olduğunu düşünün:

<svg viewBox="0 0 720 428" role="img" aria-label="Dört GPU&#x27;da dört panel. AllReduce (toplam): GPU&#x27;lar 10, 20, 30 ve 40 ile başlar, hepsi 100 ile biter. AllGather: her GPU bir değerle başlar, hepsi dört parçanın tamamıyla biter. ReduceScatter (toplam): her GPU dört dilimle başlar; GPU i, i. dilimin tüm GPU&#x27;lardaki toplamıyla biter. AllToAll: her GPU dört parçayla başlar; GPU i&#x27;nin j. parçası GPU j&#x27;ye gider, matrisin transpozu gibi. AllReduce, ReduceScatter ve ardından AllGather&#x27;a eşittir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="cl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Dört GPU&#x27;da dört kolektif (önce → sonra)</text>
<rect x="16" y="40" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllReduce(sum)</text>
<text x="110" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="178" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="246" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="314" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="28" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="28" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="208.0" y="140" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="86" y="86" width="48" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="110.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">10</text>
<rect x="86" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="110.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<rect x="154" y="86" width="48" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="178.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">20</text>
<rect x="154" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="178.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<rect x="222" y="86" width="48" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="246.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">30</text>
<rect x="222" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="246.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<rect x="290" y="86" width="48" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="314.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">40</text>
<rect x="290" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="314.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<text x="28" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">herkes toplamı bilir</text>
<rect x="364" y="40" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllGather</text>
<text x="458" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="526" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="594" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="662" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="376" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="376" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="556.0" y="140" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="446" y="86" width="24" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="458.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">10</text>
<rect x="432" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="445" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="458" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="471" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="514" y="86" width="24" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="526.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">20</text>
<rect x="500" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="513" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="526" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="539" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="582" y="86" width="24" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">30</text>
<rect x="568" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="581" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="594" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="607" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="650" y="86" width="24" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="662.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">40</text>
<rect x="636" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="649" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="662" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="675" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="376" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">her parça herkeste olur</text>
<rect x="16" y="224" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="244" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">ReduceScatter(sum)</text>
<text x="110" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="178" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="246" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="314" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="28" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="28" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="208.0" y="324" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="84" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="90.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="97" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="103.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="110" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="116.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="123" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="129.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="92" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="110.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ0</text>
<rect x="152" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="158.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="165" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="171.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="178" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="184.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="191" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="197.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="160" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="178.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ1</text>
<rect x="220" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="226.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="233" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="239.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="246" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="252.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="259" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="265.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="228" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="246.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ2</text>
<rect x="288" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="294.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="301" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="307.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="314" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="320.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="327" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="333.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="296" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="314.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ3</text>
<text x="28" y="384" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU i, i. dilimin toplamını tutar</text>
<rect x="364" y="224" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="244" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllToAll</text>
<text x="458" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="526" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="594" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="662" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="376" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="376" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="556.0" y="324" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="432" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="445" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="451.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="458" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="464.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="471" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="477.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="432" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="445" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="451.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="458" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="464.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="471" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="477.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="500" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="506.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="513" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="519.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="526" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="532.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="539" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="545.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="500" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="506.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="513" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="519.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="526" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="532.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="539" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="545.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="568" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="574.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="581" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="587.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="594" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="600.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="607" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="613.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="568" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="574.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="581" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="587.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="594" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="600.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="607" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="613.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="636" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="642.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="649" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="655.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="662" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="668.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="675" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="681.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="636" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="642.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="649" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="655.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="662" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="668.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="675" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="681.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<text x="376" y="384" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU i&#x27;nin j. parçası GPU j&#x27;ye gider</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Renk = kaynak GPU · rakam = dilim no · Σi = i. dilimin tüm GPU&#x27;lardaki toplamı. AllReduce = ReduceScatter + AllGather.</text>
</svg>

| İşlem | Çıktı | Sezgisi |
|---|---|---|
| **AllReduce(sum)** | Her GPU: 100 | Herkes katkısını verir, herkes toplamı bilir |
| **AllGather** | Her GPU: [10,20,30,40] | Herkes parçalarını paylaşır; tüm parçalar herkeste olur |
| **ReduceScatter(sum)** | Toplamın farklı dilimleri GPU'lara dağılır | Önce birleştir, sonucu bölüş |
| **AllToAll** | Her GPU herkese farklı parça yollar | Departmanlar karşılıklı ayrı dosyalar yollar |

### 10.2. Ring AllReduce: Adım adım

Bütün GPU'ların aynı toplamı öğrenmesini istiyoruz. En basit çözüm, tüm veriyi tek GPU'da toplamak olabilir. Ama binlerce GPU'da tek merkezi nokta anında darboğaz haline gelir. Ring AllReduce bu işi halka boyunca dağıtır; katılımcılar hem veri gönderir hem birleştirmeye katkı sağlar.

Buradaki güzel fikir, mesajları parçalara ayırarak bütün GPU'ların iletişime katılmasıdır. Bedeli ise her parçanın birden fazla adımda hareket etmesidir.

Ring algoritması veriyi halka boyunca dolaştırır. İlk fazda *reduce-scatter*, ikinci fazda *all-gather* uygulanır.

<svg viewBox="0 0 720 342" role="img" aria-label="Halka biçiminde bağlı dört GPU ile ring AllReduce: GPU 0 → 1 → 2 → 3 → 0. Veri S/N boyutunda dört parçaya bölünür. İlk üç adımda (reduce-scatter) her GPU komşusuna bir parça gönderir, komşu kendi payını ekler; kısmi toplamlar halkada dolaşır ve sonunda her GPU bir bitmiş toplamı tutar. Sonraki üç adımda (all-gather) bitmiş toplamlar dolaştırılır ve her GPU hepsine sahip olur. Her GPU 2(N−1)/N · S byte gönderir; N = 8, S = 8 GB için GPU başına 14 GB eder. Süre kabaca 2(N−1) gecikme adımı artı bu hacmin bant genişliğine bölümüdür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="rg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">N = 4&#x27;te ring AllReduce: 2(N−1) = 6 adım, adım başına S/N&#x27;lik bir parça</text>
<path d="M136 102 Q 190 90 178 144" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M178 176 Q 190 230 136 218" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M104 218 Q 50 230 62 176" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M62 144 Q 50 90 104 102" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="120" cy="86" r="22" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.4"/>
<text x="120" y="90" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU0</text>
<circle cx="194" cy="160" r="22" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.4"/>
<text x="194" y="164" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU1</text>
<circle cx="120" cy="234" r="22" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.4"/>
<text x="120" y="238" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU2</text>
<circle cx="46" cy="160" r="22" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="46" y="164" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU3</text>
<text x="120" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">halka</text>
<text x="297.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 1</text>
<text x="367.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 2</text>
<text x="437.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 3</text>
<text x="507.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 4</text>
<text x="577.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 5</text>
<text x="647.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 6</text>
<text x="254" y="93" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU0</text>
<rect x="266" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +0</text>
<rect x="336" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +3</text>
<rect x="406" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +2</text>
<rect x="476" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ1</text>
<rect x="546" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ0</text>
<rect x="616" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ3</text>
<text x="254" y="133" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU1</text>
<rect x="266" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +1</text>
<rect x="336" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +0</text>
<rect x="406" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +3</text>
<rect x="476" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ2</text>
<rect x="546" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ1</text>
<rect x="616" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ0</text>
<text x="254" y="173" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU2</text>
<rect x="266" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +2</text>
<rect x="336" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +1</text>
<rect x="406" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +0</text>
<rect x="476" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ3</text>
<rect x="546" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ2</text>
<rect x="616" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ1</text>
<text x="254" y="213" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU3</text>
<rect x="266" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +3</text>
<rect x="336" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +2</text>
<rect x="406" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +1</text>
<rect x="476" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ0</text>
<rect x="546" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ3</text>
<rect x="616" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ2</text>
<line x1="266" y1="238" x2="468" y2="238" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="476" y1="238" x2="678" y2="238" style="stroke:var(--c-accent-2);stroke-width:2"/>
<text x="367.0" y="254" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:600">N−1 adım: kısmi toplamlar dolaşır</text>
<text x="577.0" y="254" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">N−1 adım: bitmiş toplamlar yayılır</text>
<rect x="16" y="278" width="688" height="52" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="298" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU 2(N−1)/N · S byte gönderir. N = 8, S = 8 GB → 2 · 7/8 · 8 = GPU başına 14 GB (toplam trafik değil).</text>
<text x="28" y="318" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Süre ≈ 2(N−1)·L + 2(N−1)/N · S/W: gecikme terimi N ile büyür, bant genişliği terimi neredeyse sabit kalır.</text>
</svg>

Bir GPU'nun başlangıçtaki collective tensör boyutu $S$ byte, GPU sayısı $N$ ise ideal ring algoritmasında **GPU başına iletilen yaklaşık byte**:

$$
V_{\mathrm{ring}}=2\frac{N-1}{N}S
$$

Örneğin N=8 ve S=8 GB:

$$
V_{\mathrm{ring}}=2\cdot\frac78\cdot8=14\ \mathrm{GB/GPU}
$$

Bütün GPU'ların toplam gönderim baytı ayrıca hesaplanır; yukarıdaki değer toplam küme trafiği değildir. Süreye dönüştürmek için link kapasitesi, kullanılabilir algoritmik bant genişliği, transfer boyutu ve **latency** eklenmelidir.

İletişim süresini kaba bir $\alpha$–$\beta$ modeliyle düşünebiliriz:

$$
T_{\mathrm{ring}}\approx2(N-1)L+2\frac{N-1}{N}\frac{S}{W}
$$

Buradaki $L$ her iletişim adımının sabit gecikmesini, $W$ etkin haberleşme bant genişliğini temsil eder. Bu, ideal varsayımlı alt/üst düzey modeldir; gerçek NCCL algoritması ve ağ topolojisi daha karmaşıktır.

Aynı modelde AllGather tek fazdır: $N-1$ adımda her GPU $S/N$'lik bir parça gönderir, yani $T_{\mathrm{AllGather}}\approx(N-1)L+\frac{N-1}{N}\frac{S}{W}$. AllReduce'un yaklaşık iki kat uzun sürmesinin nedeni de budur: önce reduce-scatter, sonra all-gather. Küçük mesajlarda $(N-1)L$ gecikme terimi, büyüklerde $S/W$ bant genişliği terimi baskındır; ölçülen NCCL bant genişliğinin ancak mesajlar yeterince büyüdüğünde doymasının nedeni budur. [K11]

### 10.3. SHARP: Switch yalnızca yönlendirme yapmayabilir

Şimdi daha şaşırtıcı bir fikir düşünelim: GPU'lar birbirlerine sayılar gönderecek ve herkes bu sayıları toplayacaksa, toplama işleminin bir bölümünü ağ switch'i yapsa nasıl olur?

Evet, bazı modern ağ teknolojileri bunu gerçekleştirebilir. Böylece ağ, yalnızca paketin nereye gideceğini bilmekle kalmaz; desteklenen reduction işlemlerine de katılır. Buna **in-network computing** denir.

**SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)**, destekleyen NVIDIA InfiniBand ağlarında bazı reduction operasyonlarının bir bölümünü switch donanımında gerçekleştirebilir.

<svg viewBox="0 0 720 334" role="img" aria-label="İki panel. Geleneksel: A, B ve C GPU&#x27;ları veriyi yalnızca paket ileten bir switch üzerinden değiş tokuş eder ve toplamalar GPU&#x27;larda yapılır; veri ring adımları boyunca switch&#x27;ten defalarca geçer. SHARP: switch desteklenen azaltma işlemlerini donanımda yapar; her GPU girdisini bir kez yukarı yollar ve sonucu bir kez aşağı alır. Kazanç mesaj boyutuna, azaltma türüne ve trafiğe bağlıdır; sabit %50 tasarruf değildir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sh-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="sh-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="sh-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Toplama nerede yapılıyor?</text>
<rect x="16" y="40" width="336" height="262" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Geleneksel: GPU&#x27;larda reduce</text>
<rect x="134" y="84" width="100" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="184" y="102" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">switch</text>
<text x="184" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yalnızca iletir</text>
<rect x="46" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU A</text>
<text x="82" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">toplar</text>
<path d="M72 196 L164 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M174 126 L92 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="186" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU B</text>
<text x="186" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">toplar</text>
<path d="M176 196 L184 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M194 126 L196 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="254" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="290" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU C</text>
<text x="290" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">toplar</text>
<path d="M280 196 L204 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M214 126 L300 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="28" y="262" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">veri switch&#x27;ten defalarca geçer (ring adımları)</text>
<text x="28" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">aritmetik GPU çekirdeklerinde yapılır</text>
<rect x="368" y="40" width="336" height="262" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="380" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">SHARP: switch içinde reduce</text>
<rect x="486" y="84" width="100" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="536" y="102" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">switch</text>
<text x="536" y="117" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">donanımda Σ</text>
<rect x="398" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="434" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU A</text>
<text x="434" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir kez yollar</text>
<path d="M424 196 L516 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M526 126 L444 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="502" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="538" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU B</text>
<text x="538" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir kez yollar</text>
<path d="M528 196 L536 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M546 126 L548 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="606" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="642" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU C</text>
<text x="642" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir kez yollar</text>
<path d="M632 196 L556 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M566 126 L652 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="380" y="262" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">bir kez yukarı: girdiler</text>
<text x="551.944" y="262" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">· bir kez aşağı: sonuç</text>
<text x="380" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">desteklenen azaltma ağda yapılır</text>
<text x="16" y="322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Kazanç mesaj boyutuna, azaltma türüne ve trafiğe bağlıdır; sabit %50 tasarruf değildir.</text>
</svg>

İletilen veri hacmi ve bazı collective süreleri azalabilir. Fakat 'SHARP her iş yükünde trafiği tam %50 düşürür' veya 'tüm eğitim süresini ikiye böler' demek yanlıştır. SHARP'ın faydası reduction örüntüsüne, algoritmaya, mesaj boyutuna ve trafik koşullarına bağlıdır. [K7]

Etkinin büyüklüğünü bir örnekle görelim. Llama 3 70B'yi her DGX içinde TP = 8, 127 DGX boyunca veri paralel eğittiğimizi düşünelim. Her GPU 70 milyar parametrenin sekizde birinin BF16 gradyanını tutar: $70\times10^9\times2/8=17{,}5$ GB. Aynı numaralı GPU'lar (tek rail) arasında 127 katılımcılı ring AllReduce, GPU başına kabloya $2\cdot\frac{126}{127}\cdot17{,}5\approx34{,}7$ GB basar; 50 GB/s'lik NIC'te bu yaklaşık **0,69 sn**'dir. SHARP'ta her GPU gradyanını bir kez yukarı yollar ve sonucu bir kez alır: yaklaşık 17,5 GB, ideal durumda **0,35 sn**. Fark büyük ve bant genişliği sınırlı mesajlarda 2 kata yaklaşır; küçük mesajlarda ve gerçek trafik altında daralır.

### 10.4. Compute–communication overlap

İletişim tamamlanana kadar hiçbir şey yapmadan beklemek zorunda olduğumuz durumlar vardır. Ama bazen mevcut katmanın bağımsız bir parçasını hesaplamaya devam edebiliriz. Mesela ekibin bir kısmı raporu gönderirken diğerleri yeni bölümü hazırlayabiliyorsa toplam bekleme azalır.

Bu fikir, dağıtık eğitimin pek çok farklı optimizasyonunda karşımıza çıkacak: **iletişim ile hesaplamayı birbirinin üzerine bindirmek**.

GPU'nun **compute stream**'inde matris hesaplanırken, bağımsız bir iletişim parçası asenkron aktarılabiliyorsa ağ gecikmesinin bir kısmı gizlenir.

<svg viewBox="0 0 720 274" role="img" aria-label="Hesap ve ağ stream&#x27;leri olan iki zaman çizelgesi. Kötü zamanlama: hesap 1–2 çalışır, ardından hesap boşta beklerken ağ aktarımı yapılır, sonra hesap 3–4 gelir. Örtüşmeli: hesap 1–4 kesintisiz çalışırken ağ aktarımının bağımsız parçası altında yürür; adım, aktarım süresi kadar erken biter. Bu yalnızca aktarım yanındaki hesaba bağımlı değilse işe yarar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="os-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="os-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">İki CUDA stream&#x27;inde hesap ve iletişim</text>
<text x="16" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">kötü zamanlama</text>
<text x="140" y="82" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">hesap</text>
<text x="140" y="116" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">ağ</text>
<line x1="150" y1="96" x2="704" y2="96" style="stroke:var(--c-border);stroke-width:0.6;stroke-dasharray:4 3"/>
<rect x="150" y="66" width="140" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="220.0" y="82.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap 1–2</text>
<rect x="290" y="100" width="140" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ</text>
<rect x="290" y="66" width="140" height="26" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="360.0" y="82.96" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">boşta</text>
<rect x="430" y="66" width="140" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="500.0" y="82.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap 3–4</text>
<line x1="570" y1="56" x2="570" y2="132" style="stroke:var(--c-text);stroke-width:1.4"/>
<text x="576" y="138" text-anchor="start" style="fill:var(--c-text);font-size:11px">bitiş</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">örtüşmeli</text>
<text x="140" y="186" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">hesap</text>
<text x="140" y="220" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">ağ</text>
<line x1="150" y1="200" x2="704" y2="200" style="stroke:var(--c-border);stroke-width:0.6;stroke-dasharray:4 3"/>
<rect x="150" y="170" width="280" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="290.0" y="186.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap 1–4</text>
<rect x="220" y="204" width="170" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="305.0" y="220.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ (bağımsız parça)</text>
<line x1="430" y1="160" x2="430" y2="236" style="stroke:var(--c-text);stroke-width:1.4"/>
<text x="436" y="242" text-anchor="start" style="fill:var(--c-text);font-size:11px">bitiş</text>
<path d="M434 183 H566" marker-start="url(#os-b)" marker-end="url(#os-b)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5"/>
<text x="500" y="176" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:600">kazanılan süre</text>
<text x="16" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca aktarım yanındaki hesaptan bağımsızsa işe yarar; gerçek bir bağımlılık kritik yolda kalır.</text>
</svg>

Bütün iletişim her zaman saklanamaz. Bir katman, başka bir GPU'nun sonucuna bağımlıysa o sonuç hazır olmadan ilerlemek doğru değildir.

---



---

### Ağ topolojisini gerçekten anlamak için "veri nereye gidiyor?" sorusu

GPU'lar arasında haberleşme dendiğinde akla ilk olarak kablonun bant genişliği gelir. Oysa asıl belirleyici olan, verinin fiziksel topolojide izlediği güzergâhtır: Mesaj aynı sunucudaki NVLink alanında mı kalıyor, PCIe switch üzerinden komşu sokete mi geçiyor, yoksa InfiniBand leaf-spine kumaşına çıkarak optik switch atlamaları mı yapıyor?

Bir mesajın yolculuğunda iki farklı fiziksel rejim çalışır:
- **Gecikme (latency) rejimi:** Küçük mesajlarda (örneğin TP collective veya senkronizasyon bariyerleri) ağın switch atlama (hop) sayısı ve kuyruk gecikmesi belirleyicidir; hat bant genişliği ne kadar yüksek olursa olsun donanımsal gecikme tabanı aşılamaz.
- **Bant genişliği (bandwidth) rejimi:** Büyük tensör aktarımlarında (örneğin DP gradient AllReduce veya FSDP AllGather) kesit bant genişliği (bisection bandwidth) ve paket kaybı baskındır.

Ağ tasarımının ilk adımı, iş yükünün bu iki rejimden hangisinde çalıştığını teşhis etmektir.


### NVLink alanı değişince TP tasarımı neden yeniden düşünülür?

Birçok H100 DGX/HGX sisteminde sekiz GPU'nun yüksek hızlı NVLink/NVSwitch ile birbirine bağlı olması, TP için doğal bir yerleşim sunar. Ancak **TP=8 bir fizik yasası değildir**. Tek sunucuyu aşan TP mümkündür; karşılığında farklı bir iletişim yolu ve maliyet alınır.

Yeni nesil rack-scale sistemler bu tartışmayı bir adım daha ileri taşır. Örneğin GB300 NVL72'de 72 GPU'luk NVLink alanı bulunur. Bu, “sekiz GPU'dan fazlasıyla TP imkânsız” cümlesinin niçin doğru olmadığını somutlaştırır. Ancak 72 GPU'luk *scale-up alanı*, bütün işlerin TP=72 olacağı anlamına da gelmez: Modelin katman boyutları, bölünebilirlik, bellek gereksinimi ve iletişim maliyeti hâlâ belirleyicidir. [K51]

### İletişim bekleyişi gizlenebilir mi?

Overlap kavramını bir restoran servis hattında düşünelim. Biri ekmekleri getirirken aşçı beklemek yerine diğer siparişin sosunu hazırlıyorsa teslim gecikmesinin bir kısmı üretimle örtüşür. Fakat aşçının sosu yapması için önce ekmeğin gelmesi gerekiyorsa örtüşme mümkün değildir.

Dağıtık eğitimde benzer bir **bağımlılık grafiği** vardır. Bir katmanın hesaplaması, bir sonraki katmanın parametrelerinin AllGather işlemiyle getirilmesinden bağımsızsa ikisi örtüşebilir. Bağımlılık varsa ağ bekleyişi görünür kalır. Ayrıca overlap için iletişim ve hesaplama aynı GPU/NIC/bellek kaynaklarını paylaşıyorsa ikisi birbirinin performansını düşürebilir.

Bu yüzden “iletişim %95 gizlendi” gibi ölçümler ancak **hangi mimaride, hangi boyuttaki mesajlarda, hangi hesaplamayla kıyaslandığı** bilindiğinde anlamlıdır.


## 11. Ağ performansını anlamak: Topoloji kadar trafik biçimi de neden önemlidir?

GPU'ların birbirini beklemesinin tek nedeni kablo hızının düşük olması değildir. Yanlış GPU–NIC eşleşmesi, tek yöne yığılan trafik, fazla ağ atlaması, ağda paket kaybı veya yazılımın yanlış arayüzü seçmesi de benzer semptomlar üretir.

### 11.1. Önce 'hangi trafik?' sorusu

| Trafik | Tipik özellik | Baskın risk |
|---|---|---|
| TP collective | Sık ve gecikmeye duyarlı | Katman içi bekleme |
| DP AllReduce/ReduceScatter | Büyük tensör aktarımları | Ağ bant genişliği ve congestion |
| MoE AllToAll | Çok hedefli, düzensiz yönlendirme | Hotspot ve dengesiz yük |
| PP activation transfer | Komşu stage'ler arası | Pipeline bubble |
| Checkpoint | Depolamaya yüksek hacimli burst | Storage fabric ve I/O throttle |
| Disaggregated KV transfer | Prefill→decode | KV taşıma gecikmesinin kazancı silmesi |

Bu tabloda gördüğümüz kritik şey: **Bir ağı test ederken tek bir iperf sonucu bütün iş yüklerini temsil etmez.** NCCL collective mesaj boyutu ve iletişim topolojisi farklıysa sonuç da farklı olur.

### 11.2. Basit bir ağ aktarım modeli

Bir mesaj için ilk yaklaşım:

$$
T_{\mathrm{transfer}}\approx T_{\mathrm{startup}} + \frac{B}{W_{\mathrm{effective}}}
$$

- $T_{startup}$: İletişimi başlatma, el sıkışma ve sabit gecikme maliyeti.
- $B$: Gönderilen byte.
- $W_{effective}$: Uygulamada elde edilen bant genişliği.

Sürekli 400 Gb/s olarak pazarlanan bir bağlantının teorik byte hızı **50 GB/s**'dir; uygulamada protokol giderleri, PCIe/NIC sınırlamaları ve paylaşılan trafik nedeniyle daha azını görebilirsiniz. Bir kolektif işlemin toplam zamanı yalnızca tek bir linkin teorik hızından hesaplanmaz: halka/ağaç algoritması, aktarılan veri sayısı ve ağın çapı da önemlidir.

### 11.3. Bisection bandwidth ve oversubscription

**Bisection bandwidth**, bir ağı kabaca iki eşit yarıya ayırdığınızda bu iki yarı arasında taşınabilecek toplam veriyi tanımlar. Bir veri merkezinde her host'un 400 Gb/s NIC'i olması, tüm host çiftlerinin aynı anda 400 Gb/s çalışabileceği anlamına gelmez. Leaf–spine uplink kapasitesi yetersizse trafik **oversubscribed** olur.

Örnek: Bir leaf switch'e bağlı sekiz host × 400 Gb/s = 3,2 Tb/s host yönlü kapasite var; uplink yalnızca 1,6 Tb/s ise bu leaf için kaba oran **2:1** olur. Trafiğin tümü rack dışına çıkıyorsa uplink darboğazına yaklaşırız; trafik çoğunlukla leaf içinde kalıyorsa aynı hesap farklı sonuç verir.

### 11.4. RoCE mi InfiniBand mi? 'Hangi daha hızlı?' sorusuna daha iyi cevap

Mimar önce şu maddelere bakmalıdır: seçilen switch/NIC ailesinin RDMA desteği; iş yükünün collective paterni; operasyon ekibinin Ethernet congestion yönetimi yetkinliği; uçtan uca telemetry; hata ayıklama ve tahsis modeli. RoCE, Ethernet üzerinde RDMA taşır ama performans ve kararlılık **ağ ayarlarının gerçekten doğrulanmasına** bağlıdır. PFC/ECN gibi congestion mekanizmalarını körlemesine açmak çözüm değildir; yanlış yapılandırma head-of-line blocking veya başka trafik sorunları yaratabilir.

İnfiniBand ve Ethernet/RoCE için avantaj–dezavantajı tek bir fiyat etiketine indirgemek de doğru değildir: switch, NIC, kablolama, yönetim yazılımı, operasyon personeli ve ağın mevcut ekosistemi birlikte kıyaslanmalıdır.

### 11.5. Ölçüm piramidi: Kablodan modele

<svg viewBox="0 0 720 306" role="img" aria-label="Aşağıdan yukarı doğrulanan altı katmanlı ölçüm piramidi: 1, fiziksel hatlar; link durumu, optik ve hata sayaçları. 2, host veri yolu; PCIe Gen5, NUMA ve GPU↔NIC yakınlığı, nvidia-smi topo -m ve lspci ile. 3, noktadan noktaya bant genişliği ve gecikme için RDMA mikrobenchmark. 4, all_reduce_perf busbw, AllGather ve AllToAll gibi NCCL kolektifleri. 5, model step süresi ve hesap/iletişim örtüşmesi. 6, time-to-quality, MFU ve p99 dahil iş yükü SLO&#x27;su ve maliyet. Her katmanın baz çizgisi ancak alttaki katman temizse anlamlıdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mp2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Ağ ölçüm piramidi: aşağıdan yukarı doğrula</text>
<path d="M150 40 L190 40 L210 76 L130 76 Z" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="214" y1="58" x2="322" y2="58" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="55" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">6 · İş yükü SLO &amp; maliyet</text>
<text x="330" y="71" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time-to-quality, MFU, p99</text>
<path d="M130 80 L210 80 L230 116 L110 116 Z" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="234" y1="98" x2="322" y2="98" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">5 · Model step süresi</text>
<text x="330" y="111" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hesap/iletişim örtüşmesi</text>
<path d="M110 120 L230 120 L250 156 L90 156 Z" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="254" y1="138" x2="322" y2="138" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="135" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">4 · NCCL kolektifleri</text>
<text x="330" y="151" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">all_reduce_perf busbw, AllGather, AllToAll</text>
<path d="M90 160 L250 160 L270 196 L70 196 Z" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<line x1="274" y1="178" x2="322" y2="178" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="175" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">3 · RDMA mikrobenchmark</text>
<text x="330" y="191" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">noktadan noktaya bant genişliği ve gecikme</text>
<path d="M70 200 L270 200 L290 236 L50 236 Z" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<line x1="294" y1="218" x2="322" y2="218" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="215" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">2 · Host veri yolu</text>
<text x="330" y="231" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe Gen5, NUMA, GPU↔NIC: nvidia-smi topo -m, lspci</text>
<path d="M50 240 L290 240 L310 276 L30 276 Z" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.2"/>
<line x1="314" y1="258" x2="322" y2="258" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="255" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">1 · Fiziksel hatlar</text>
<text x="330" y="271" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">link durumu, optik, hata sayaçları</text>
<text x="16" y="294" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Her katmanın baz çizgisi ancak alttaki katman temizse anlamlıdır; mesaj boyutlarını tarayın, yükseltme öncesi ve sonrası tekrarlayın.</text>
</svg>

**Öğretici test örneği:** Aynı sunucu içinde `all_reduce_perf`, sonra iki sunucu, sonra bir rack, sonra rack'ler arası test. Her aşamada mesaj boyutlarını küçükten büyüğe tarayın; `busbw`, uygulama bandwidth'i, p50/p99 süre ve hata sayaçlarını kaydedin. Bu baz çizgiyi yazılım/sürücü yükseltmelerinden önce ve sonra karşılaştırın. NCCL Tests'in kullanım biçimi ve örnekleri kaynak [K39]'da bulunur.

```bash
# Sadece örnek: NCCL Tests ikili dosyası kuruluysa ve node size ait 8 GPU boşsa.
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

# İzinler ve sürücüler uygunsa GPU↔NIC yakınlığını inceleme
nvidia-smi topo -m

# PCI aygıtlarını ve link durumlarını tanıma
lspci | grep -Ei 'nvidia|network|infiniband'
```

Üç komut da kök nedeni kendiliğinden çözmez. Ama sistematik teşhis için ortak bir başlangıç noktası oluşturur. Üretimde yoruma ihtiyaç duymadan kopyalanan agresif NCCL environment değişkenleri yerine, önce fiziksel ve yazılım topolojisi doğrulanmalıdır. [K38][K39]

---


## 12. GPU kümelerinde doğrulama testleri: Donanım görünürlüğü neden yeterli değildir?

Bir AI cluster'ı satın almak, sistemi üretime hazır teslim aldığınız anlamına gelmez. 1.024 GPU'nun `nvidia-smi` çıktısında görünmesi iyi haber; ama o cluster'da 1.024 GPU'nun **birlikte, kararlı, beklenen hızda** çalıştığının kanıtı değildir.

### 12.1. Golden image ve sürüm matrisi

Sürücü, CUDA, NCCL, container runtime, CUDA-aware framework, kernel, NIC firmware, OFED/driver katmanı ve GPU fabric yönetimi birbirine bağlıdır. **Golden image**, doğrulanmış ve sürümlenmiş işletim sistemi/yazılım tabanıdır. Her düğümü bağımsız elle güncellemek yerine imajı ve konfigürasyonu otomasyonla aynı tutmak sürprizleri azaltır.

Sürüm tablosu örneği:

| Bileşen | Kaydedilecek bilgi | Neden |
|---|---|---|
| GPU | Model, SKU, BIOS/firmware | Kapasite ve özellik tutarlılığı |
| Driver / CUDA | Tam sürüm | ABI ve destek matrisi |
| NCCL / framework | Exact version + image digest | Collective davranışı yeniden üretmek |
| NIC | Model, firmware, port hızları | RDMA hata ve performans teşhisi |
| OS / kernel | Sürüm, BIOS NUMA/ACS/IOMMU | P2P veri yolunu etkileyebilir |
| Storage client | Dosya sistemi sürümü, mount policy | Dataset ve checkpoint süreleri |
| Scheduler | Queue / topology / admission politikası | Yanlış placement kaynaklı yavaşlığı ayırmak |

### 12.2. Farklı test seviyeleri neyi görünür kılar?

**Aşama 1 — Node sağlığı:** GPU görünürlüğü, beklenen bellek, ECC/Xid, NVLink durumu, ısı, güç, fan/pompa gibi platform bileşenleri, disk SMART ve NIC link durumları.

**Aşama 2 — GPU compute ve bellek:** Sustained GEMM, HBM bandwidth, GPU↔GPU kopyalama testleri. Kısa benchmark geçip yük altında bozulan donanımı yakalamak için süreli stress ve soak test gerekir. Ancak agresif testlerin servis dışı veya drain edilmiş düğümlerde yapılması gerekir.

**Aşama 3 — Network:** Farklı rack ve NIC ray'lerinde RDMA testleri ve NCCL collective sweep.

**Aşama 4 — Storage:** Eşzamanlı çok istemcili okuma, checkpoint yazma ve storage yeniden başlatma senaryoları. Ortalama değil kuyruk gecikmesi ve metadata baskısını da ölçün.

**Aşama 5 — Uygulama:** Referans eğitim step'i, uzun süren soak run ve eşdeğer konfigürasyonda tutarlı sonuç.

**Aşama 6 — Failure injection:** Bir worker'ın kapanması, bir ağ yolunun düşmesi, depolama yavaşlaması ve node'un sağlıksız işaretlenmesi gibi kontrollü senaryolar. Prod ortamında izinsiz arıza enjeksiyonu yapılmaz.

### 12.3. DCGM ile hangi sınıf problemi yakalarız?

**DCGM**, GPU telemetri ve sağlık teşhisi için NVIDIA'nın veri merkezi araç ailesidir. PCIe replay, ECC double-bit error, page retirement, row remap failure, Xid, termal ve güç sinyalleri gibi koşulları yakalayabilir. Ancak her metriğin her GPU'da bulunması garanti değildir; exporter'ın aktif collector yapılandırması ve donanım destek matrisi kontrol edilmelidir. Bazı profiling sayaçları multiplex edilir; tek noktalık sıfır okumak mutlaka "GPU boş" anlamına gelmez. [K36][K37]

**Operasyonel açıdan önemli ayrım:** GPU'da düzeltilemeyen bellek hatası veya ciddi Xid gözlenirse "MFU artırmak için tuning" yapılmaz. Önce düğümü güvenli biçimde devreden çıkarma, etkisini sınırlama ve üretici sağlık prosedürünü izleme gerekir.

### 12.4. Örnek doğrulama başlıkları

```text
[ ] Donanım envanteri ve sürüm matrisi kaydedildi
[ ] DCGM/firmware sağlık kontrolleri geçti
[ ] GPU↔GPU ve GPU↔NIC topolojisi doğrulandı
[ ] Aynı node / iki node / rack / rackler arası NCCL baseline alındı
[ ] Storage throughput ve checkpoint restore test edildi
[ ] Model kalite/regresyon testleri başarılı
[ ] SLO'ya uygun gerçek trafik benchmark'ı başarılı
[ ] Rollback, secrets ve ağ segmentasyonu denetlendi
[ ] Tekil arıza ve sürüm yükseltme senaryosu denendi
[ ] Monitoring, alarmlar ve on-call runbook'ları hazır
```

Bir mimar için bu listenin asıl değeri şu: Donanım, yazılım, ağ ve model ekipleri aynı "hazır" tanımında anlaşır.

---




---

**KISIM IV — Modeli dağıtmak ve eğitim adımını tamamlamak**

*Ağırlıklar, aktivasyonlar, gradyanlar ve optimizer durumları gerçekten nereye gidiyor?*


## 13. Modeli bölme sanatı: TP, PP, DP, FSDP, CP, EP

Binlerce GPU'muzun olduğunu varsayalım. Önümüzdeki 405 milyar parametreli modeli nasıl paylaştıracağız?

Bu sorunun tek bir cevabı yok. Bazen aynı modelin farklı veriler üzerinde çalışmasını isteriz. Bazen tek bir katmanın ağırlıkları bir GPU'ya sığmaz. Bazen model derinliği yüzünden katmanları farklı GPU'lara dağıtırız. Bazen de büyük context veya MoE expert'leri başka bir bölme ekseni gerektirir.

Bu stratejileri birbirine rakip gibi değil, farklı sorunları çözen araçlar gibi düşünmek gerekiyor.

Model büyüdükçe tek bir paralellik stratejisi yeterli olmayabilir. Her strateji başka bir sorunu çözer.

### 13.1. Data Parallelism (DP): Farklı veriyi aynı modelle işle

Önce en anlaşılır yöntemle başlayalım. Dört öğrenciye aynı soruları öğretmek yerine aynı modelin dört kopyasını dört farklı veri grubunda eğitiyoruz. Her kopya kendine verilen veriden gradyan hesaplıyor; ardından sonuçlar birleştiriliyor. Böylece aynı zamanda daha fazla örnek üzerinde çalışabiliyoruz.

DP'de her grup modelin bir kopyasını tutar; farklı örneklerle forward ve backward çalışır; gradyanlar senkronize edilir.

<svg viewBox="0 0 720 296" role="img" aria-label="Veri paralelliği. Global batch A, B ve C parçalarına bölünür. Her parça tam model kopyası tutan bir GPU grubuna gider; grup ileri ve geri geçişi çalıştırıp A, B ve C gradyanlarını üretir. AllReduce gradyanların ortalamasını alır ve her kopya aynı optimizer adımını uygular. Veriyle ölçeklemek kolaydır, ama her kopya tüm model durumunu saklar ve gradyanlar her adımda senkronize edilmelidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dp-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Veri paralelliği: aynı model, farklı veri, ortak güncelleme</text>
<rect x="16" y="104" width="96" height="60" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="64.0" y="131.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Global batch</text>
<text x="64.0" y="146.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">3&#x27;e bölünür</text>
<path d="M112 134 C 130 134, 130 70, 150 70" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="56" width="58" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="179.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">parça A</text>
<line x1="208" y1="70" x2="224" y2="70" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="46" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU grubu A</text>
<text x="319" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tam model kopyası · ileri + geri</text>
<line x1="412" y1="70" x2="428" y2="70" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="56" width="70" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad A</text>
<path d="M500 70 C 520 70, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M112 134 C 130 134, 130 132, 150 132" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="118" width="58" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="179.0" y="135.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">parça B</text>
<line x1="208" y1="132" x2="224" y2="132" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="108" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="128" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU grubu B</text>
<text x="319" y="144" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tam model kopyası · ileri + geri</text>
<line x1="412" y1="132" x2="428" y2="132" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="118" width="70" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="465.0" y="135.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad B</text>
<path d="M500 132 C 520 132, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M112 134 C 130 134, 130 194, 150 194" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="180" width="58" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="179.0" y="197.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">parça C</text>
<line x1="208" y1="194" x2="224" y2="194" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="170" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="190" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU grubu C</text>
<text x="319" y="206" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tam model kopyası · ileri + geri</text>
<line x1="412" y1="194" x2="428" y2="194" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="180" width="70" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="465.0" y="197.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad C</text>
<path d="M500 194 C 520 194, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="540" y="106" width="164" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="622.0" y="132.0" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">AllReduce</text>
<text x="622.0" y="147.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">gradyan ortalaması</text>
<path d="M622 162 V240 H319 V220" marker-end="url(#dp-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="470" y="256" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">her kopyada aynı optimizer adımı</text>
<text x="16" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">+ veriyle ölçeklemek kolay   − her kopya tüm model durumunu saklar; her adımda gradyan senkronu</text>
</svg>

**Avantaj:** Veriye göre ölçeklenmek görece basit. **Bedel:** Kopyalanan model durumunun belleği ve gradient synchronization maliyeti.

DP kendi iletişimine ne zaman yenilir? Hesap batch'le büyür, gradyan AllReduce'u ise model boyutuyla sabittir. Scaling Book'un birinci mertebe koşulu [K11]:

$$
\frac{B}{X}\gtrsim\frac{C_{\mathrm{peak}}}{W_{\mathrm{collective}}}
$$

Şöyle okuyun: $B/X$ GPU başına token batch'i, $C_{\mathrm{peak}}$ GPU'nun tepe FLOP/s'i, $W_{\mathrm{collective}}$ kolektifin üzerinde çalıştığı bant genişliğidir. Düğüm içinde (NVLink, 450 GB/s) bu $990\times10^{12}/450\times10^{9}=2.200$, düğümler arasında (düğüm çıkışı 400 GB/s) $990\times10^{12}/400\times10^{9}=2.475$ token/GPU eder. GPU başına batch bunun altına düştüğünde gradyan senkronu hesaptan uzun sürer; overlap'in ve FSDP zamanlamasının önemi buradan gelir.

### 13.2. Tensor Parallelism (TP): Aynı katmanın matrisini böl

Peki tek bir katman bile GPU belleğine sığmıyorsa? Bu durumda modelin yalnızca verisini değil, **katmanın içindeki ağırlık matrisini** bölmemiz gerekir. Dört kişinin aynı kitabın farklı sayfalarını okumasından çok, aynı dev hesaplama tablosunun farklı sütunlarını hesaplaması gibi düşünebilirsiniz.

Katman çarpımı $Y=XW$ için sütunları ayırarak:

$$
W=[W_1\ W_2\ W_3\ W_4],\qquad
Y=[XW_1\ XW_2\ XW_3\ XW_4]
$$

Her GPU kendi $W_i$ parçasını işler. Sonuçlar daha sonra mimariye göre birleştirilir veya sonraki işlemde reduction ile bütünleşir. Megatron tarzı Transformer MLP'de column-parallel ve row-parallel işlemler birlikte kullanılır; bazı aşamalarda AllReduce/ReduceScatter gerekir. **Her TP katmanında mutlaka aynı sayıda blocking AllReduce vardır** iddiası doğru değildir.

<svg viewBox="0 0 720 352" role="img" aria-label="Megatron tarzı MLP&#x27;de dört GPU ile tensor paralelliği. Her GPU aynı X girdisini alır. GPU i, W1&#x27;in i. sütun bloğunu tutar ve iletişim olmadan act(X·W1ᵢ)&#x27;yi hesaplar; ardından W2&#x27;nin i. satır bloğuyla çarpıp kısmi sonuç Zᵢ&#x27;yi üretir. AllReduce dört kısmi sonucu toplayıp Z&#x27;yi verir. Bu AllReduce her katmanın kritik yolundadır; bu yüzden TP genellikle NVLink alanında tutulur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tp-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Megatron tarzı MLP&#x27;de tensor paralelliği (TP = 4)</text>
<rect x="16" y="140" width="70" height="60" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="51.0" y="175.76" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-family:var(--font-mono)">X</text>
<text x="51" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">X her GPU&#x27;da</text>
<text x="51" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">aynı</text>
<path d="M86 170 C 100 170, 100 70, 116 70" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="44" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<rect x="196" y="54" width="82" height="32" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="74.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 0]</text>
<line x1="278" y1="70" x2="292" y2="70" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="54" width="82" height="32" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="335.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="70" x2="390" y2="70" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="54" width="116" height="32" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="450.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[0, :] = Zᵢ</text>
<path d="M518 70 C 540 70, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M86 170 C 100 170, 100 134, 116 134" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="108" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="130" y="126" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<rect x="196" y="118" width="82" height="32" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="237.0" y="138.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 1]</text>
<line x1="278" y1="134" x2="292" y2="134" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="118" width="82" height="32" rx="4" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="335.0" y="137.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="134" x2="390" y2="134" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="118" width="116" height="32" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="450.0" y="137.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[1, :] = Zᵢ</text>
<path d="M518 134 C 540 134, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M86 170 C 100 170, 100 198, 116 198" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="172" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="130" y="190" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<rect x="196" y="182" width="82" height="32" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="237.0" y="202.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 2]</text>
<line x1="278" y1="198" x2="292" y2="198" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="182" width="82" height="32" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="335.0" y="201.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="198" x2="390" y2="198" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="182" width="116" height="32" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="450.0" y="201.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[2, :] = Zᵢ</text>
<path d="M518 198 C 540 198, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M86 170 C 100 170, 100 262, 116 262" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="236" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="130" y="254" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<rect x="196" y="246" width="82" height="32" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="266.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 3]</text>
<line x1="278" y1="262" x2="292" y2="262" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="246" width="82" height="32" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="335.0" y="265.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="262" x2="390" y2="262" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="246" width="116" height="32" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="450.0" y="265.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[3, :] = Zᵢ</text>
<path d="M518 262 C 540 262, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="196" y="302" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sütun paralel (iletişim yok)</text>
<text x="392" y="302" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">satır paralel (kısmi toplam)</text>
<rect x="562" y="140" width="142" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="633.0" y="167.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">AllReduce</text>
<text x="633.0" y="183.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">Z = Σ Zᵢ</text>
<text x="633" y="222" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">kritik yolda,</text>
<text x="633" y="236" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">her katmanda → NVLink</text>
<text x="16" y="324" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Her GPU W1 ve W2&#x27;nin ¼&#x27;ünü tutar. MLP ileri geçişi başına bir AllReduce: küçük, sık, gecikmeye duyarlı;</text>
<text x="16" y="340" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">bu yüzden mümkünse TP&#x27;yi NVLink alanında tutun.</text>
</svg>

TP'nin neden mümkünse yüksek hızlı NVLink alanında seçildiğini artık anlayabiliriz: her katmanda sık sık haberleşme gerekebilir. Ancak **TP>8 teknik olarak mümkündür**; seçim, model boyutu, ağ ve benchmark'a bağlıdır. [K3]

Bu sezgiyi bir eşitsizliğe çevirebiliriz. $Y$ GPU'ya bölünmüş bir Megatron MLP'sinde hesap süresi ve AllReduce süresi kabaca şöyledir [K11]:

$$
T_{\mathrm{hesap}}=\frac{4BDF}{Y\,C_{\mathrm{peak}}},\qquad T_{\text{iletişim}}=\frac{4BD}{W_{\mathrm{collective}}}
$$

Şöyle okuyun: $B$ token sayısı, $D$ gizli boyut, $F$ MLP ara boyutu, $Y$ TP derecesi. Hesabın iletişimden uzun sürmesi için $T_{\mathrm{hesap}}\ge T_{\text{iletişim}}$ olmalıdır:

$$
Y\le\frac{F\,W_{\mathrm{collective}}}{C_{\mathrm{peak}}}
$$

Llama 3 70B için $F=28.672$. NVLink'te $C/W=990\times10^{12}/450\times10^{9}=2.200$ olduğundan $Y\le28.672/2.200\approx13$; ikinin kuvveti olarak doğal seçim **8**'dir. Aynı TP'yi düğümler arasında GPU başına 50 GB/s'lik NIC üzerinden yapmaya kalkarsak $C/W=19.800$ olur ve sınır $Y\le1{,}45$'e düşer: TP neredeyse hiç ölçeklenmez. TP = 8'in ikinci, pratik nedeni bölünebilirliktir: Llama 3 70B'nin 64 attention head'i ve 8 KV head'i vardır [K21]. TP = 8'de her GPU tam bir KV head ve sekiz query head alır; TP = 16'da KV head'lerin kopyalanması gerekir.

### 13.3. Pipeline Parallelism (PP): Katmanları farklı GPU'lara ver

Bir fabrikanın üretim hattını hayal edin: ilk istasyon gövdeyi hazırlıyor, ikinci istasyon montaj yapıyor, üçüncü istasyon kontrol ediyor. Aynı ürün tüm aşamalardan sırayla geçmek zorunda, ama üretim hattı dolduğunda farklı ürünler farklı istasyonlarda aynı anda işlenebilir.

Pipeline Parallelism'in sezgisi budur. Transformer'ın ilk katmanları bir GPU grubunda, sonrakiler başka bir gruptadır. Ancak hat yeni başladığında ve iş bittiğinde bazı istasyonlar boş kalabilir; buna **pipeline bubble** deriz.

<svg viewBox="0 0 720 302" role="img" aria-label="4 stage ve 6 microbatch&#x27;li, yalnızca ileri geçişi gösteren 9 zaman dilimlik pipeline çizelgesi. Stage 0, 1–6. dilimlerde microbatch 1–6&#x27;yı çalıştırır; sonraki her stage bir dilim geç başlar, stage 3 onları 4–9. dilimlerde çalıştırır. Sonraki stage&#x27;lerin başındaki ve önceki stage&#x27;lerin sonundaki boş dilimler pipeline bubble&#x27;dır: stage başına 9 dilimin 3&#x27;ü, yaklaşık %33; (p − 1)/(m + p − 1) ile aynıdır. 16 stage&#x27;de ideal bubble 16 microbatch için %48,4, 32 için %31,9, 128 için %10,5&#x27;tir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Pipeline bubble: p = 4 stage, m = 6 microbatch (ileri geçiş)</text>
<text x="142.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t1</text>
<text x="206.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t2</text>
<text x="270.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t3</text>
<text x="334.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t4</text>
<text x="398.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t5</text>
<text x="462.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t6</text>
<text x="526.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t7</text>
<text x="590.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t8</text>
<text x="654.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t9</text>
<text x="16" y="74" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 0</text>
<rect x="112" y="56" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="142.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="176" y="56" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="206.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="240" y="56" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="270.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="304" y="56" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="334.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="368" y="56" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="398.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="432" y="56" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="462.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<rect x="496" y="56" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="560" y="56" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="624" y="56" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="16" y="112" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 1</text>
<rect x="112" y="94" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="176" y="94" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="206.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="240" y="94" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="270.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="304" y="94" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="334.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="368" y="94" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="398.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="432" y="94" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="462.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="496" y="94" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="526.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<rect x="560" y="94" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="624" y="94" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="16" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 2</text>
<rect x="112" y="132" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="176" y="132" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="240" y="132" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="270.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="304" y="132" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="334.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="368" y="132" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="398.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="432" y="132" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="462.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="496" y="132" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="526.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="560" y="132" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="590.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<rect x="624" y="132" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="16" y="188" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 3</text>
<rect x="112" y="170" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="176" y="170" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="240" y="170" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="304" y="170" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="334.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="368" y="170" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="398.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="432" y="170" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="462.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="496" y="170" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="526.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="560" y="170" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="590.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="624" y="170" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="654.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<text x="110" y="222" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">kesikli kırmızı = bubble (boş stage)</text>
<rect x="16" y="234" width="688" height="56" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="254" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Bubble ≈ (p − 1) / (m + p − 1) = burada 3 / 9 ≈ %33.</text>
<text x="28" y="274" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">p = 16: m = 16 → %48,4 · m = 32 → %31,9 · m = 128 → %10,5 (idealize, eşit stage)</text>
</svg>

Her stage, modelin bazı ardışık Transformer katmanlarını tutar. Stage'ler arasında ağırlık matrisi yerine sıklıkla **aktivasyon ve gradient sınır tensörleri** taşınır.

Bu sınır tensörü ne kadar büyük? Llama 3 405B'nin gizli boyutu 16.384'tür; 8.192 token'lık bir microbatch'in BF16 aktivasyonu $8192\times16384\times2\approx268$ MB eder. TP = 8 bunu sekiz GPU'ya paylaştırdığı için her GPU komşu stage'e microbatch başına yaklaşık **34 MB** gönderir [K5]. TP'nin katman başına iki AllReduce'una kıyasla bu hacim küçüktür: PP'nin asıl bedeli bant genişliği değil, birazdan göreceğimiz **bubble**'dır.

Ama ilk microbatch son stage'e gelene kadar diğer stage'ler bir süre boş kalabilir. Buna **pipeline bubble** denir.

İdealize edilmiş, eşit maliyetli stage ve klasik 1F1B benzeri çizelge için yaklaşık bubble oranı:

$$
Bubble\approx\frac{p-1}{m+p-1}
$$

- $p$: pipeline stage sayısı.
- $m$: microbatch sayısı.

Örneğin $p=16$:

| Microbatch sayısı | İdeal yaklaşık bubble oranı |
|---:|---:|
| 16 | 15/31 ≈ %48,4 |
| 32 | 15/47 ≈ %31,9 |
| 128 | 15/143 ≈ %10,5 |

Bu oran çizelgeye, pipeline'ın interleaving durumuna, backward zamanına ve yük dengesizliğine bağlıdır. Fazla microbatch ise activation memory veya scheduler maliyetlerini etkileyebilir.

### 13.4. FSDP ve ZeRO: Kopyaları neden saklıyoruz?

Data Parallelism'in bellekte bir maliyeti olduğunu hatırlayın: Her model kopyası parametreleri ve çoğu zaman optimizer durumlarını tutmak ister. Binlerce GPU'ya çıktığımızda aynı bilgiyi yüzlerce kez saklamak pahalı hale gelir.

FSDP ve ZeRO ailesi, bu tekrarı azaltmak için model durumunu bölmeyi hedefler. Fakat bellekte tasarruf edilen her şey bir yerlerde yeniden toplanmak veya uygun zamanda paylaşılmak zorunda kalabilir. Böylece yine aynı trade-off'a geliriz: **bellek tasarrufu karşılığında iletişim maliyeti**.

**FSDP (Fully Sharded Data Parallelism)**, veri paralel gruplardaki model parametreleri, gradient ve optimizer durumlarının uygun bileşenlerini parçalayarak belleği azaltır.

- **ZeRO-1:** Optimizer states parçalara ayrılır.
- **ZeRO-2:** Optimizer states + gradients parçalara ayrılır.
- **ZeRO-3:** Optimizer states + gradients + parameters parçalara ayrılır.

Böylece GPU başına tutulan *kalıcı* model durumu azalabilir. Ancak gerektiğinde **AllGather** ile parametre toplama ve **ReduceScatter** ile gradyan dağıtımı gereklidir.

<svg viewBox="0 0 720 298" role="img" aria-label="4 yönlü veri paralelliğinde Adam ile GPU başına model durumu, parametre başına byte. Düz DP her şeyi her GPU&#x27;da tutar: 2 byte parametre, 2 byte gradyan, 12 byte optimizer durumu, toplam 16 byte. ZeRO-1 optimizer&#x27;ı böler: 2 + 2 + 3 = 7 byte, ReduceScatter ile. ZeRO-2 gradyanları da böler: 2 + 0,5 + 3 = 5,5 byte. ZeRO-3 / FSDP parametreleri de böler: 0,5 + 0,5 + 3 = 4 byte; AllGather ve ReduceScatter gerekir. Kazanılan bellek iletişimle ödenir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="zr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Adam ile GPU başına model durumu (parametre başına byte, DP = 4)</text>
<text x="16" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">düz DP</text>
<text x="16" y="83" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">hiçbir şey bölünmez</text>
<rect x="130" y="54" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="54" width="43" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="196.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="218" y="54" width="263" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="350.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">12</text>
<text x="490" y="72" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">16 B</text>
<text x="704" y="72" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">—</text>
<text x="16" y="118" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-1</text>
<text x="16" y="133" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">optimizer bölünür</text>
<rect x="130" y="104" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="104" width="43" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="196.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="218" y="104" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="251.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="292" y="122" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">7 B</text>
<text x="704" y="122" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">ReduceScatter</text>
<text x="16" y="168" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-2</text>
<text x="16" y="183" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">+ gradyanlar bölünür</text>
<rect x="130" y="154" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="154" width="10" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="185" y="154" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="218.0" y="172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="259.0" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">5,5 B</text>
<text x="704" y="172" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">ReduceScatter</text>
<text x="16" y="218" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-3 / FSDP</text>
<text x="16" y="233" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">+ parametreler bölünür</text>
<rect x="130" y="204" width="10" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="204" width="10" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="152" y="204" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="185.0" y="222" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="226.0" y="222" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">4 B</text>
<text x="704" y="222" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">AllGather + ReduceScatter</text>
<rect x="16" y="254" width="12" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="34" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">parametre</text>
<rect x="120" y="254" width="12" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="138" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradyan</text>
<rect x="222" y="254" width="12" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="240" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">optimizer (FP32 master + 2 moment)</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Kazanılan bellek iletişimle ödenir: bölünmüş durum gerektiğinde toplanmalı veya dağıtılmalıdır.</text>
</svg>

**Sezgi:** Sekiz kişinin aynı 1.000 sayfalık kitabı ayrı ayrı saklaması yerine kitabın bölüm bölüm paylaşılması depolamayı azaltır. Fakat herkesin bir bölüme erişmesi gerektiğinde haberleşme oluşur. İşte FSDP'nin bellek–iletişim takası budur.

### 13.5. Context Parallelism (CP) ve Sequence Parallelism (SP)

**CP**, uzun diziyi (*sequence length*) farklı GPU'lara ayırır. Özellikle attention'da bir token'ın diğer konumlardaki K/V bilgisine ihtiyaç duyması ek haberleşme gerektirebilir. **SP**, bazı Transformer işlemlerinin aktivasyonlarını sekans boyutunda paylaştıran daha sınırlı bir yöntemdir. İkisi aynı şey değildir. [K3]

### 13.6. Expert Parallelism (EP): MoE'nin uzmanlarını dağıt

Herkesin aynı uzman tarafından yanıtlanması gerekmediğini düşünelim. Bir şirkette hukuk sorusu hukuk ekibine, finans sorusu finans ekibine gider. MoE mimarisinde de bütün token'lar aynı feed-forward alt ağını kullanmak zorunda değildir; bir yönlendirici seçili expert'leri etkinleştirir.

Bu, hesaplamayı azaltabilir ama yeni bir sorun doğurur: seçilen expert başka GPU'daysa aktivasyonun o GPU'ya ulaşması gerekir. İşte Expert Parallelism bu dağıtımı yönetir.

**Mixture-of-Experts (MoE)** modellerinde her token çoğunlukla sınırlı sayıda expert'e yönlendirilir. **Router**, token'ın hangi expert'lere gideceğini belirler. Experts farklı GPU'lara dağıtılıyorsa token aktivasyonları **AllToAll** ile yönlendirilir.

<svg viewBox="0 0 720 298" role="img" aria-label="Her biri bir uzman barındıran dört GPU ile expert paralelliği. Her GPU&#x27;daki token&#x27;lar router&#x27;ın seçtiği uzmanın rengindedir. AllToAll dispatch her token&#x27;ı uzmanının bulunduğu GPU&#x27;ya gönderir; E2 uzmanına altı, diğerlerine ikişer token gider ve E2 sıcak noktaya döner. Uzmanlar çalıştıktan sonra ikinci bir AllToAll (combine) sonuçları kaynak GPU&#x27;lara geri yollar. MoE token başına aktif FLOP&#x27;u azaltır ama her katmana iki ağ karışımı ekler; dengesiz router bir darboğaz yaratır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="me-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="me-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Expert paralelliği: token&#x27;lar uzmanlarına gider ve döner (2 × AllToAll)</text>
<rect x="16" y="50" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="75" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 0</text>
<rect x="80" y="59" width="22" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="91.0" y="74.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E0</text>
<rect x="107" y="59" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="118.0" y="74.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="134" y="59" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="145.0" y="74.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="16" y="102" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="127" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 1</text>
<rect x="80" y="111" width="22" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="91.0" y="126.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E1</text>
<rect x="107" y="111" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="118.0" y="126.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="134" y="111" width="22" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="145.0" y="126.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E3</text>
<rect x="16" y="154" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="179" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 2</text>
<rect x="80" y="163" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="91.0" y="178.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="107" y="163" width="22" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="118.0" y="178.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E0</text>
<rect x="134" y="163" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="145.0" y="178.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="16" y="206" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="231" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 3</text>
<rect x="80" y="215" width="22" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="91.0" y="230.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E3</text>
<rect x="107" y="215" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="118.0" y="230.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="134" y="215" width="22" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="145.0" y="230.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E1</text>
<rect x="196" y="50" width="92" height="196" rx="4" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="242" y="140" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">AllToAll</text>
<text x="242" y="156" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dispatch</text>
<line x1="166" y1="150" x2="194" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<line x1="288" y1="150" x2="316" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="318" y="50" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="330" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E0</text>
<text x="330" y="83" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 0</text>
<rect x="410" y="61" width="12" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="426" y="61" width="12" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="318" y="102" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="330" y="120" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E1</text>
<text x="330" y="135" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 1</text>
<rect x="410" y="113" width="12" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.6"/>
<rect x="426" y="113" width="12" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.6"/>
<rect x="318" y="154" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="330" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E2</text>
<text x="330" y="187" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 2</text>
<rect x="410" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="426" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="442" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="458" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="474" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="490" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="368" y="187" text-anchor="start" style="fill:var(--c-danger);font-size:10.5px;font-weight:700">sıcak!</text>
<rect x="318" y="206" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="330" y="224" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E3</text>
<text x="330" y="239" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 3</text>
<rect x="410" y="217" width="12" height="20" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<rect x="426" y="217" width="12" height="20" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<line x1="518" y1="150" x2="546" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="548" y="50" width="92" height="196" rx="4" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594" y="140" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">AllToAll</text>
<text x="594" y="156" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">combine</text>
<line x1="640" y1="150" x2="668" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<text x="686" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kaynak</text>
<text x="686" y="192" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">GPU&#x27;ya</text>
<text x="16" y="270" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Token başına aktif FLOP azalır, ama her MoE katmanında iki ağ karışımı olur;</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">dengesiz router da bir uzmanı herkesin beklediği darboğaza çevirir.</text>
</svg>

**Dense model:** Her token büyük ölçüde aynı ağırlık yolu üzerinde ilerler.

**MoE model:** Her token birçok expert arasından seçilen alt küme üzerinden ilerler; aktif FLOP azaltılabilir fakat iletişim ve yük dengesi yeni maliyetler oluşturur.

DeepSeek-V3 raporu, expert dağıtımı ve **DualPipe** adlı compute/communication örtüştürme yöntemini tartışır. Bu yaklaşımın gerçek faydası kullanılan mimari ve H800 ağ yapılandırmasıyla birlikte değerlendirilmelidir. [K9]

MoE, DP'nin kritik batch eşiğini de değiştirir. Her token $E$ expert'ten yalnızca $k$ tanesini çalıştırdığında GPU başına hesap $k/E$ oranında küçülür ama senkronize edilecek ağırlık aynı kalır; eşik $E/k$ kat büyür [K11]:

$$
\left(\frac{B}{X}\right)_{\mathrm{MoE}}\gtrsim\frac{E}{k}\cdot\frac{C_{\mathrm{peak}}}{W_{\mathrm{collective}}}
$$

$E=128$ ve $k=4$ için bu, düğümler arasında $32\times2.475=79.200$ token/GPU eder; hiçbir gerçek eğitimin ulaşmadığı bir sayı. Büyük MoE eğitiminde expert ağırlıklarının DP ile kopyalanmayıp EP ile GPU'lara dağıtılmasının nedeni budur; bedeli de şekildeki iki AllToAll'dur.

### 13.7. Hangi stratejiyi nereye yerleştirmeli?

| Strateji | Bölünen eksen | Ana hedef | İletişim örüntüsü |
|---|---|---|---|
| DP | Batch | Veri throughput'u | Gradient reduction |
| TP | Katman içi ağırlık | Büyük layer'ı paylaştırma | Sık collective |
| PP | Katman sırası | Derin modeli bölme | Stage komşularına aktivasyon |
| FSDP | Model durumu | VRAM tasarrufu | AllGather/ReduceScatter |
| CP | Sequence | Uzun bağlam ve aktivasyon | K/V/attention bilgisi paylaşımı |
| EP | Expert | MoE uzman dağıtımı | Token dispatch, AllToAll |

Yukarıda türettiğimiz iki sınır, TP ve DP için yan yana:

<svg viewBox="0 0 720 310" role="img" aria-label="H100 için 990 TFLOP/s&#x27;de iki iletişim roofline&#x27;ı. Solda tensor paralelliği: Y derecesi F · W / C&#x27;nin altında kalmalıdır. F = 28.672 olan Llama 3 70B MLP&#x27;si için 450 GB/s&#x27;lik NVLink Y ≈ 13&#x27;e izin verir, TP 8 tek DGX&#x27;e sığar; 50 GB/s&#x27;lik NIC yalnızca ≈ 1,45&#x27;e izin verir. Sağda veri paralelliği: gradyan senkronunu örtmek için GPU başına token batch&#x27;i C / W&#x27;yi aşmalıdır: düğüm içinde 2.200, düğüm başına 400 GB/s ile düğümler arasında 2.475 ve 128 expert&#x27;ten 4&#x27;ü etkin bir MoE için 79.200 token; sınır E/k kat büyür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="cr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">İletişim roofline&#x27;ları: bölmek ne zaman kazandırmaz?</text>
<rect x="16" y="40" width="340" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700;font-family:var(--font-mono)">TP: Y ≤ F · W / C</text>
<text x="28" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP, F = 28.672</text>
<text x="40" y="106" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NVLink 450 GB/s</text>
<rect x="40" y="112" width="236.218" height="22" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="282.21818181818185" y="128" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Y ≤ 13,0</text>
<text x="40" y="168" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NIC 50 GB/s</text>
<rect x="40" y="174" width="26.2465" height="22" rx="2" style="fill:var(--c-danger);fill-opacity:.55"/>
<text x="72.24646464646464" y="190" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Y ≤ 1,45</text>
<line x1="185" y1="92" x2="185" y2="230" style="stroke:var(--c-accent);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="185.0" y="244" text-anchor="middle" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">TP = 8 (bir DGX)</text>
<text x="40.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">0</text>
<text x="112.5" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">4</text>
<text x="185.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">8</text>
<text x="257.5" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">12</text>
<text x="330.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">16</text>
<rect x="364" y="40" width="340" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="60" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700;font-family:var(--font-mono)">DP: B/X ≥ C / W</text>
<text x="376" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradyan senkronunu örtmek için GPU başına token</text>
<text x="388" y="102" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">düğüm içi (450 GB/s)</text>
<rect x="388" y="108" width="49.9937" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="443.9937114000421" y="123" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2.200</text>
<text x="388" y="150" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">düğümler arası (400 GB/s/düğüm)</text>
<rect x="388" y="156" width="57.462" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="451.4619796773598" y="171" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2.475</text>
<text x="388" y="198" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">MoE, E = 128, k = 4 (× E/k)</text>
<rect x="388" y="204" width="277.214" height="20" rx="2" style="fill:var(--c-danger);fill-opacity:.55"/>
<text x="659.213876512066" y="219" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">79.200</text>
<text x="388.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">1K</text>
<text x="534.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">10K</text>
<text x="680.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">100K</text>
<text x="16" y="298" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">C = 990 TFLOP/s (H100 BF16). Sınırlar birinci mertebe (Scaling Book); overlap ve topoloji onları kaydırır, ama 10 kat değil.</text>
</svg>

**Dikkat:** Dünya büyüklüğünü çoğu dense model için $TP\times PP\times CP\times DP$ ile gösterebiliriz. EP de ayrıca eklenebilir, ancak gerçek uygulamalarda expert-parallel grupları başka paralellik eksenleriyle iç içe veya örtüşük tanımlanabilir. Bu nedenle her modelde beş boyutu mekanik biçimde çarpmak doğru bir kural değildir.

### 13.8. Gerçek Llama 3 405B yapılandırması

Meta teknik raporundaki örneklerden ikisi: [K5]

| GPU | TP | CP | PP | DP | Sequence length | Raporlanan BF16 MFU |
|---:|---:|---:|---:|---:|---:|---:|
| 16.384 | 8 | 1 | 16 | 128 | 8.192 | %41 |
| 16.384 | 8 | 16 | 16 | 8 | 131.072 | %38 |

İlk satır için:

$$
8\times1\times16\times128=16{,}384
$$

<svg viewBox="0 0 720 324" role="img" aria-label="Llama 3 405B&#x27;nin 16.384 H100&#x27;e yerleşimi. 8 GPU&#x27;luk tensor paralel grup bir DGX&#x27;i doldurur ve NVLink üzerinden konuşur. Bu sunuculardan 16&#x27;sı zincir hâlinde bir model kopyasının 16 pipeline stage&#x27;ini, yani 128 GPU&#x27;yu oluşturur ve aktivasyonları sunucudan sunucuya aktarır. 128 kopya veri paralel boyutu oluşturur ve gradyanları senkronlar. 8K bağlamda TP 8, CP 1, PP 16, DP 128 düzeni %41, 128K bağlamda TP 8, CP 16, PP 16, DP 8 düzeni %38 BF16 MFU&#x27;ya ulaştı. En konuşkan eksen en hızlı bağlantıda kalır; Meta&#x27;nın ağı RoCE idi." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="l4-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="l4-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="l4-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="l4-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Llama 3 405B, 16.384 H100: 8 × 16 × 128 (TP × PP × DP)</text>
<path d="M620 58 H630 V208 H40" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<path d="M613 51 H623 V201 H33" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<rect x="16" y="44" width="600" height="150" rx="10" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="28" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">bir model kopyası = 128 GPU</text>
<rect x="28" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="33" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="33" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="33" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="33" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="45.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">0</text>
<line x1="62" y1="116" x2="64" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="64" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="69" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="69" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="69" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="69" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="81.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">1</text>
<line x1="98" y1="116" x2="100" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="100" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="105" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="105" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="105" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="105" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="117.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">2</text>
<line x1="134" y1="116" x2="136" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="136" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="141" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="153.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">3</text>
<line x1="170" y1="116" x2="172" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="172" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="177" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="177" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="177" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="177" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="189.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">4</text>
<line x1="206" y1="116" x2="208" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="208" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="213" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="213" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="213" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="213" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="225.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">5</text>
<line x1="242" y1="116" x2="244" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="244" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="249" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="261.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">6</text>
<line x1="278" y1="116" x2="280" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="280" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="285" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="285" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="285" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="285" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="297.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">7</text>
<line x1="314" y1="116" x2="316" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="316" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="321" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="321" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="321" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="321" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="333.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">8</text>
<line x1="350" y1="116" x2="352" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="352" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="357" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="357" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="357" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="357" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="369.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">9</text>
<line x1="386" y1="116" x2="388" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="388" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="393" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="393" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="393" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="393" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="405.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">10</text>
<line x1="422" y1="116" x2="424" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="424" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="429" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="429" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="429" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="429" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="441.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">11</text>
<line x1="458" y1="116" x2="460" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="460" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="465" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="465" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="465" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="465" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="477.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">12</text>
<line x1="494" y1="116" x2="496" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="496" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="501" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="501" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="501" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="501" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="513.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">13</text>
<line x1="530" y1="116" x2="532" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="532" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="537" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="537" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="537" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="537" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="549.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">14</text>
<line x1="566" y1="116" x2="568" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="568" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="573" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="573" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="573" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="573" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="585.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">15</text>
<text x="28" y="184" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">PP = 16 stage (her biri bir DGX) → aktivasyon sunucudan sunucuya geçer</text>
<rect x="632" y="44" width="72" height="92" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="668" y="62" text-anchor="middle" style="fill:var(--c-accent);font-size:12px;font-weight:700">TP = 8</text>
<rect x="650" y="70" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="70" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="650" y="85" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="85" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="650" y="100" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="100" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="650" y="115" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="115" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="668" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir DGX,</text>
<text x="668" y="161" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVLink</text>
<text x="632" y="226" text-anchor="end" style="fill:var(--c-warn);font-size:12.5px;font-weight:700">× 128 kopya = DP</text>
<text x="16" y="226" text-anchor="start" style="fill:var(--c-warn);font-size:11px">gradyanlar kopyalar arasında senkronlanır</text>
<rect x="16" y="240" width="688" height="24" rx="4" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">8K bağlam</text>
<text x="160" y="256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">TP 8 · CP 1 · PP 16 · DP 128</text>
<text x="692" y="256" text-anchor="end" style="fill:var(--c-accent);font-size:12px;font-weight:700">BF16 MFU %41</text>
<rect x="16" y="268" width="688" height="24" rx="4" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="284" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">128K bağlam</text>
<text x="160" y="284" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">TP 8 · CP 16 · PP 16 · DP 8</text>
<text x="692" y="284" text-anchor="end" style="fill:var(--c-accent);font-size:12px;font-weight:700">BF16 MFU %38</text>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hızlı bağlantı en konuşkan ekseni taşır: TP NVLink içinde kalır, PP ve DP (RoCE) ağı geçer. Meta %38–43 MFU raporlar.</text>
</svg>

**Çok önemli:** Kaynağın girişindeki '%38 MFU' bir temsilî sorun tanımıydı. Meta'nın tüm 16.384 H100 eğitimleri her zaman %38'e sıkışmış değildi. Aynı raporda iş yüküne ve paralellik ayarına göre **%38–43** aralığında MFU raporlanıyor. 8K context'li 16K GPU koşulunda **%41**, 128K context'li senaryoda **%38** görülmüştü.

---

### 13.9. GPU'ları bölmek yetmez: İşleri doğru zamanda ve doğru yerde başlatmak

TP, PP, DP ve EP ile modelin **nasıl** paylaştırılacağını öğrendik. Şimdi şu soruyu soralım: Büyük şirketin aynı kümesinde aynı anda 405B pretraining, 70B fine-tuning, deneysel notebook ve canlı inference servisleri varsa **hangi iş hangi GPU'yu alacak?**

Bu, model paralelliğinden ayrı bir problem olan **GPU orkestrasyonu (orchestration)** ve **scheduler** tasarımıdır.

Bir otelde sekiz kişilik bir konferans ekibi düşünün. Sadece dört kişiye oda verirseniz toplantı başlayamaz; buna rağmen odalar dolu göründüğü için başka müşteriler de yerleşemeyebilir. Dağıtık eğitim işlerinde de yalnızca işin bir bölümünü çalıştırmak çoğu zaman faydasızdır.

Bu nedenle **gang scheduling / all-or-nothing admission**, birbirine bağımlı GPU çalışanlarının gerekli kaynakları birlikte alabilmesini hedefler. Kubernetes ekosistemindeki Kueue bu problemi quota ve admission mekanizmalarıyla ele alır. Ancak kaynak kabulünü garanti etmek, bütün worker'ların hiç hata olmadan ve aynı mikro saniyede başlayacağını garanti etmez. [K24]

| Kavram | Kısa tanım | AI altyapısında sorulan soru |
|---|---|---|
| **Scheduler** | İşleri uygun donanıma yerleştiren bileşen | Bu iş hangi node'larda başlayacak? |
| **Gang scheduling** | Bağımlı iş parçalarını birlikte kabul etme | TP/PP worker'ları topluca hazır mı? |
| **Topology-aware placement** | GPU, NIC, NVLink ve rack yakınlığını dikkate alma | TP aynı hızlı bağlantı adasında mı? |
| **Quota** | Takım/tenant kaynak sınırı | Bir ekip tüm kümeyi kilitliyor mu? |
| **Preemption** | Daha yüksek öncelikli işe yer açmak | Canlı servis kritikse ne durdurulabilir? |
| **Fragmentation** | Serbest kaynakların kullanılamayacak biçimde dağılması | Toplam sekiz boş GPU var ama aynı uygun node'da mı? |

Kubernetes, Slurm ve benzeri araçların arasında yalnızca özellik listesine bakarak seçim yapılmaz. Büyük, uzun süren senkronize training işleri ile küçük ve dinamik inference işleri farklı planlama özellikleri ister. Bazı platformlarda Kubernetes üstünde ek GPU operatörleri ve kuyruk/admission bileşenleri; HPC ağırlıklı ortamlarda Slurm benzeri batch scheduler'lar tercih edilebilir.

#### Çok kullanıcılı GPU paylaşımı: MIG, time slicing ve MPS aynı şey değildir

**MIG (Multi-Instance GPU)**, destekleyen GPU'yu bağımsız GPU instance'larına donanımsal olarak ayırabilir. Böylece küçük iş yükleri için öngörülebilir kapasite ve belirli izolasyon kazanımları elde edilebilir. Fakat tüm GPU modellerinin aynı MIG özelliklerini sunduğu varsayılamaz; fiziksel topoloji ve destek matrisi kontrol edilmelidir. [K25]

**Time slicing**, süreçlere GPU zamanını sırayla paylaştırır. **MPS (Multi-Process Service)** ise desteklenen CUDA süreçlerinin GPU kaynaklarını daha verimli ortak kullanmasına yardımcı olabilir. Bunlar MIG ile aynı izolasyon veya performans garantisini vermek zorunda değildir.

Training'in bütün GPU'ları ve hızlı bağlantıları kesintisiz kullanması gereken bir durumda agresif çoklu paylaşım, sayısal kullanım oranını yükseltirken işin bitiş süresini kötüleştirebilir. Başarılı bir cluster scheduler'ı bu yüzden sadece **allocated GPU / total GPU** oranına bakmaz; **işlerin tamamlanma süresini, kuyruk beklemesini ve SLO'yu** da dikkate alır.

#### Ayrı havuz stratejisi

Üretim koşullarında sık kullanılan bir yaklaşım, ihtiyaçlara göre farklı **GPU pool** tanımlamaktır:

- **Latency-sensitive inference:** Gecikme hedefi ve hızlı ölçekleme önceliği.
- **Senkronize training:** Toplu kabul, homojen donanım ve sabit topoloji önceliği.
- **Fine-tuning / development:** Daha küçük iş, öncelik ve kota yönetimi.

Bu zorunlu bir mimari kalıp değildir; iş yükleri benzerse ortak havuz daha verimli olabilir. Asıl amaç kaynak yerleşiminin **iş yükü davranışına göre** planlanmasıdır.

---


## 14. Tek bir LLM training step boyunca veri gerçekte nereye gider?

Şimdiye kadar parçalara baktık. Bu noktada hepsini aynı sahnede görmemiz gerekiyor.

Bir depoda bekleyen tokenize edilmiş veri parçasını seçelim ve onun tek bir eğitim adımında başına gelenleri takip edelim. Veri nereden okunuyor? Hangi GPU'ya gidiyor? Hangi noktada Tensor Core çalışıyor? Hangi noktada GPU başka GPU'yu bekliyor? Ağırlıklar ne zaman güncelleniyor? Bir kayıt almak istersek hangi parça depolamaya yazılıyor?

Bu yolculuğu öğrendiğinizde “GPU niçin bekliyor?” sorusuna yaklaşımınız değişecek. Sorunu belirli bir noktaya yerleştirebileceksiniz.

Artık parçaları biliyoruz. Şimdi büyük bir dense modelin **tek eğitim adımını** sistemin içinden geçiyormuş gibi izleyelim. Llama 3 405B boyutlarını ve Meta raporundaki TP=8, PP=16, DP=128 düzenini referans alacağız. Fakat aşağıdaki zaman bütçeleri ve veri boyutu örnekleri **öğretici yaklaşık hesap** olacaktır, Meta'nın raporlamadığı ayrıntıları ölçülmüş gerçek gibi göstermeyeceğiz. [K5]

### Adım 1 — Token verisi depodan okunur

Eğitim dosyaları genellikle bir ön işleme hattından geçirilip tokenize edilmiş örneklere dönüştürülür. **Tokenization**, metni model sözlüğündeki sayısal token kimlikleri dizisine dönüştürme işlemidir.

<svg viewBox="0 0 720 300" role="img" aria-label="1. adımda token verisinin yolu: dağıtık dosya sistemi veya obje deposu, örnek seçen, sıralayan, padding/packing yapan CPU DataLoader worker&#x27;larınca okunur; batch pinned host RAM&#x27;e iner ya da direkt I/O kullanılır; ardından PCIe üzerinden H2D kopyayla GPU HBM&#x27;e gelir ve embedding ile Transformer ileri geçişi başlar. Yanında boyut kıyası: 16 milyon token × 4 byte, global step başına yalnızca 64 MB token ID eder; yaklaşık 7,5 TB&#x27;lık checkpoint dalgasının yanında küçücüktür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dp2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dp2-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">1. adımın içi: token&#x27;lar HBM&#x27;e nasıl ulaşır?</text>
<rect x="16" y="42" width="300" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="166.0" y="68.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Dağıtık dosya sistemi / obje deposu</text>
<line x1="166" y1="84" x2="166" y2="98" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="95" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">dosya okuma</text>
<rect x="16" y="100" width="300" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="166.0" y="126.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU DataLoader worker&#x27;ları</text>
<line x1="166" y1="142" x2="166" y2="156" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">örnek seçme, sıralama, padding/packing</text>
<rect x="16" y="158" width="300" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="166.0" y="184.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Pinned host RAM (veya direkt I/O)</text>
<line x1="166" y1="200" x2="166" y2="214" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="211" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">PCIe üzerinden H2D kopya</text>
<rect x="16" y="216" width="300" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="166.0" y="242.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM</text>
<text x="326" y="242" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">→ embedding → Transformer forward</text>
<rect x="380" y="42" width="324" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="392" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Global step başına (Llama 3 405B ölçeği)</text>
<text x="392" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">16 × 10⁶ token × 4 B</text>
<text x="392" y="106" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">= 64 MB token ID</text>
<rect x="392" y="116" width="3" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.8"/>
<text x="400" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">checkpoint yanında küçücük:</text>
<rect x="392" y="140" width="300" height="16" rx="2" style="fill:var(--c-danger);fill-opacity:.45"/>
<text x="392" y="174" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:600">~7,5 TB checkpoint dalgası</text>
<text x="392" y="190" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-style:italic">(çubuklar ölçekli değil)</text>
<text x="16" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Metin pretraining&#x27;de zor olan ham token baytı değil, erişim düzeni ve checkpoint dalgasıdır; prefetch okumayı hesabın arkasına saklar.</text>
</svg>

**DataLoader**, uygulamanın doğru eğitim örneklerini zamanında sağlamasından sorumludur. **Pinned memory (page-locked RAM)**, CPU–GPU veri aktarımında kullanılabilen sabitlenmiş bellek türüdür. **Prefetch**, gelecek batch'i mevcut batch hesaplanırken önden hazırlama stratejisidir. **Packing**, kısa örnekleri token kapasitesini daha verimli kullanacak biçimde birleştirme yaklaşımıdır.

Sezgisel fakat şaşırtıcı hesap: Bir global step 16 milyon adet token içeriyorsa ve token kimlikleri 4-byte tamsayı olarak saklanıyorsa, ham token ID hacmi:

$$
16\times10^6\ \mathrm{token}\times 4\ \mathrm{byte/token}=64\ \mathrm{MB}
$$

Bu sayı büyük bir checkpoint'e göre küçüktür! Dolayısıyla saf metin pretraining'de depolama hattının ağır yükü çoğu zaman token ID'lerinin ham byte kapasitesinden çok, birçok çalışan arasındaki erişim düzeni ve **checkpoint burst** yüküdür. Görüntü/video veri kümelerinde bu sonuç değişebilir.

Saniye cinsinden de tablo değişmez. 16.384 GPU'nun her biri ~400 TFLOP/s faydalı hesap üretiyorsa ve token başına $6\times405\times10^{9}$ FLOP gerekiyorsa küme saniyede $16384\times400\times10^{12}/(6\times405\times10^{9})\approx2{,}7$ milyon token işler; 4 byte'lık token ID'lerle bu yalnızca **~11 MB/s** eder. Zor olan bant genişliği değil, her GPU'yu takılmadan beslemektir. Bu yüzden üretim çerçeveleri veriyi önceden tokenize edip bitişik bir `.bin` token ID dosyası ve belge ofsetlerini tutan bir `.idx` dosyası olarak saklar ve `mmap` ile okur (bkz. §16.3). [K5][K29]

### Adım 2 — Tensor parallel grup kendi katman parçasını işler

Veri artık GPU'da. Önce token ID'leri embedding vektörlerine dönüştürülür, sonra Transformer katmanlarındaki matris çarpmaları başlar. Modeli TP=8 olarak dağıttıysak, sekiz GPU'nun her biri ilgili katmanın ağırlıklarının kendi payına düşen bölümüyle çalışır.

Burada tekrar Roofline'a dönelim: Ağırlıkları HBM'den getirmek, shared memory'de tekrar kullanmak ve Tensor Core'ları mümkün olduğunca verimli çalıştırmak, tek GPU'da öğrendiğimiz tüm kavramların iş başına geçtiği andır.

Sekiz H100 birlikte bir katman içindeki büyük matrisleri hesaplayabilir. Her GPU'nun kendi weight shard'ı vardır. GEMM ve attention kernel'ları SM'lerde çalışırken ağırlıklar HBM ve yakın önbelleklerden okunur.

Bu noktada iki farklı hesaplamayı ayırmalıyız:

1. **Useful compute:** Katmanın gerçekten istediği matris çarpımları.
2. **Execution overhead:** Kernel başlatma, küçük tensör işlemleri, senkronizasyon ve veri biçimi dönüşümleri.

İkincisi de GPU'nun meşgul olmasına neden olabilir fakat teorik MFU hesabına aynı oranda katkı sağlamaz.

### Adım 3 — Gerekli kolektif iletişim gerçekleştirilir

Örneğin bir row-parallel projeksiyondaki kısmi sonuçların birleştirilmesi gerekebilir. Bu sırada **NCCL (NVIDIA Collective Communications Library)** desteklenen yol ve algoritmayla veri hareketini gerçekleştirir.

İdeal durumda iletişim daha önce hesaplanan bağımsız işler ile örtüştürülür. Ancak sonuç başka katmana girdi olacaksa o bağımlılık tamamlanmadan ilerlenemez.

Hacmi kestirelim: Llama 3 405B'de 8.192 token'lık microbatch'in aktivasyonu ~268 MB olduğundan NVLink'te tek bir TP AllReduce'u kabaca $2\cdot\frac{7}{8}\cdot268\ \mathrm{MB}/450\ \mathrm{GB/s}\approx1$ ms sürer. Bir pipeline stage'i yaklaşık 8 katman taşır (126/16); katman başına ileri geçişte 2, geri geçişte 2 AllReduce ve DP kopyası başına 16 microbatch ile bir adımda ~500 AllReduce, yani yaklaşık **0,5 sn** NVLink iletişimi olur. Bunun ne kadarı hesapla örtüşürse MFU'ya o kadar az yansır.

### Adım 4 — Pipeline stage aktivasyonunu sonraki stage'e gönderir

Stage 0 kendi katmanlarını bitirdiğinde ürettiği aktivasyonları stage 1'e aktarır. Tensor paralelliğinin aksine burada her adımda büyük ağırlık matrisi taşımak zorunda olmayabiliriz; genellikle **sınır aktivasyon tensörü** iletilir.

**Microbatch**, global batch'in pipeline içinde dolaştırılan küçük parçasıdır. Çok sayıda microbatch, pipeline stage'lerinin dolu tutulmasına yardımcı olabilir.

### Adım 5 — Loss ve backward

Forward sonunda model bir tahmin yaptı. Fakat eğitim yalnızca bir tahmin üretmek değildir; modelin hangi parametrelerinin nasıl değişmesi gerektiğini de bulmamız gerekir. Loss değerinin üzerinden geriye doğru yürüyerek gradyanları hesaplarız.

Bu aşama belleğe de yük bindirir. Çünkü backward sırasında bazı forward ara sonuçlarına yeniden ihtiyaç duyulur. Eğer hepsini saklarsak HBM dolabilir; eğer bazılarını saklamazsak geri giderken yeniden hesaplamak zorunda kalabiliriz. **Activation recomputation** tam olarak bu takastır.

Son stage modelin kayıp değerini oluşturur. Ardından chain rule kullanılarak gradyanlar ters yönde hesaplanır. **Backward**, ilgili aktivasyonlara ihtiyaç duyduğu için bellek açısından pahalı olabilir.

**Activation checkpointing / recomputation**, bazı ara aktivasyonları eğitim sırasında saklamayıp backward sırasında yeniden hesaplama tekniğidir. Diskteki eğitim checkpoint'i ile aynı şey değildir:

| Terim | Ne için kullanılır? |
|---|---|
| **Activation checkpointing** | GPU HBM alanından tasarruf etmek, gerekirse hesaplamayı tekrarlamak |
| **Training checkpoint** | Arızadan sonra eğitimi devam ettirebilmek için durumu kalıcı kaydetmek |

### Adım 6 — Gradients senkronize edilir

DP/FSDP grupları arasında gradyanlara ilişkin reduction ve sharding işlemleri yapılır. Gradyan senkronizasyonu, birçok farklı GPU'nun aynı optimizer ilerlemesine uyum sağlamasını amaçlar.

Eğitim performansını etkileyen mesele: **Gradyanların hazır olduğu an** ile **collective'ın bitmesi gereken an** birbirine ne kadar yakın? İletişimi backward ile örtüştürmek, bekleme süresini azaltabilir.

Llama 3 405B ölçeğinde: TP = 8 × PP = 16 ile her GPU yaklaşık $405\times10^{9}/128\approx3{,}2$ milyar parametrelik bir dilimden sorumludur. FSDP bu dilimin FP32 gradyanlarını reduce-scatter (~12,7 GB), BF16 ağırlıklarını all-gather (~6,3 GB) ile taşır: toplam ~19 GB, GPU başına 50 GB/s'lik NIC'te yaklaşık **0,38 sn**. Backward ile örtüştüğü ölçüde adım süresine hiç yansımaz.

### Adım 7 — Optimizer ağırlıkları günceller

Örneğin AdamW optimizer, gradient, birinci moment ve ikinci moment ile parametreleri günceller. **Weight decay**, belirli parametreleri düzenlileştirmek için kullanılan bileşendir.

Optimizer durumları, dağıtılmış optimizer/FSDP yaklaşımında farklı GPU'larda parçalı tutulabilir. Böylece model durumunun hepsini her GPU'da tekrarlamak gerekmez.

### Adım 8 — Belirli aralıklarla checkpoint yazılır

Eğitim adımı bitti, optimizer ağırlıkları güncelledi. Peki tüm sunucular tam şimdi kapansa, eğitime nereden devam edebiliriz? Eğer kalıcı bir kayıt yoksa, çok değerli hesaplamaları yeniden yapmak zorunda kalabiliriz.

Bu nedenle eğitim durumunu periyodik olarak saklarız. Checkpoint dediğimiz şey basit bir `.pt` dosyası olmaktan çok daha kapsamlı olabilir; özellikle model durumu terabaytlarla ölçüldüğünde depolama ve hata kurtarma mimarisinin parçasıdır.

Model ağırlıkları, optimizer durumu, RNG ve eğitim ilerlemesini yeniden kurmaya gereken metadata, seçilen stratejiye göre kalıcı depolamaya aktarılır.

Bu işlem bazen tüm GPU'ları bloke eder; bazen **asenkron checkpointing** kullanılarak modelin tutarlı bir snapshot'ı arka planda yazılabilir.

<svg viewBox="0 0 720 342" role="img" aria-label="Bir training step&#x27;in beş yerdeki kulvar görünümü: depolama, CPU ve host RAM, GPU HBM ve SM, NVLink ve ağ. 1, dataloader CPU tarafında depodan token okur. 2, ileri geçiş GEMM&#x27;leri GPU&#x27;da çalışır. 3, tensor paralel all-reduce NVLink üzerinden yapılır. 4, pipeline aktivasyonları ağı geçer. 5, loss ve geri geçiş GPU&#x27;da çalışır. 6, DP veya FSDP gradyan senkronu ağı geçer. 7, optimizer ağırlıkları GPU&#x27;da günceller ve döngü sonraki adım için ileri geçişe döner. 8, checkpoint periyodik olarak depolamaya yazılır. 2–6. adımlar her microbatch&#x27;te tekrarlanır; 3 ve 6 hesapla kısmen örtüşebilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ts-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ts-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="ts-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir training step: her evre fiziksel olarak nerede gerçekleşir?</text>
<rect x="16" y="44" width="688" height="46" rx="6" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="26" y="71" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Depolama</text>
<rect x="16" y="96" width="688" height="46" rx="6" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="26" y="123" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">CPU + host RAM</text>
<rect x="16" y="148" width="688" height="46" rx="6" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="26" y="175" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU: HBM + SM</text>
<rect x="16" y="200" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.05;stroke:var(--c-success);stroke-width:1.2"/>
<text x="26" y="227" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NVLink (TP)</text>
<rect x="16" y="252" width="688" height="46" rx="6" style="fill:var(--c-warn);fill-opacity:.05;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="26" y="279" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Ağ (PP, DP)</text>
<rect x="131" y="100" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="164.0" y="115" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">oku</text>
<text x="164.0" y="129" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">1 · dataloader</text>
<rect x="203" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">forward</text>
<text x="236.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">2 · GEMM</text>
<rect x="275" y="204" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="308.0" y="219" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">TP</text>
<text x="308.0" y="233" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">3 · all-reduce</text>
<rect x="347" y="256" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.0" y="271" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">PP</text>
<text x="380.0" y="285" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">4 · aktivasyon</text>
<rect x="419" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="452.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">backward</text>
<text x="452.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">5 · + loss</text>
<rect x="491" y="256" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="524.0" y="271" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">DP/FSDP</text>
<text x="524.0" y="285" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">6 · grad senk.</text>
<rect x="563" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="596.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">optimizer</text>
<text x="596.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">7 · güncelle</text>
<rect x="635" y="48" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="668.0" y="63" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">checkpoint</text>
<text x="668.0" y="77" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">8 · periyodik</text>
<path d="M196.0 119.0 H236.0 V150" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M268.0 171.0 H308.0 V202" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M340.0 223.0 H380.0 V254" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M412.0 275.0 H452.0 V192" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M484.0 171.0 H524.0 V254" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M556.0 275.0 H596.0 V192" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M628.0 164 H668.0 V88" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M122 66 H138 V98" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M596.0 192 V310 H236.0 V192" marker-end="url(#ts-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2–6. adımlar her microbatch&#x27;te tekrarlanır; 3 ve 6 hesapla kısmen örtüşebilir. 8. adım her N adımda bir çalışır (senkron/asenkron).</text>
</svg>

### 14.1. Bellek hesabı: 405B model için sayıların kökeni

Yalnızca temel model eğitim durumunu düşünelim:

<svg viewBox="0 0 720 294" role="img" aria-label="405B modelin eğitim durumunun yığılmış çubuğu, parametre başına 16 byte: BF16 ağırlık 2 byte yani 0,81 TB, BF16 gradyan 2 byte yani 0,81 TB, FP32 Adam birinci ve ikinci momentleri 4&#x27;er byte yani 1,62&#x27;şer TB, opsiyonel FP32 master kopya 4 byte yani 1,62 TB; toplam 6,48 TB. GPU başına: TP 8 × PP 16 ile 128 GPU&#x27;ya ideal bölünürse 50,6 GB; optimizer durumu DP 128&#x27;e de bölünürse yaklaşık 12,96 GB. İkisi de 80 GB HBM çizgisinin altındadır, ama aktivasyon, tampon ve fragmentasyon hariçtir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="m4-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">405B modelin eğitim durumu: parametre başına 16 byte</text>
<rect x="17" y="40" width="84" height="54" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<text x="59.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">BF16 ağırlık</text>
<text x="59.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2 B/param</text>
<text x="59.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,81 TB</text>
<rect x="103" y="40" width="84" height="54" rx="2" style="fill:var(--c-success);fill-opacity:.45"/>
<text x="145.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">BF16 gradyan</text>
<text x="145.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2 B/param</text>
<text x="145.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,81 TB</text>
<rect x="189" y="40" width="170" height="54" rx="2" style="fill:var(--c-warn);fill-opacity:.38"/>
<text x="274.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Adam m (FP32)</text>
<text x="274.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="274.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,62 TB</text>
<rect x="361" y="40" width="170" height="54" rx="2" style="fill:var(--c-warn);fill-opacity:.38"/>
<text x="446.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Adam v (FP32)</text>
<text x="446.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="446.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,62 TB</text>
<rect x="533" y="40" width="170" height="54" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45"/>
<text x="618.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">FP32 master (opsiyonel)</text>
<text x="618.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="618.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,62 TB</text>
<text x="704" y="112" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-weight:700">toplam 16 B × 405 milyar = 6,48 TB</text>
<text x="16" y="140" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">GPU başına (aktivasyon ve tampon hariç):</text>
<text x="240" y="171" text-anchor="end" style="fill:var(--c-text);font-size:11.5px">÷ 128 (TP 8 × PP 16)</text>
<rect x="250" y="158" width="238.118" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="494.11764705882354" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">50,6 GB</text>
<text x="240" y="207" text-anchor="end" style="fill:var(--c-text);font-size:11.5px">+ optimizer DP 128&#x27;e bölünür</text>
<rect x="250" y="194" width="60.9882" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="316.98823529411766" y="208" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">≈ 12,96 GB</text>
<line x1="626.471" y1="148" x2="626.471" y2="226" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="626.4705882352941" y="240" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">80 GB HBM</text>
<text x="16" y="266" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">İdealize bölüşüm: AllGather tamponları, katman dengesizliği, aktivasyon ve fragmentasyon üstüne eklenir;</text>
<text x="16" y="282" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">FSDP kesin bir sayı garanti etmez.</text>
</svg>

**Önemli:** Master weights her karma hassasiyetli eğitim reçetesinde ayrı tutulmak zorunda değildir. Ayrıca checkpoint sırasında gradyanların yazılması da zorunlu değildir. Şekil *olası eğitim belleğini* modellemek içindir, gerçek dosya boyutu değildir.

TP=8 ve PP=16, ideal olarak parametreleri 128 GPU boyunca kusursuz paylaşabilseydi, kaba model state belleği GPU başına:

$$
\frac{6.48\ \mathrm{TB}}{128}=50.625\ \mathrm{GB/GPU}
$$

Kopyalanan optimizer state ayrıca DP=128 boyunca shard edilirse bu kaba bellek belirgin azalır; fakat geçici AllGather tamponları, katman dengesizliği, aktivasyonlar ve fragmentasyon bu rakamın üstüne eklenir. **'FSDP tam olarak 12,9 GB garanti eder'** diyemeyiz. Aşağıdaki değer yalnızca alternatif bir idealize örnektir:

- Ağırlık ve gradyanlar TP×PP sonrasında kopyalı: yaklaşık $6,33+6,33$ GB.
- Optimizer master+moment durumu: yaklaşık $37,97$ GB, 128 DP kopyası arasında parçalanırsa $0,30$ GB.
- Kaba toplam: **yaklaşık 12,96 GB/GPU**, *aktivasyonlar ve geçici buffer'lar hariç*.

Gerçek FSDP, gradient shard davranışı ve model durumunun yeniden toplanma zamanı sonucu değiştirir. Asıl ders, **HBM kapasitesi ile ağ iletişiminin birbirine takas edilmesidir**.

### 14.2. Step süresine ilişkin kontrol hesabı

Dense Transformer eğitiminin ilk mertebe toplam FLOP tahmini çoğu zaman $6PT$ alınır. $P$ parametre sayısı, $T$ işlenen token sayısıdır; attention ve diğer ayrıntılar sapma yaratır.

$$
F_{\mathrm{step}}\approx6\times405\times10^9\times16\times10^6
=3.888\times10^{19}\ \mathrm{FLOP}
$$

Meta raporundaki 16.384 GPU, GPU başına yaklaşık 400 TFLOP/s ve %41 MFU koşulunda toplam etkin model hesaplama hızı yaklaşık $6.55\ \mathrm{EFLOP/s}$ düzeyidir. Yukarıdaki sade modelle:

$$
T_{\mathrm{step}}\approx\frac{38.88\ \mathrm{EFLOP}}{6.55\ \mathrm{EFLOP/s}}
\approx5.9\ \mathrm{saniye}
$$

Bu, 'Meta her step'i tam 5,9 saniyede tamamladı' iddiası değildir. Bildirilen throughput ile $6PT$ varsayımının tutarlılığını görmek için bir **yaklaşık mühendislik kontrolüdür**. [K5]

---


### 14.3. Veri hattının görünmeyen tarafı: Kalite, köken ve tekrarlanabilirlik

Eğitim adımını anlattığımızda token'ların depodan okunduğunu gördük. Ama bir AI Architect için soru bir adım önce başlar: **Bu token'lar nereden geldi, nasıl temizlendi, hangi sürümle eğitime girdi ve aynı eğitimi altı ay sonra tekrar edebilir miyiz?**

Bir fabrikanın üretim hattı çok hızlı olsun; fakat hammadde yanlış etiketlenmiş veya kirlenmiş olsun. Fabrikanın saatte ürettiği ürün sayısı etkileyici görünür ama üretilen ürünün kalitesi düşük olabilir. Yüksek MFU ile kalitesiz eğitim verisinin ilişkisi de tam olarak böyledir.

Üretim kalitesinde bir **data pipeline** tipik olarak şu adımları düşünmelidir:

<svg viewBox="0 0 720 182" role="img" aria-label="Altı aşamalı üretim veri hattı: ham metin içeren kaynak kayıtlar; lisans ve izin için erişim ve hak kontrolleri; kalite filtreleme ve anlamsal deduplication; packing&#x27;li tokenizer; manifest&#x27;li shard&#x27;lar; ve dataloader üzerinden GPU eğitimi. Her aşamanın altında bırakması gereken kayıt yazar: kaynak sistem, hak kararı, filtre sürümü, tokenizer ve sözlük, shard hash&#x27;leri, seed, commit ve container imajı. Bunlar birlikte, aynı koşunun aylar sonra yeniden kurulmasını sağlayan lineage&#x27;ı oluşturur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dpp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Üretim kalitesinde veri hattı ve her aşamanın bırakması gereken iz</text>
<rect x="16" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="69.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Kaynak kayıtlar</text>
<text x="69.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">ham metin</text>
<rect x="132" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="185.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Erişim &amp; haklar</text>
<text x="185.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">lisans, izin</text>
<line x1="122" y1="69" x2="131" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="248" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="301.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Kalite &amp; dedup</text>
<text x="301.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">filtre, anlamsal</text>
<line x1="238" y1="69" x2="247" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="417.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Tokenizer</text>
<text x="417.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">packing</text>
<line x1="354" y1="69" x2="363" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="480" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="533.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Shard&#x27;lar</text>
<text x="533.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">manifest</text>
<line x1="470" y1="69" x2="479" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="596" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="649.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU eğitimi</text>
<text x="649.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">dataloader</text>
<line x1="586" y1="69" x2="595" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="69" y1="96" x2="69" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="20" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="69.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">kaynak sistem</text>
<line x1="185" y1="96" x2="185" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="136" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="185.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">hak kararı</text>
<line x1="301" y1="96" x2="301" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="252" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="301.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">filtre sürümü</text>
<line x1="417" y1="96" x2="417" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="368" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="417.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">tokenizer + sözlük</text>
<line x1="533" y1="96" x2="533" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="484" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="533.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">shard hash&#x27;leri</text>
<line x1="649" y1="96" x2="649" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="600" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="649.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">seed, kod, imaj</text>
<path d="M16 152 H704" style="fill:none;stroke:var(--c-success);stroke-width:1.6"/>
<text x="360" y="170" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">lineage: hangi bayt, hangi dönüşüm, hangi kod → aynı koşu aylar sonra yeniden kurulabilir</text>
</svg>

**Provenance (veri kökeni)**, bir verinin kaynak sistemini, nasıl toplandığını ve hangi işlemlerden geçtiğini izleyebilme becerisidir. **Lineage (veri soy ağacı)** ise bir eğitim çıktısının hangi veri sürümlerinden ve dönüşümlerden oluştuğunun takibidir.

**Deduplication**, tekrarlanan veya aşırı benzer kayıtların azaltılmasıdır. Neden gereklidir? Aynı belge milyonlarca kez gelirse model belirli örneklere gereğinden fazla ağırlık verebilir, veri dağılımı bozulabilir ve değerlendirme verisiyle eğitim verisi arasında gizli örtüşme (*contamination*) meydana gelebilir. Meta'nın Llama 3 veri hazırlama açıklamalarında heuristic filtreler, anlamsal deduplication ve kalite sınıflandırıcıları kullanılması bu problemin gerçek ölçeğini gösteriyor. [K28]

**Determinism / reproducibility**, eğitim sürecinin tekrarlandığında beklenen sonuçlara yakın biçimde yeniden yürütülebilmesidir. Her GPU kernel'i ve bütün dağıtık yürütmeler için bit düzeyinde özdeşlik garanti etmek her zaman pratik değildir. Yine de şu bilgileri saklamak gerekir:

| Kaydedilecek bilgi | Niçin gerekli? |
|---|---|
| Dataset manifest ve shard hash'leri | Hangi veri baytları kullanıldı? |
| Tokenizer sürümü ve sözlüğü | Aynı metin neden farklı token'lara bölünmesin? |
| Train/validation ayrımı ve sampling seed | Karşılaştırma dürüst olsun |
| Model mimarisi, optimizer, precision, global batch | Eğitim yapılandırması tekrarlanabilsin |
| Kod commit'i, container digest'i, CUDA/NCCL sürümleri | Yazılım ortamı sabitlensin |
| Checkpoint ID ve değerlendirme raporu | Model çıktısı geriye doğru izlenebilsin |

NVIDIA Megatron Bridge'in `.bin/.idx` tarzı indexed dataset ve sürümlü training recipe yaklaşımı bu ihtiyaçların uygulamadaki örneklerinden biridir. Ancak hangi veri formatının kullanılacağı iş yüküne, veri boyutuna ve sürümlere göre seçilmelidir. [K29]

> **Mimari karar:** Veri kümesi için aynı titizliği göstermeden GPU MFU'sunu optimize etmek, yanlış ham maddeyle fabrikayı hızlandırmaya benzer. **Quality-adjusted throughput**, çıplak token/s'den daha değerlidir.

### 14.4. Scaling laws: Kaç parametre ve kaç token eğitmeliyiz?

GPU sayısına karar vermeden önce çok daha temel bir karar veririz: Modelin büyüklüğü ne olmalı, kaç token üzerinde eğitilmeli ve hangi doğruluk düzeyi hedeflenmeli?

Bunu kaynak dağıtımı problemi olarak düşünün. Aynı bütçeyle daha büyük bir modelin daha az veri görmesini mi istersiniz, yoksa daha küçük modelin daha çok veriyle eğitim almasını mı? Bu soruya cevap arayan çalışmalardan biri **Chinchilla / compute-optimal training** yaklaşımıdır. Çalışma, belirli varsayımlarda model büyüklüğü ve veri miktarını dengeli artırmanın toplam hesaplama bütçesini daha etkili kullanabileceğini gösterdi. Bu, her model için evrensel bir parametre/token oranı anlamına gelmez. [K30]

Dense Transformer pretraining için çok kullanılan ilk mertebe hesaplama kestirimi şöyledir:

$$
C_{\mathrm{train}}\approx6ND
$$

- $N$: Parametre sayısı.
- $D$: Toplam eğitim token sayısı.
- $C_{\mathrm{train}}$: Yaklaşık training FLOP miktarı.

Örneğin 70B parametre ve 1T token için:

$$
C\approx6\times70\times10^9\times10^{12}
=4.2\times10^{23}\ \mathrm{FLOP}
$$

Bu, **teorik ilk yaklaşım**dır. Attention maliyeti, sequence length, embedding, activation recomputation, padding, MoE aktif uzmanlar ve diğer teknikler gerçek FLOP hesabını değiştirir. Ayrıca modelin ne kadar iyi öğreneceği bu formülden çıkmaz.

Şimdi bunu donanımla birleştirebiliriz:

$$
T_{\mathrm{idealized}}\approx\frac{C_{\mathrm{train}}}
{N_{\mathrm{GPU}}\times F_{\mathrm{peak}}\times\mathrm{MFU}}
$$

Bu bir **compute-only tahminidir**. Gerçek takvim süresine veri hazırlama, değerlendirme, checkpoint, kesintiler, kapasite bulma ve yeniden başlatma gibi yükler eklenir. Üstelik MFU değeri tüm eğitim boyunca sabit kalmayabilir.

**Neden önemli?** Mimari ekip GPU sayısına bakıp eğitim süresi vaadi vermeden önce hedef kaliteyi, veri sayısını ve model tasarımını bilmelidir. Yoksa sonradan "modeli büyüttük ama veri yetmedi" gibi pahalı sürprizler yaşar.

### 14.5. Hızlı eğitim mi, doğru eğitim mi? Mixed precision ve convergence

Bir modelin belirli loss seviyesine 10 gün yerine 7 günde ulaştığını düşünelim. Bu güçlü bir kazanımdır. Fakat aynı model daha sık sayısal taşma (*overflow*) yaşıyor ve sonunda daha kötü kalite veriyorsa başarı ölçümü ne olmalıdır?

**Convergence (yakınsama)**, eğitim ilerledikçe modelin belirli öğrenme hedefine yaklaşmasıdır. **Validation loss** ve göreve özel değerlendirme metrikleri, yalnızca eğitim throughput'unun değil model davranışının da izlenmesini sağlar.

FP8, BF16 ve karma hassasiyetli eğitimde şu konulara bakılır:

- **Scaling / calibration:** Düşük bit aralığına hangi değerler nasıl eşleniyor?
- **Overflow / underflow:** Çok büyük ya da çok küçük sayılar temsil sınırlarını aşıyor mu?
- **Gradient norm ve loss spikes:** Sayısal kararsızlık eğitimi etkiliyor mu?
- **Per-layer exclusions:** Bazı katmanların daha yüksek hassasiyette tutulması gerekiyor mu?
- **Time-to-quality:** Belirli kalite eşiğine ulaşmak için kaç saat ve kaç GPU-saati harcandı?

Bu yüzden kıyaslama yaparken yalnızca `tokens/s` değil, **aynı veri, aynı hedef kalite, karşılaştırılabilir hyperparameter ve yeterli tekrarla tamamlanan süre** önemlidir. MLPerf Training kurallarında kalite hedefi ve çalışma zamanının birlikte ele alınmasının sebebi de budur. [K31]

---



---

### Bir training step'i muhasebe defteri gibi okumak

Eğitim çoğu zaman “GPU matrisleri çarpıyor” diye özetleniyor. Oysa bir adımın sonunda yalnızca hesaplama yapılmış olmaz; **veri okunmuş**, **aktivasyon tutulmuş**, **gradyan üretilmiş**, **parametre paylaşılmış**, **optimizer state güncellenmiş** ve bazen **checkpoint yazılmış** olur.

Şöyle bir zihinsel defter tutmak, sistemin gerçek maliyetini anlamamızı sağlar:

| Kaynak | Forward sırasında | Backward sırasında | Optimizer sırasında |
|---|---|---|---|
| Tensor Core | Attention / MLP hesaplaması | Türev GEMM'leri | Genellikle daha az önemlidir |
| HBM | Ağırlık ve aktivasyon okuma/yazma | Gradyan ve aktivasyon işlemleri | Parametre ve optimizer durum erişimi |
| Ağ | TP/EP/PP transferleri, FSDP parametre getirme | TP/DP/FSDP iletişimi | Stratejiye göre ek durum paylaşımı |
| Depolama | Veri besleme; ara sıra restore | Genellikle dolaylı | Ara sıra checkpoint |

Bu tablo, **neden bir optimizasyonun başka darboğazı büyütebileceğini** görünür kılar. Aktivasyon recomputation HBM tüketimini azaltır, ama fazladan hesaplama getirir. FSDP bellekte tutulan kopyaları azaltır, ama parametre AllGather maliyeti ekleyebilir. Daha küçük microbatch bellek için iyidir, fakat GEMM verimliliğini düşürebilir. [K48]

### Parametre boyutu, eğitim durumunun tamamı değildir

70B parametreli bir model BF16 ağırlıklarla yaklaşık **140 GB** tutar. Fakat pretraining sırasında bu yalnızca başlangıçtır. BF16 gradyanlar yaklaşık 140 GB daha tutabilir; iki FP32 Adam moment tensörü yaklaşık 560 GB eder. FP32 master weights kopyası tutulursa yaklaşık 280 GB daha eklenir. Yalnızca bu durumların toplamı **1,12 TB** eder; aktivasyonlar, CUDA tamponları ve iletişim buffer'ları dahil değildir.

Bu sayı, modelin her GPU'da bu kadar alan kullanacağı anlamına gelmez. TP, PP, DP ve özellikle FSDP/ZeRO ile bu durumların farklı parçaları GPU'lara dağıtılır. Önemli sezgi: *Parametre sayısı modelin büyüklüğünü anlatır, GPU başına eğitim belleği ise nasıl parçalandığına bağlıdır.*

### "6ND FLOP" yaklaşımını doğru anlamak

Bölüm 14.4'te türettiğimiz $6ND$ kuralı, her token için forward geçişinde yaklaşık $2N$, backward geçişinde ise girdi ve ağırlık gradyanları için $4N$ işlem yapıldığı ilk mertebe modeline dayanır. Ancak bu formül bir doğa kanunu değil, idealize edilmiş bir alt sınırdır.

Gerçek üretim iş yüklerinde bu kestirimden üç ana mekanizma nedeniyle sapılır:
1. **Uzun bağlam karesel maliyeti ($O(S^2)$):** Standart attention katmanlarında sekans uzunluğu uzadıkça attention FLOP'ları karesel büyür; 32k ve üzerindeki pencerelerde $6ND$ formülünün ihmal ettiği terim toplam hesaba baskın gelmeye başlar.
2. **Aktivasyon recomputation cezası:** HBM tasarrufu sağlamak için forward aktivasyonları atılıp backward sırasında yeniden hesaplandığında (selective recomputation), toplam FLOP bütçesi yaklaşık $+2ND$ artarak $8ND$ mertebesine yükselir.
3. **Sparse MoE yönlendirmesi:** Mixture-of-Experts mimarilerinde her token tüm parametreleri değil, yalnızca seçilen uzmanları ($N_{\mathrm{active}}$) çalıştırdığı için kaba parametre sayısı yerine aktif parametre hacmi esas alınmalıdır. [K30]


## 15. Dağıtık eğitimin maliyeti nasıl oluşur? Hesap, veri, optimizer ve kalite

Bir modeli binlerce GPU'ya bölmek, bir bina inşa ederken işçileri çoğaltmak gibidir: Görev planı iyi değilse aynı koridora on ekip sıkışır. Training mimarisinde dört soruyu aynı anda çözmeliyiz: Model belleğe sığıyor mu? Her GPU'da yeterli hesap var mı? İletişim hesaplamayı bloke ediyor mu? Modelin öğrenme davranışı değişti mi?

### 15.1. Global batch nasıl oluşur?

Veri paralel grup büyüklüğü $D$, her GPU pipeline'a giren micro-batch büyüklüğü $m$, optimizer güncellemesi öncesi accumulation adımı $a$ olsun. Basit veri paralel eğitimde:

$$
B_{\mathrm{global}}=D\times m\times a
$$

Örnek: $D=16$, $m=2$, $a=8$ ise optimizer güncellemesi başına 256 örnek işlenir. Ancak LLM eğitiminde örneklerin token uzunluğu önemli olduğundan token bazında da **global tokens per update** tutulmalıdır. Sabit-length packing yapılmıyorsa eşit örnek sayısı, eşit token sayısı demek değildir.

### 15.2. Transformer training FLOPs nasıl kestirilir?

Yoğun decoder-only modeller için sık kullanılan kaba yaklaşım:

$$
F_{\mathrm{train}}\approx 6ND
$$

$N$ parametre sayısı, $D$ görülen eğitim token sayısıdır. Buradaki "6", her parametrenin her token üzerinde yaklaşık forward ve backward maliyetini temsil eden ilk mertebe yaklaşımdır. Attention'ın uzun dizi maliyeti, embedding, activation recomputation, optimizer ve farklı mimariler için düzeltme gerekebilir. MoE için toplam parametreyi doğrudan $N$ almak yanıltıcı olabilir; **aktif parametreler ve gerçekten çalıştırılan expert hesapları** dikkate alınmalıdır.

1B parametre ve 10B token için kaba hesap $6\times10^9\times10^{10}=6\times10^{19}$ FLOP'tur. Bu, bir training planının ilk satırıdır; gerçek runtime ve donanım verimi eklendiğinde süre tahminine dönüşür:

$$
T_{\mathrm{idealized}}\approx\frac{F_{\mathrm{train}}}{N_{\mathrm{GPU}} C_{\mathrm{peak}} \cdot MFU}
$$

Bu formülü kullanırken **Cpeak'in aynı precision/MFU tanımıyla** eşleşmesi şarttır. Ayrıca checkpoint, değerlendirme, veri ön işleme, arıza ve model warmup süreleri toplam iş takvimine ayrıca eklenmelidir.

### 15.3. Memory budgeting eğitim için nasıl yapılır?

FP16/BF16 ağırlıklar, gradyan, optimizer momentleri, varsa FP32 master kopya, activations, temporary buffers, communication buckets ve framework allocator rezervi ayrı kalemlerdir. Dağıtılmış stratejiler bunların farklı oranlarını shard eder. Bir optimizer'ın hangi durumda FP32 master weight tuttuğunu kesin varsaymak yerine, seçili framework ve precision reçetesini kontrol edin.

**Pratik yöntemi:** Önce en uzun seq length ve hedef micro-batch ile **tek model-paralel grup** üzerinde bellek profilini çıkarın. Sonra DP ölçeğine geçin. Büyük DP grubunun birçok modeli dağıttığını görüp bir GPU'daki aktivasyon belleğinin küçüldüğünü otomatik varsaymayın: Aktivasyonlar çoğu durumda local micro-batch ve sequence length ile ilişkilidir.

### 15.4. Attention maliyeti nasıl düşünülmeli?

Standart full-attention'da sequence length $S$ büyüdükçe attention matrisinin boyutuyla ilişkili işlem büyümesi yaklaşık $O(S^2)$ olur. FlashAttention attention matrisinin tümünü HBM'de materyalize etmeyi önleyerek I/O yükünü azaltır; ama modelin temel attention bağımlılıklarını sihirli biçimde lineer hale getirmez.

Context Parallelism (CP), uzun dizileri farklı GPU'lara dağıtmak için kullanılabilir; ancak KV paylaşımı ve iletişim stratejisi maliyet yaratır. Bu yüzden "context window iki katına çıktı" ile "eğitim maliyeti iki katına çıktı" arasında otomatik bir eşitlik yoktur.

### 15.5. Step süresini fazlara ayırın

<svg viewBox="0 0 720 252" role="img" aria-label="Step süresi evrelerinin zaman çizelgesi. Kritik yolda: forward, backward, açıkta kalan iletişim dilimi, optimizer ve sonraki adım. Altında arka plan işleri: gradyan AllReduce bucket&#x27;ları backward&#x27;ın son kısmıyla örtüşür, veri prefetch hesap sürerken sonraki batch&#x27;i yükler, asenkron checkpoint kısa bir snapshot duraklamasından sonra durumu kalıcı yazar. Backward&#x27;ın altına sığmayan iletişim kritik yola düşer." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="sp-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Step süresinin evreleri: kritik yol ve yanına gizlenebilenler</text>
<text x="16" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">kritik yol</text>
<rect x="250" y="46" width="135" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.5" y="64.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">forward</text>
<rect x="385" y="46" width="180" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="475.0" y="64.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">backward</text>
<rect x="565" y="46" width="45" height="28" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="587.5" y="63.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">açıkta</text>
<rect x="610" y="46" width="45" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="632.5" y="63.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">optim.</text>
<rect x="655" y="46" width="49.5" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="679.75" y="63.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">sonraki</text>
<text x="16" y="103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">grad AllReduce</text>
<text x="16" y="118" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">bucket&#x27;lar geç backward&#x27;la örtüşür</text>
<rect x="475" y="92" width="135" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="16" y="147" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">veri prefetch</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">hesap sürerken sonraki batch</text>
<rect x="272.5" y="136" width="180" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="16" y="191" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">async checkpoint</text>
<text x="16" y="206" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">kısa snapshot sonrası kalıcı yazma</text>
<rect x="637" y="180" width="67.5" height="24" rx="4" style="fill:var(--c-danger);fill-opacity:.22;stroke:var(--c-danger);stroke-width:1.2"/>
<line x1="565" y1="40" x2="565" y2="226" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<text x="704" y="240" text-anchor="end" style="fill:var(--c-danger);font-size:11px">backward&#x27;ın altına sığmayan iletişim kritik yola düşer</text>
</svg>

Bu çizimi gerçek profiler timeline'ıyla yan yana koyun. GPU'ların çoğu backward sonrası collective bekliyorsa network veya overlap tasarımını deneyin. Uzun GPU kernel boşlukları dataloader ile çakışıyorsa veri yoluna gidin. Ama mümkün olan her şeyi overlap etmek otomatik fayda getirmez: bellek rezervasyonu, CUDA stream öncelikleri ve sürücü kaynak rekabeti de maliyet yaratır.

### 15.6. Kayıp fonksiyonu düşüyor mu? Hızlanmanın bilimsel kontrolü

MFU kazandıktan sonra modelin öğrenmesi bozulduysa ticari olarak kazanmadınız. Precision ve optimizer değiştirdiğiniz her deneyde **loss eğrisi**, gradient norm, NaN/Inf, clip sayısı, öğrenme oranı, validation sonuçları ve seçilen görevlerde kalite ölçülmelidir.

BF16→FP8 geçişi ile saniyede daha fazla token işleyebilirsiniz, ama bazı katmanlar daha yüksek hassasiyet gerektirebilir. Activation checkpointing, optimizer/offload stratejisi veya fused kernel değişimi sonrasında yalnızca throughput değil **time-to-quality** raporlanmalıdır. [K31][K41]

### 15.7. LoRA ve PEFT mimarisi ne zaman ayrı tasarım gerektirir?

Tam fine-tuning yerine **LoRA (Low-Rank Adaptation)**, büyük ağırlıkları sabit tutup eğitilebilir düşük rank adapter'lar ekleyebilir. Bu yaklaşım eğitilen parametre sayısını ve optimizer durumunu azaltır; fakat forward için ana model ağırlıkları yine gerekir. Çok tenant'lı LoRA serving'de adapter'ların yüklenmesi, versiyonlanması, cache politikası ve erişim izolasyonu da yeni tasarım boyutlarıdır.

**Karar ölçütü:** Full fine-tuning kalite ihtiyacı mı var, yoksa PEFT ile yeterli kalite daha düşük toplam maliyetle sağlanıyor mu? Bu sorunun yanıtı modelin büyüklüğünden değil **görev benchmark'ından** çıkar.

---




---

**KISIM V — Veri, checkpoint ve güvenilirlik**

*Hız kadar süreklilik neden önemli? Bir kayıt işlemi ve bir arıza, bütün kümenin hesabını nasıl değiştirir?*


## 16. Depolama duvarı: GPU'nuz hesaplamıyor, dosya sistemi bekliyor olabilir

Bir mühendis olarak yalnızca GPU hesaplamasına bakarsanız çok pahalı bir darboğazı kaçırabilirsiniz: **GPU'ya veri getirme ve eğitim durumunu güvenilir biçimde saklama süreci**.

Veri okumak ile checkpoint yazmak birbirinin simetriği değildir. Eğitim verileri sürekli küçük veya orta büyüklükte parçalar halinde gelebilir. Checkpoint ise binlerce GPU'nun neredeyse aynı anda büyük dosyalar yazdığı ani bir dalga yaratabilir.

Bir şehirde her gün düzenli dağıtım yapmakla, bütün mahallelerin aynı anda taşınmaya çalışması arasında nasıl fark varsa, depolama sisteminde de fark budur.

### 16.1. Okuma ile yazma neden farklı sorunlar doğurur?

Pretraining verisini okurken veriler görece küçük parçalar halinde çok sık istenir. Checkpoint yazarken ise kısa bir zaman aralığında çok büyük miktarda durum kalıcı depolamaya gönderilir.

Bu yüzden aynı dosya sisteminde iki farklı gereksinim vardır:

1. **Steady-state read:** Düzenli, dengeli ve gecikmesiz token/örnek akışı.
2. **Burst write:** Birden çok node'un aynı anda terabaytlarca veriyi güvenilir yazabilmesi.

### 16.2. Linux Page Cache neden bazen iyi, bazen yetersiz?

Linux sistemlerde dosyadan veri okuduğumuzda işletim sistemi bu veriyi RAM'de tutarak gelecekteki erişimleri hızlandırabilir. Bu aslında çoğu uygulama için harika bir optimizasyondur. Ama GPU'ya sürekli büyük veri aktarıyorsak veriyi CPU RAM'e kopyalayıp sonra HBM'e yeniden taşımak fazladan bir adım oluşturabilir.

Öyleyse Page Cache kötü mü? Hayır. Bir optimizasyonun faydası, veri erişim şekline bağlıdır. Sorun sistemin varlığı değil, o yolu seçtiğimizde hangi kopyalamaların yapıldığıdır.

**Page Cache**, dosya içeriğini CPU sistem RAM'inde önbelleğe alır. Tekrarlanan dosya okumalarında disk I/O miktarını azaltabilir. Büyük GPU merkezli transferlerde ise dosyadaki veriyi CPU RAM'e alıp sonra GPU HBM'e kopyalamak gereksiz ek bellek trafiği yaratabilir.

Dolayısıyla Page Cache **özünde performans düşmanı değildir**. Kullanım biçimi ve çalışma kümesinin boyutu belirleyicidir.

Klasik yolda bir byte GPU'ya ulaşana kadar genellikle dört durak yapar: (1) NVMe'den PCIe üzerinden host'a, (2) çekirdeğin page cache'ine, (3) uygulamanın user-space tamponuna (bounce buffer) ve (4) yeniden PCIe üzerinden H2D kopyayla HBM'e. Veri PCIe'yi iki kez geçer ve CPU belleğinde iki kopya oluşur. Birkaç büyük okumada bu zararsızdır; terabaytlarca checkpoint'te ise CPU'nun bellek veri yolunu ve çekirdeklerini meşgul eder. Bir sonraki bölümdeki GDS tam olarak 2. ve 3. durakları atlamayı hedefler. [K6]

### 16.3. mmap, GIL ve NUMA'yı doğru yere koyalım

**mmap**, bir dosyayı sanal adres uzayına eşler. Böylece dosyanın belirli aralıklarına bellek erişimi gibi ulaşılabilir; talep üzerine sayfa yüklemeleri yaşanabilir.

**Python GIL**, standart CPython yorumlayıcısında Python bytecode yürütmesine ilişkin eşzamanlılık sınırlamasıdır. `mmap` tek başına GIL sorununu çözmez; I/O işlem yapısını kolaylaştırabilir. Çoklu process DataLoader, C/C++ kodu, asenkron I/O ve batching ayrı tasarım araçlarıdır.

**NUMA-aware placement**, DataLoader worker'larını, bellek tahsisini ve NIC yakınlığını uygun CPU soketi etrafında düzenleyerek gereksiz uzak erişim maliyetini azaltmayı amaçlar.

Üretim dataloader'ları bu üçünü birlikte kullanır. Megatron tarzı çerçeveler veri setini bir kez `uint16`/`int32` token ID'leri hâlinde tek bir bitişik `.bin` dosyasına, belge sınırlarını ayrı bir `.idx` dosyasına yazar; worker'lar dosyayı `mmap` ile açar, böylece eğitim sırasında metin ayrıştırma ve Python nesnesi üretme yapılmaz [K29]. DGX H100'de CPU 0 GPU 0–3'ü, CPU 1 GPU 4–7'yi besler; GPU $i$'yi besleyen worker'ları aynı soketteki çekirdeklere sabitlemek ve pinned memory kullanmak, verinin UPI üzerinden karşı sokete gidip gelmesini önler.

### 16.4. GPUDirect Storage (GDS) gerçekte hangi yolu değiştirir?

Depodan ürünü önce ana ofise indirip ardından üretim bandına göndermek yerine doğrudan üretim bandına ulaştırdığınızı düşünün. Aradaki taşıma ve kopyalama işini azaltmış olursunuz.

GPUDirect Storage'ın büyük fikri buna benzer. Desteklenen sistemlerde depolama verisinin GPU belleğine CPU RAM üzerinde gereksiz ara tamponlar oluşturulmadan taşınmasını sağlar. Ama “CPU artık hiçbir şeye karışmıyor” demek doğru değil: kontrol, sürücüler ve I/O yönetimi hâlâ sistemin parçasıdır.

<svg viewBox="0 0 720 308" role="img" aria-label="Depolamadan GPU HBM&#x27;e iki veri yolu. Klasik: veri depodan CPU RAM&#x27;deki bounce buffer&#x27;a kopyalanır, sonra PCIe üzerinden yeniden HBM&#x27;e kopyalanır. GPUDirect Storage: veri cuFile ile PCIe veya NVMe-oF üzerinden DMA&#x27;yla doğrudan HBM&#x27;e gider; CPU yalnızca kontrolü yürütür. GDS bir kopyayı kaldırır, depolama sınırını değil: 25 GB/s&#x27;lik hedef 25 GB/s kalır; compatibility mode CPU yoluna geri dönebilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="gd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="gd-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Klasik bounce buffer yolu ve GPUDirect Storage</text>
<text x="16" y="56" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">1 · klasik: CPU RAM üzerinden kopya</text>
<rect x="16" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Depolama</text>
<text x="86.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVMe / paralel FS</text>
<rect x="564" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="634.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM3</text>
<text x="634.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="290" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU RAM</text>
<text x="360.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bounce buffer</text>
<line x1="156" y1="92" x2="288" y2="92" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="430" y1="92" x2="562" y2="92" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="222" y="86" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kopya 1</text>
<text x="496" y="86" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kopya 2 (PCIe)</text>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">2 · GDS: HBM&#x27;e doğrudan DMA</text>
<rect x="16" y="182" width="140" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="204.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Depolama</text>
<text x="86.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVMe / paralel FS</text>
<rect x="564" y="182" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="634.0" y="204.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM3</text>
<text x="634.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<line x1="156" y1="206" x2="562" y2="206" marker-end="url(#gd-o)" style="stroke:var(--c-success);stroke-width:2"/>
<text x="360" y="200" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">PCIe / NVMe-oF üzerinden DMA (cuFile)</text>
<rect x="290" y="240" width="140" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="360.0" y="261.75" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">CPU: yalnızca kontrol</text>
<line x1="360" y1="240" x2="360" y2="212" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="16" y="296" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GDS bir kopyayı kaldırır, depolama sınırını değil: 25 GB/s&#x27;lik hedef 25 GB/s kalır. Compatibility mode CPU yoluna geri dönebilir.</text>
</svg>

**GPUDirect Storage**, uygun donanım, dosya sistemi, sürücü ve cuFile API koşullarında GPU belleği ile depolama arasında doğrudan DMA veri yolu sağlar. Bu, CPU bounce buffer kopyalarını azaltabilir. Ancak her I/O isteği zorunlu olarak doğrudan gitmez; **compatibility mode** CPU üzerinden veri aktarımına geri dönebilir. [K6]

Bu nedenle şu üç iddiayı ayırmalıyız:

- **Doğru:** GDS bazı iş yüklerinde CPU kopyalarını azaltabilir.
- **Koşullu:** Bu sayede sistem I/O throughput'u ve gecikmesi iyileşebilir.
- **Yanlış genelleme:** GDS her checkpoint'i 10 kat hızlandırır; CPU kullanımı sıfır olur; 7,5 TB her zaman bir saniyede diske yazılır.

GDS **veri yolu optimizasyonudur**. Depolama hedefi 25 GB/s ile sınırlıysa GPU'ya doğrudan aktarım yapmak dosya sisteminin bu sınırını tek başına 250 GB/s'ye çıkarmaz.

### 16.5. 7,5 TB snapshot örneği

7,5 TB büyüklüğünde **varsayımsal** bir checkpoint olsun. Decimal birimlerle:

$$
T_{\mathrm{write}}\approx\frac{S_{\mathrm{checkpoint}}}{W_{\mathrm{storage}}}
$$

| Küme düzeyindeki etkin yazma hızı | Basit yazma süresi |
|---:|---:|
| 25 GB/s | 300 saniye = 5 dakika |
| 80 GB/s | 93,75 saniye |
| 250 GB/s | 30 saniye |
| 2 TB/s | 3,75 saniye, *ideal bölme hesabı* |

Son satırda **2 TB/s üretim dosya sistemi kapasitesi** olsa bile gerçek checkpoint süresinin tam 3,75 saniye olacağı kesin değildir. Metadata, fsync/durability, ağ paylaşımı, dosya sayısı, dağılım ve darboğazlar sonucu değiştirir.

**Tectonic örneği:** Meta'nın Llama 3 raporunda Tectonic altyapısı 240 PB depolama, 7.500 SSD sunucusu, **2 TB/s sürdürülebilir** ve **7 TB/s tepe throughput** olarak açıklanır. Bunlar tek bir checkpoint'in gerçek tamamlanma süresi değil, sistemin raporlanan depolama kapasitesi ve throughput özellikleridir. [K5]

### 16.6. Asenkron checkpoint ile gerçek kazanım

Bir projeyi kaydederken bütün ekibin beş dakika bilgisayar başında beklediğini düşünün. Peki kaydı arka planda yapıp ekip çalışmaya devam edebilseydi? Kaydetmenin toplam süresi aynı kalsa bile işin durduğu süre azalabilirdi.

Asenkron checkpoint'in temel hedefi budur. Fakat bunun karşılığında yeni bir soruya dikkat etmeliyiz: Yazma işlemi henüz tamamlanmamış bir snapshot gerçekten kurtarma için güvenilir midir?

**Synchronous checkpoint**, eğitim adımının kaydetme bitene kadar duraklaması anlamına gelebilir.

**Asynchronous checkpoint**, durumun tutarlı bir snapshot'ını alarak eğitim ile depolama yazma işlemlerini kısmen örtüştürür. Ancak iki ayrı süreyi izlemek gerekir:

- **Snapshot / pause time:** Eğitim için gerçekten bloklayıcı süre.
- **Persistence / durability time:** Dosyaların güvenilir ve tamamlanmış biçimde depoda bulunduğu zamana kadar geçen süre.

Arıza son kaydın depoda tamamlanmasından önce olursa o checkpoint henüz kullanılabilir olmayabilir. Yüksek ölçekli sistemde tutarlılık, atomik metadata, shard tamamlanma kontrolü ve kurtarma testleri kritik önemdedir.

---


## 17. 16.384 GPU'luk sistemde arızalar neden kaçınılmaz görünüyor?

Bir dizüstü bilgisayarın aylarca sorunsuz çalışması olağandır. Fakat binlerce sunucunun bulunduğu bir veri merkezinde, herhangi bir sunucunun o gün arızalanma ihtimali artık küçümsenemez.

Buradaki paradoks ilginçtir: Her bir sunucu oldukça güvenilir olabilir. Ama **tüm kümenin hiçbir bileşeninde arıza yaşanmaması** çok daha zor bir hedeftir. Bu nedenle büyük model eğitiminde güvenilirlik, “arızaları tamamen ortadan kaldırma” kadar, “arızayı hızla algılayıp işi devam ettirme” problemidir.

### 17.1. MTBF: Tek sunucunun güvenilirliği kümenin güvenilirliği değildir

**MTBF (Mean Time Between Failures)**, onarılabilir bir sistemde arızalar arasında geçen ortalama süreyi ifade eder. Tek node için 26.000 saat varsayalım. Bir kümede 2.048 node varsa ve arızalar *bağımsız, sabit oranlı* kabul edilebiliyorsa:

$$
M_{\mathrm{cluster}}\approx\frac{26000}{2048}=12.7\ \mathrm{saat}
$$

Bu **gerçek sistemin ölçülmüş MTBF değeri değildir**; bağımsız arıza modelinin öğretici sonucudur. Ağ kesintisi, ortak enerji problemi, yazılım bug'ı, termal nedenler ve toplu bakım olayları bu basitleştirmeyi bozar.

### 17.2. Gerçek olay: Llama 3 405B'de 54 gün

Bu fikrin yalnızca matematiksel bir varsayım olmadığını gerçek eğitim raporlarından görebiliriz. Meta'nın Llama 3 405B raporundaki duraklama istatistikleri, büyük bir AI kümesinde otomatik kurtarma sistemlerinin neden temel altyapı bileşeni olduğunu gösteriyor.

Meta'nın yayınladığı teknik raporda, 54 günlük pretraining gözlem penceresinde:

- **466 toplam job interruption**.
- **47 planlı duraklama** (bakım ve operasyonel değişiklikler).
- **419 beklenmedik duraklama**.
- Beklenmedik duraklamaların yaklaşık **%78'i doğrulanmış veya şüpheli donanım sorunlarıyla** ilişkilendirilmiş.
- Yalnızca **3 olayda önemli manuel müdahale** gerekmiş; geri kalanları otomasyon ele almış.

Bu durumda beklenmedik duraklamalar arası basit ortalama:

$$
\frac{54\times24}{419}\approx3.09\ \mathrm{saat}
$$

Bunu 'her 3 saatte bir fiziksel H100 bozuldu' şeklinde okumamalıyız. Bu sayı **beklenmedik job interruption** oranıdır; nedeni hem donanım hem yazılım/ağ sorunları olabilir. [K5]

<svg viewBox="0 0 720 196" role="img" aria-label="Llama 3 405B pretraining&#x27;inin 54 gününde yaşanan 466 iş kesintisinin yığılmış çubuğu: 47 planlı, 419 beklenmedik; beklenmediklerin yaklaşık %78&#x27;i, kabaca 327&#x27;si doğrulanmış veya şüpheli donanım sorunlarına, 92 kadarı diğer nedenlere bağlandı. Bu, ortalama 3,09 saatte bir beklenmedik kesinti demektir. Yalnızca 3&#x27;ü ciddi manuel müdahale gerektirdi; kalanını otomasyon çözdü." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="li-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Llama 3 405B pretraining: 54 günde 466 iş kesintisi</text>
<rect x="17" y="44" width="67.3906" height="36" rx="2" style="fill:var(--c-text-mute);fill-opacity:.25"/>
<text x="50.69527896995708" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">47 planlı</text>
<rect x="86.3906" y="44" width="480.781" height="36" rx="2" style="fill:var(--c-danger);fill-opacity:.45"/>
<text x="326.78111587982835" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">~327 donanım (beklenmedik olanların ≈%78&#x27;i)</text>
<rect x="569.172" y="44" width="133.828" height="36" rx="2" style="fill:var(--c-warn);fill-opacity:.4"/>
<text x="636.0858369098713" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">~92 diğer</text>
<path d="M86.39055793991416 88 V94 H703 V88" style="fill:none;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="394.6952789699571" y="110" text-anchor="middle" style="fill:var(--c-danger);font-size:12px;font-weight:700">419 beklenmedik</text>
<rect x="16" y="124" width="688" height="60" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="146" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">54 gün × 24 sa ÷ 419 ≈ beklenmedik kesintiler arası 3,09 sa</text>
<text x="28" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca 3&#x27;ü ciddi manuel müdahale gerektirdi; kalanını otomasyon çözdü. “Her 3 saatte bir H100 bozuldu” demek değildir.</text>
</svg>

### 17.3. Young–Daly: Checkpoint neden ne çok sık ne çok seyrek alınmalı?

Diyelim ki bir saat boyunca hiç checkpoint almadık ve eğitim 59. dakikada çöktü. Neredeyse bir saatlik işimiz boşa gitti. Buna karşılık her dakika checkpoint alırsak zamanımızın büyük bölümünü dosya yazmakla geçirebiliriz.

İyi bir kayıt politikası bu iki uç arasında denge kurmalı. Matematik bize yaklaşık optimum noktayı bulmak için oldukça zarif bir yöntem sunuyor.

Bir arıza sonrası son başarılı checkpoint'e geri dönüyorsak iki maliyet var:

- Sık checkpoint: yazma yükü büyür.
- Seyrek checkpoint: arızada yeniden hesaplanacak iş büyür.

Yaklaşık bir kayıp fonksiyonu:

$$
Loss(\tau)\approx\frac{\delta}{\tau}+\frac{\tau}{2M}
$$

Burada:

- $\tau$: checkpoint'ler arası süre.
- $\delta$: eğitimi bloke eden bir checkpoint maliyeti.
- $M$: job'ın ortalama arızalar arası süresi.

Birinci terim yazma kaybı, ikinci terim arıza sonrası ortalama yeniden hesaplama kaybıdır. Türevi sıfıra eşitleyelim:

$$
\frac{dLoss}{d\tau}=-\frac{\delta}{\tau^2}+\frac{1}{2M}=0
$$

$$
\boxed{\tau_{\mathrm{opt}}\approx\sqrt{2\delta M}}
$$

Bu **Young–Daly'nin klasik ilk mertebe yaklaşımı**dır; farklı failure modeli, recovery maliyeti, asenkron snapshot ve sistemin gerçek davranışı düzeltme gerektirir. [K10]

### 17.4. Üç saat MTBF varsayımıyla karşılaştırma

M = 3 saat = 10.800 saniye; checkpoint büyüklüğü 7,5 TB:

| Metrik | 25 GB/s storage | 250 GB/s storage |
|---|---:|---:|
| Bloklayıcı yazma süresi $\delta$ | 300 saniye | 30 saniye |
| Yaklaşık optimum $\tau$ | 42,4 dakika | 13,4 dakika |
| Kayıt overhead'i $\delta/\tau$ | %11,8 | %3,7 |
| Beklenen yeniden hesaplama kaybı $\tau/(2M)$ | %11,8 | %3,7 |
| **Toplam yaklaşık kayıp** | **%23,6** | **%7,5** |

<svg viewBox="0 0 720 326" role="img" aria-label="7,5 TB checkpoint ve 3 saatlik ortalama kesinti aralığı için toplam kaybın checkpoint aralığına göre Young–Daly eğrileri. 25 GB/s depolamada bloklayıcı yazma 300 saniye sürer; eğrinin dibi 42,4 dakikada, %23,6 kayıptadır. 250 GB/s&#x27;de yazma 30 saniye sürer; optimum 13,4 dakikada, %7,5 kayıptadır. Çok kısa aralıklarda yazma, çok uzun aralıklarda arıza sonrası yeniden hesaplama baskındır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="yd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Young–Daly: toplam kayıp W(τ) = δ/τ + τ/2M; M = 3 sa, 7,5 TB checkpoint</text>
<line x1="70" y1="280" x2="690" y2="280" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="284.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%0</text>
<line x1="70" y1="234" x2="690" y2="234" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="238.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%10</text>
<line x1="70" y1="188" x2="690" y2="188" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="192.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%20</text>
<line x1="70" y1="142" x2="690" y2="142" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="146.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%30</text>
<line x1="70" y1="96" x2="690" y2="96" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="100.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%40</text>
<line x1="70" y1="50" x2="690" y2="50" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="54.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%50</text>
<text x="70.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="173.33333333333331" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">15</text>
<text x="276.66666666666663" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">30</text>
<text x="380.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">45</text>
<text x="483.3333333333333" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">60</text>
<text x="586.6666666666667" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">75</text>
<text x="690.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">90</text>
<line x1="70" y1="280" x2="690" y2="280" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="70" y1="50" x2="70" y2="280" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="380.0" y="314" text-anchor="middle" style="fill:var(--c-text);font-size:12px">checkpoint aralığı τ (dakika)</text>
<path d="M144.4 53.2 L145.9 57.3 L147.5 61.2 L149.1 64.9 L150.6 68.5 L152.2 71.9 L153.7 75.2 L155.2 78.3 L156.8 81.4 L158.3 84.3 L159.9 87.1 L161.4 89.8 L163.0 92.4 L164.6 94.9 L166.1 97.3 L167.7 99.6 L169.2 101.9 L170.8 104.0 L172.3 106.1 L173.8 108.2 L175.4 110.1 L176.9 112.0 L178.5 113.8 L180.1 115.6 L181.6 117.3 L183.1 119.0 L184.7 120.6 L186.2 122.1 L187.8 123.6 L189.4 125.1 L190.9 126.5 L192.4 127.9 L194.0 129.2 L195.6 130.5 L197.1 131.8 L198.7 133.0 L200.2 134.2 L201.8 135.3 L203.3 136.4 L204.8 137.5 L206.4 138.5 L207.9 139.6 L209.5 140.5 L211.1 141.5 L212.6 142.4 L214.2 143.3 L215.7 144.2 L217.2 145.1 L218.8 145.9 L220.3 146.7 L221.9 147.5 L223.4 148.3 L225.0 149.0 L226.6 149.8 L228.1 150.5 L229.7 151.1 L231.2 151.8 L232.8 152.5 L234.3 153.1 L235.9 153.7 L237.4 154.3 L238.9 154.9 L240.5 155.4 L242.1 156.0 L243.6 156.5 L245.2 157.1 L246.7 157.6 L248.2 158.0 L249.8 158.5 L251.3 159.0 L252.9 159.4 L254.4 159.9 L256.0 160.3 L257.5 160.7 L259.1 161.1 L260.6 161.5 L262.2 161.9 L263.8 162.3 L265.3 162.6 L266.9 163.0 L268.4 163.3 L270.0 163.7 L271.5 164.0 L273.1 164.3 L274.6 164.6 L276.1 164.9 L277.7 165.2 L279.2 165.5 L280.8 165.7 L282.4 166.0 L283.9 166.3 L285.4 166.5 L287.0 166.7 L288.6 167.0 L290.1 167.2 L291.6 167.4 L293.2 167.6 L294.8 167.8 L296.3 168.0 L297.9 168.2 L299.4 168.4 L300.9 168.6 L302.5 168.7 L304.1 168.9 L305.6 169.0 L307.1 169.2 L308.7 169.3 L310.2 169.5 L311.8 169.6 L313.4 169.8 L314.9 169.9 L316.4 170.0 L318.0 170.1 L319.6 170.2 L321.1 170.3 L322.6 170.4 L324.2 170.5 L325.8 170.6 L327.3 170.7 L328.9 170.8 L330.4 170.9 L331.9 170.9 L333.5 171.0 L335.1 171.1 L336.6 171.1 L338.1 171.2 L339.7 171.2 L341.2 171.3 L342.8 171.3 L344.4 171.4 L345.9 171.4 L347.4 171.4 L349.0 171.5 L350.6 171.5 L352.1 171.5 L353.6 171.5 L355.2 171.5 L356.8 171.6 L358.3 171.6 L359.9 171.6 L361.4 171.6 L362.9 171.6 L364.5 171.6 L366.1 171.6 L367.6 171.6 L369.1 171.5 L370.7 171.5 L372.2 171.5 L373.8 171.5 L375.4 171.5 L376.9 171.4 L378.4 171.4 L380.0 171.4 L381.6 171.4 L383.1 171.3 L384.6 171.3 L386.2 171.2 L387.8 171.2 L389.3 171.2 L390.9 171.1 L392.4 171.1 L393.9 171.0 L395.5 170.9 L397.0 170.9 L398.6 170.8 L400.1 170.8 L401.7 170.7 L403.2 170.6 L404.8 170.6 L406.3 170.5 L407.9 170.4 L409.4 170.4 L411.0 170.3 L412.6 170.2 L414.1 170.1 L415.6 170.0 L417.2 170.0 L418.8 169.9 L420.3 169.8 L421.9 169.7 L423.4 169.6 L424.9 169.5 L426.5 169.4 L428.1 169.3 L429.6 169.2 L431.2 169.1 L432.7 169.0 L434.2 168.9 L435.8 168.8 L437.4 168.7 L438.9 168.6 L440.5 168.5 L442.0 168.4 L443.6 168.3 L445.1 168.2 L446.6 168.1 L448.2 168.0 L449.8 167.8 L451.3 167.7 L452.9 167.6 L454.4 167.5 L455.9 167.4 L457.5 167.2 L459.1 167.1 L460.6 167.0 L462.1 166.9 L463.7 166.7 L465.2 166.6 L466.8 166.5 L468.4 166.3 L469.9 166.2 L471.4 166.1 L473.0 165.9 L474.5 165.8 L476.1 165.7 L477.6 165.5 L479.2 165.4 L480.8 165.2 L482.3 165.1 L483.8 165.0 L485.4 164.8 L486.9 164.7 L488.5 164.5 L490.1 164.4 L491.6 164.2 L493.1 164.1 L494.7 163.9 L496.2 163.8 L497.8 163.6 L499.4 163.5 L500.9 163.3 L502.4 163.1 L504.0 163.0 L505.6 162.8 L507.1 162.7 L508.7 162.5 L510.2 162.4 L511.8 162.2 L513.3 162.0 L514.9 161.9 L516.4 161.7 L518.0 161.5 L519.5 161.4 L521.0 161.2 L522.6 161.0 L524.1 160.9 L525.7 160.7 L527.2 160.5 L528.8 160.4 L530.4 160.2 L531.9 160.0 L533.5 159.8 L535.0 159.7 L536.5 159.5 L538.1 159.3 L539.6 159.2 L541.2 159.0 L542.8 158.8 L544.3 158.6 L545.9 158.4 L547.4 158.3 L549.0 158.1 L550.5 157.9 L552.0 157.7 L553.6 157.5 L555.1 157.4 L556.7 157.2 L558.2 157.0 L559.8 156.8 L561.3 156.6 L562.9 156.4 L564.5 156.2 L566.0 156.1 L567.5 155.9 L569.1 155.7 L570.6 155.5 L572.2 155.3 L573.8 155.1 L575.3 154.9 L576.9 154.7 L578.4 154.5 L580.0 154.3 L581.5 154.1 L583.0 154.0 L584.6 153.8 L586.1 153.6 L587.7 153.4 L589.2 153.2 L590.8 153.0 L592.4 152.8 L593.9 152.6 L595.5 152.4 L597.0 152.2 L598.5 152.0 L600.1 151.8 L601.6 151.6 L603.2 151.4 L604.8 151.2 L606.3 151.0 L607.9 150.8 L609.4 150.6 L611.0 150.4 L612.5 150.2 L614.0 150.0 L615.6 149.8 L617.1 149.6 L618.7 149.3 L620.2 149.1 L621.8 148.9 L623.4 148.7 L624.9 148.5 L626.5 148.3 L628.0 148.1 L629.5 147.9 L631.1 147.7 L632.6 147.5 L634.2 147.3 L635.8 147.1 L637.3 146.8 L638.9 146.6 L640.4 146.4 L642.0 146.2 L643.5 146.0 L645.0 145.8 L646.6 145.6 L648.1 145.4 L649.7 145.1 L651.2 144.9 L652.8 144.7 L654.4 144.5 L655.9 144.3 L657.5 144.1 L659.0 143.8 L660.5 143.6 L662.1 143.4 L663.6 143.2 L665.2 143.0 L666.8 142.8 L668.3 142.5 L669.9 142.3 L671.4 142.1 L673.0 141.9 L674.5 141.7 L676.0 141.4 L677.6 141.2 L679.1 141.0 L680.7 140.8 L682.2 140.6 L683.8 140.3 L685.4 140.1 L686.9 139.9 L688.5 139.7" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<circle cx="362.271" cy="171.577" r="5" style="fill:var(--c-danger);fill-opacity:.9"/>
<line x1="362.271" y1="171.577" x2="362.271" y2="280" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<text x="370.27080289043965" y="193.5769602180627" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">τ* = 42,4 dk → %23,6</text>
<line x1="428.222" y1="110.4" x2="450.222" y2="110.4" style="stroke:var(--c-danger);stroke-width:2.2"/>
<text x="456.2222222222222" y="114.4" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">25 GB/s → δ = 300 sn</text>
<path d="M77.8 74.1 L79.3 107.9 L80.8 132.0 L82.4 149.9 L84.0 163.8 L85.5 174.9 L87.0 183.9 L88.6 191.4 L90.2 197.6 L91.7 203.0 L93.2 207.5 L94.8 211.5 L96.3 215.0 L97.9 218.0 L99.5 220.7 L101.0 223.1 L102.5 225.3 L104.1 227.2 L105.7 228.9 L107.2 230.5 L108.8 231.9 L110.3 233.2 L111.8 234.4 L113.4 235.4 L115.0 236.4 L116.5 237.3 L118.0 238.1 L119.6 238.9 L121.2 239.5 L122.7 240.2 L124.2 240.7 L125.8 241.3 L127.3 241.7 L128.9 242.2 L130.4 242.6 L132.0 242.9 L133.6 243.3 L135.1 243.6 L136.7 243.9 L138.2 244.1 L139.8 244.3 L141.3 244.6 L142.8 244.7 L144.4 244.9 L145.9 245.1 L147.5 245.2 L149.1 245.3 L150.6 245.4 L152.2 245.5 L153.7 245.5 L155.2 245.6 L156.8 245.6 L158.3 245.7 L159.9 245.7 L161.4 245.7 L163.0 245.7 L164.6 245.7 L166.1 245.7 L167.7 245.7 L169.2 245.6 L170.8 245.6 L172.3 245.5 L173.8 245.5 L175.4 245.4 L176.9 245.3 L178.5 245.3 L180.1 245.2 L181.6 245.1 L183.1 245.0 L184.7 244.9 L186.2 244.8 L187.8 244.7 L189.4 244.6 L190.9 244.5 L192.4 244.3 L194.0 244.2 L195.6 244.1 L197.1 244.0 L198.7 243.8 L200.2 243.7 L201.8 243.5 L203.3 243.4 L204.8 243.2 L206.4 243.1 L207.9 242.9 L209.5 242.8 L211.1 242.6 L212.6 242.4 L214.2 242.3 L215.7 242.1 L217.2 241.9 L218.8 241.8 L220.3 241.6 L221.9 241.4 L223.4 241.2 L225.0 241.0 L226.6 240.8 L228.1 240.7 L229.7 240.5 L231.2 240.3 L232.8 240.1 L234.3 239.9 L235.9 239.7 L237.4 239.5 L238.9 239.3 L240.5 239.1 L242.1 238.9 L243.6 238.7 L245.2 238.5 L246.7 238.3 L248.2 238.0 L249.8 237.8 L251.3 237.6 L252.9 237.4 L254.4 237.2 L256.0 237.0 L257.5 236.8 L259.1 236.5 L260.6 236.3 L262.2 236.1 L263.8 235.9 L265.3 235.7 L266.9 235.4 L268.4 235.2 L270.0 235.0 L271.5 234.8 L273.1 234.5 L274.6 234.3 L276.1 234.1 L277.7 233.8 L279.2 233.6 L280.8 233.4 L282.4 233.2 L283.9 232.9 L285.4 232.7 L287.0 232.4 L288.6 232.2 L290.1 232.0 L291.6 231.7 L293.2 231.5 L294.8 231.3 L296.3 231.0 L297.9 230.8 L299.4 230.5 L300.9 230.3 L302.5 230.1 L304.1 229.8 L305.6 229.6 L307.1 229.3 L308.7 229.1 L310.2 228.8 L311.8 228.6 L313.4 228.4 L314.9 228.1 L316.4 227.9 L318.0 227.6 L319.6 227.4 L321.1 227.1 L322.6 226.9 L324.2 226.6 L325.8 226.4 L327.3 226.1 L328.9 225.9 L330.4 225.6 L331.9 225.4 L333.5 225.1 L335.1 224.9 L336.6 224.6 L338.1 224.4 L339.7 224.1 L341.2 223.8 L342.8 223.6 L344.4 223.3 L345.9 223.1 L347.4 222.8 L349.0 222.6 L350.6 222.3 L352.1 222.1 L353.6 221.8 L355.2 221.5 L356.8 221.3 L358.3 221.0 L359.9 220.8 L361.4 220.5 L362.9 220.3 L364.5 220.0 L366.1 219.7 L367.6 219.5 L369.1 219.2 L370.7 219.0 L372.2 218.7 L373.8 218.4 L375.4 218.2 L376.9 217.9 L378.4 217.7 L380.0 217.4 L381.6 217.1 L383.1 216.9 L384.6 216.6 L386.2 216.3 L387.8 216.1 L389.3 215.8 L390.9 215.5 L392.4 215.3 L393.9 215.0 L395.5 214.8 L397.0 214.5 L398.6 214.2 L400.1 214.0 L401.7 213.7 L403.2 213.4 L404.8 213.2 L406.3 212.9 L407.9 212.6 L409.4 212.4 L411.0 212.1 L412.6 211.8 L414.1 211.6 L415.6 211.3 L417.2 211.0 L418.8 210.8 L420.3 210.5 L421.9 210.2 L423.4 210.0 L424.9 209.7 L426.5 209.4 L428.1 209.2 L429.6 208.9 L431.2 208.6 L432.7 208.4 L434.2 208.1 L435.8 207.8 L437.4 207.5 L438.9 207.3 L440.5 207.0 L442.0 206.7 L443.6 206.5 L445.1 206.2 L446.6 205.9 L448.2 205.7 L449.8 205.4 L451.3 205.1 L452.9 204.8 L454.4 204.6 L455.9 204.3 L457.5 204.0 L459.1 203.8 L460.6 203.5 L462.1 203.2 L463.7 203.0 L465.2 202.7 L466.8 202.4 L468.4 202.1 L469.9 201.9 L471.4 201.6 L473.0 201.3 L474.5 201.0 L476.1 200.8 L477.6 200.5 L479.2 200.2 L480.8 200.0 L482.3 199.7 L483.8 199.4 L485.4 199.1 L486.9 198.9 L488.5 198.6 L490.1 198.3 L491.6 198.0 L493.1 197.8 L494.7 197.5 L496.2 197.2 L497.8 196.9 L499.4 196.7 L500.9 196.4 L502.4 196.1 L504.0 195.8 L505.6 195.6 L507.1 195.3 L508.7 195.0 L510.2 194.8 L511.8 194.5 L513.3 194.2 L514.9 193.9 L516.4 193.7 L518.0 193.4 L519.5 193.1 L521.0 192.8 L522.6 192.5 L524.1 192.3 L525.7 192.0 L527.2 191.7 L528.8 191.4 L530.4 191.2 L531.9 190.9 L533.5 190.6 L535.0 190.3 L536.5 190.1 L538.1 189.8 L539.6 189.5 L541.2 189.2 L542.8 189.0 L544.3 188.7 L545.9 188.4 L547.4 188.1 L549.0 187.9 L550.5 187.6 L552.0 187.3 L553.6 187.0 L555.1 186.7 L556.7 186.5 L558.2 186.2 L559.8 185.9 L561.3 185.6 L562.9 185.4 L564.5 185.1 L566.0 184.8 L567.5 184.5 L569.1 184.3 L570.6 184.0 L572.2 183.7 L573.8 183.4 L575.3 183.1 L576.9 182.9 L578.4 182.6 L580.0 182.3 L581.5 182.0 L583.0 181.7 L584.6 181.5 L586.1 181.2 L587.7 180.9 L589.2 180.6 L590.8 180.4 L592.4 180.1 L593.9 179.8 L595.5 179.5 L597.0 179.2 L598.5 179.0 L600.1 178.7 L601.6 178.4 L603.2 178.1 L604.8 177.8 L606.3 177.6 L607.9 177.3 L609.4 177.0 L611.0 176.7 L612.5 176.5 L614.0 176.2 L615.6 175.9 L617.1 175.6 L618.7 175.3 L620.2 175.1 L621.8 174.8 L623.4 174.5 L624.9 174.2 L626.5 173.9 L628.0 173.7 L629.5 173.4 L631.1 173.1 L632.6 172.8 L634.2 172.5 L635.8 172.3 L637.3 172.0 L638.9 171.7 L640.4 171.4 L642.0 171.1 L643.5 170.9 L645.0 170.6 L646.6 170.3 L648.1 170.0 L649.7 169.7 L651.2 169.5 L652.8 169.2 L654.4 168.9 L655.9 168.6 L657.5 168.3 L659.0 168.1 L660.5 167.8 L662.1 167.5 L663.6 167.2 L665.2 166.9 L666.8 166.7 L668.3 166.4 L669.9 166.1 L671.4 165.8 L673.0 165.5 L674.5 165.3 L676.0 165.0 L677.6 164.7 L679.1 164.4 L680.7 164.1 L682.2 163.8 L683.8 163.6 L685.4 163.3 L686.9 163.0 L688.5 162.7" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<circle cx="162.424" cy="245.714" r="5" style="fill:var(--c-success);fill-opacity:.9"/>
<line x1="162.424" y1="245.714" x2="162.424" y2="280" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<text x="170.42414306999132" y="267.7136243450032" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">τ* = 13,4 dk → %7,5</text>
<line x1="428.222" y1="124.2" x2="450.222" y2="124.2" style="stroke:var(--c-success);stroke-width:2.2"/>
<text x="456.2222222222222" y="128.2" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">250 GB/s → δ = 30 sn</text>
<text x="194.0" y="63.80000000000001" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">kısa τ: yazma baskın</text>
<text x="428.2222222222222" y="91.4" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">uzun τ: arıza sonrası yeniden hesap baskın</text>
</svg>

İki önemli sonuç çıkar:

1. Depolama yazma maliyetini on kat azaltmak, optimum kayıt aralığını $\sqrt{10}$ oranında kısaltabilir.
2. Yazma ve tekrar-hesaplama kaybı optimum noktada aynı büyüklüğe gelir.

**Ama:** Asenkron checkpoint'te $\delta$ olarak mutlaka 'dosyaların yazılması için geçen tüm süreyi' koymayın; eğitim üzerindeki gerçek bloklayıcı etkisini ve snapshot dayanıklılığını modelleyin.

---

### 17.5. Checkpoint tek başına yeterli mi? Sağlık kontrolleri ve arızalı düğümü dışarı alma

Kaydetme ve yeniden başlatma stratejisini öğrendik. Fakat büyük bir kümede esas soru bazen "kaç dakikada checkpoint alırım?" değil, **"bir node bozulduğunda bunu kaç dakikada anlar ve eğitimden izole ederim?"** olur.

Bir trende sekiz vagon olduğunu düşünün. Bir vagonda mekanik sorun varsa bütün treni aynı arızalı vagonla yeniden hareket ettirmek, yola tekrar çıkmış olmak anlamına gelir ama sorunu çözmez. AI cluster'ında da hatalı GPU veya NIC yerinde dururken checkpoint'ten tekrar tekrar başlamak aynı başarısızlığa yol açabilir.

**RAS (Reliability, Availability, Serviceability)**, sistemin arızalarını saptama, hizmeti sürdürebilme ve sorunları giderilebilir hale getirme ilkelerinin genel adıdır. NCCL'in RAS bileşenleri çalışmakta olan dağıtık işlerde erişilemeyen process'leri ve iletişim problemlerini teşhis etmeye yardımcı olur; bu altyapı doğrudan otomatik kurtarma garantisi vermez. [K32]

Gerçek bir GPU platformunda en az üç seviyede sağlık kontrolü gerekir:

1. **Node seviyesi:** GPU ECC/Xid olayları, sıcaklık, power throttling, driver ve BMC durumları.
2. **Fabric seviyesi:** NIC link hataları, throughput, retransmission, RDMA erişilebilirliği, NCCL collective sapmaları.
3. **İş seviyesi:** Step süresi dağılımı, loss/NaN kontrolü, straggler, checkpoint başarısı, job restart sayısı.

NVIDIA'nın SuperPOD işletim rehberi, sorunlu düğümü batch havuzundan çıkarıp tek ve çok düğümlü testlerle doğrulamayı öneriyor. NCCL testleri bu doğrulama repertuarının bir parçası. [K33]

<svg viewBox="0 0 720 342" role="img" aria-label="Düğüm karantina döngüsü. Sağlıklı havuzdaki bir düğüm Xid, ECC veya link hatası gibi telemetriyle ya da yavaş step süresiyle işaretlenir. Boşaltılıp batch havuzundan izole edilir, ardından kademeli olarak doğrulanır: DCGM tanılama, tek düğüm NCCL testi ve çok düğüm NCCL testi. Geçerse havuza kademeli olarak döner; kalırsa onarıma veya RMA&#x27;ya gider ve sonra yeniden test edilir. Arızalı düğüm hâlâ işteyken checkpoint&#x27;ten yeniden başlamak aynı arızayı tekrar üretir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="qr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="qr-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="qr-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Düğüm karantina döngüsü: tespit → izole et → sağlığı kanıtla → geri al</text>
<rect x="16" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="91.0" y="72.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Sağlıklı havuz</text>
<text x="91.0" y="86.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">işler burada çalışır</text>
<rect x="206" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="281.0" y="64.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Tespit</text>
<text x="281.0" y="79.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Xid · ECC · link hatası</text>
<text x="281.0" y="93.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yavaş step süresi</text>
<rect x="396" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="471.0" y="72.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Boşalt + izole et</text>
<text x="471.0" y="86.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">batch havuzundan çıkar</text>
<line x1="166" y1="74" x2="204" y2="74" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="356" y1="74" x2="394" y2="74" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="380" y="128" width="324" height="132" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="392" y="148" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Doğrula (kademeli)</text>
<rect x="392" y="158" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="175.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">1 · DCGM tanılama</text>
<rect x="392" y="190" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="207.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">2 · tek düğüm NCCL testi</text>
<rect x="392" y="222" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="239.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">3 · çok düğüm NCCL testi</text>
<line x1="471" y1="104" x2="471" y2="126" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M380 200 H91 V106" marker-end="url(#qr-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<text x="236" y="192" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">geçti → kademeli dönüş</text>
<path d="M542 260 V286 H360" marker-end="url(#qr-b)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="206" y="270" width="152" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="282.0" y="295.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Onarım / RMA</text>
<text x="452" y="300" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">kaldı</text>
<path d="M282 270 V244 H378" marker-end="url(#qr-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="300" y="238" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yeniden test</text>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Arızalı düğüm hâlâ işteyken checkpoint&#x27;ten yeniden başlamak aynı arızayı tekrar üretir.</text>
</svg>

**Silent Data Corruption (SDC)**, sistemin her zaman açık hata sinyali üretmeden yanlış veriler/sonuçlar oluşturabilmesidir. Bu nedenle sadece "job hiç durmadı" diye her şeyin doğru olduğunu varsaymak güvenli değildir. Küçük deterministik kontroller, eğitim metriklerindeki anormal sıçramalar, checksum ve kritik hesaplama doğrulamaları arıza stratejisine dahil edilmelidir.

İki hedefi birbirinden ayırın: **RPO (Recovery Point Objective)** en fazla ne kadar eğitim ilerlemesini kaybetmeyi kabul ettiğinizi; **RTO (Recovery Time Objective)** ise arızadan sonra işe ne kadar sürede geri dönmek istediğinizi anlatır. RPO checkpoint sıklığı ve tutarlılığıyla, RTO restore hızı ve yedek kaynak hazır bulunuşuyla yakından ilişkilidir.

Bir **recovery objective** tanımlayın: Arıza olduğunda kaç dakikalık iş kaybı kabul edilebilir? Eğitim işinde bu, checkpoint interval ve kurtarma mimarisini etkiler. Canlı inference'ta ise hedef çoğu zaman kullanıcı etkisini minimize etmek, trafiği sağlıklı worker'a taşımak ve yeni istekleri sürdürebilmektir. İki problem için aynı kurtarma politikası uygun olmayabilir.

---



---

### Checkpoint yalnızca bir dosya değildir, zaman makinesinin biletidir

Bir model günlerce eğitim gördükten sonra bir node kaybederse soru “dosyayı nereye yazmıştık?” değildir. Asıl soru şudur: **Modeli hangi noktadan, hangi tutarlılık garantileriyle tekrar ilerletebiliriz?** Ağırlıklarla birlikte optimizer state, scheduler state, veri örnekleme konumu ve RNG durumu da önemli olabilir. Ağırlıkları kaydetmek, eğitime aynı istatistiksel noktadan devam edebilmek için her zaman yeterli değildir.

Burada iki farklı zaman vardır:

- **Checkpoint I/O time:** Verinin depolamaya aktarılmasının toplam süresi.
- **Training pause time:** Eğitimin bu nedenle gerçekten durduğu görünür süre.

Eğer eğitim snapshot'ı güvenli biçimde ayrılıp arka planda aktarılabiliyorsa bu iki süre birbirinden ayrışır. Çok yüksek paralel disk bant genişliği gerekli olabilir, fakat tek başına *işin kaç saniye durduğunu* söylemez.

**RPO/RTO kavramını eğitim açısından sezgisel düşünelim:** RPO, bir arızada ne kadar eğitim ilerlemesinin kaybedilebileceğini; RTO ise eğitimin kullanılabilir duruma ne kadar zamanda dönebileceğini tarif etmeye yardımcı olur. Üretim servislerinde RPO/RTO'nun anlamı daha geniştir; kaybolan istekler, erişilebilirlik ve kullanıcı deneyimi de devreye girer.

### Young–Daly yaklaşımı neden karekök içerir?

Young–Daly formülündeki karekök ($\tau_{\mathrm{opt}}\approx\sqrt{2\delta M}$), iki zıt maliyet eğrisinin geometrik dengesinden doğar. Checkpoint aralığı $\tau$ daraldıkça harcanan I/O blokaj süresi hiperbolik olarak ($1/\tau$) yükselir; aralık genişledikçe arıza anında kaybedilen ve yeniden hesaplanması gereken iş doğrusal olarak ($\tau/2M$) artar.

Toplam kayıp eğrisi, bu iki zıt maliyetin türevlerinin birbirini sıfırladığı noktada minimuma ulaşır. Formülün pratik mühendislik mesajı şudur: Depolama blokaj süresini ($\delta$) 10 kat düşürmek, optimal kayıt aralığını $\sqrt{10}\approx 3{,}16$ kat sıklaştırmanıza olanak tanır. Ancak asenkron snapshot mimarilerinde $\delta$ parametresi toplam disk aktarım süresi değil, yalnızca GPU'nun durduğu bellek kopyalama süresidir.


## 18. Arıza sonrasında süreklilik: Blast radius, RPO, RTO ve geri dönüş

Binlerce GPU'nun çalıştığı bir veri merkezinde "hiç arıza olmayacak" varsayımı hem pahalı hem kırılgandır. Güvenilir mimarinin hedefi, arızanın etkisini **sınırlamak, tespit etmek, kurtarmak ve aynı arızadan ders çıkarmaktır**.

### 18.1. Blast radius: Bir hata kaç kişiyi etkiliyor?

**Blast radius**, tek bir hata veya yanlış değişikliğin etkileyebileceği alanı tarif eder. Bir inference worker'ının kapanması tek replica kaybı olabilir; yanlış bir global routing policy, bütün kullanıcıları etkileyebilir. Bir network switch arızası tek rack'i etkileyebilir; yanlış ortak firmware yükseltmesi tüm cluster'ı.

Bu yüzden **failure domain (arıza alanı)** tasarlanır: host, rack, power feed, leaf switch, network rail, availability zone veya storage failure group gibi. Çoklu replika yaparken bütün replikaları aynı node/rack üzerine koymak, isim olarak yüksek erişilebilirlik olsa da riskleri gerçekten ayırmaz.

### 18.2. Eğitim ile servis kesintisi aynı şekilde kurtarılmaz

| Olay | Büyük training | Online inference |
|---|---|---|
| Tek GPU fail | Distributed job durabilir; checkpoint'ten restore | Request başka replica'ya yönlendirilir; mevcut streaming kısmen kaybolabilir |
| Ağ rail sorunu | Collective hang / timeout | Cross-node KV transfer ve router etkilenebilir |
| Storage outage | Data loader/ckpt yavaşlar | Model loading, log veya cache tier etkilenebilir |
| Yanlış model rollout | Loss/kalite sapması | Anlık kullanıcı kalitesi ve latency etkisi |
| Yetki politikası hatası | Veri sızıntısı riski | Yetkisiz sorgu, tool ve retrieval riski |

### 18.3. RPO, RTO ve point-in-time restore

**RPO (Recovery Point Objective)** kabul edilebilir veri/ilerleme kaybını; **RTO (Recovery Time Objective)** ise hizmetin ne kadar sürede kabul edilebilir seviyeye döneceğini ifade eder. Training'de "en fazla 20 dakika hesap kaybı" gibi bir RPO tanımlanabilir; serving'de amaç çoğu zaman yeni istekleri saniyeler içinde başka replika ile karşılamaktır.

Checkpoint varlığı, **geri yüklenebilirlik** garantisi değildir. Checkpoint'in yeterince tam olup olmadığı, sürüm bağımlılıkları, bozulma kontrolü, erişim izni ve farklı storage failure domain'den restore edilebilirliği ayrıca test edilmelidir.

### 18.4. Silent Data Corruption (SDC) ve model kalitesi

Bazı hatalar sistemi çökertmeden yanlış ara değer oluşturabilir. **SDC**, sessiz veri bozulmasıdır. ECC gibi donanım korumaları bazı bellek hatalarını tespit eder/onarır, fakat model seviyesindeki bütün olası yanlışlıkları kapsayan bir garanti değildir. Eğitimde anormal loss spike, tek bir rank'a özgü NaN/Inf, model checkpoint karşılaştırması veya validasyon kalitesi değişimi operasyonel alarm olabilir. Ancak her loss spike SDC değildir; optimizer, veri örnekleri ve precision değişimi gibi sebepler önce ayıklanmalıdır.

### 18.5. Güvenilir rollout: Canary → Shadow → Production

Yeni inference engine sürümünü doğrudan bütün GPU'lara yüklemek yerine:

1. **Staging ve replay:** Kayıtları hassas veri politikasına uygun anonimleştirilmiş veya sentetik trafikle yeniden oynatın.
2. **Shadow traffic:** Uygunsa üretim isteğinin kopyasını kullanıcıyı etkilemeden yeni sisteme gönderin. Verinin ikincil işleme hattına aktarım yetkisini ayrıca doğrulayın.
3. **Canary:** Küçük kullanıcı yüzdesinde gerçek SLO ve kalite ölçün.
4. **Kademeli rollout:** Her adımda regresyon kapısını kontrol ederek genişletin.
5. **Rollback:** Model, tokenizer, runtime, router ve policy sürümlerinin uyumlu önceki setine geri dönebileceğinizi test edin.

**Uyarı:** Bir model ağırlık sürümünü geri almak, veritabanı şeması veya index değişikliği gibi tüm dış bağımlılıkları otomatik geri almaz. Release birimlerini ve dependency pin'lerini birlikte tutmak gerekir.

### 18.6. Bir olayın anlaşılmasını kolaylaştıran bilgiler

Tek alarm "GPU utilization düştü" ise arıza çözümü uzar. İdeal olay kartında request ID veya job ID, model sürümü, GPU/node/fault-domain bilgisi, son değişiklik, hatanın başlangıç zamanı, DCGM/Xid, NCCL durumu, ağ counter'ları, queue metrics, storage I/O ve ilk güvenli müdahale adımı bulunmalıdır.

---




---

**KISIM VI — Ölçüm: MFU, latency, kalite, maliyet**

*Bir grafikte görünen yüzdeyi gerçek performansla karıştırmamayı ve ölçümler arasındaki ilişkiyi öğreneceğiz.*


## 19. %38 MFU: GPU gerçekten nereye zaman harcıyor?

Başlangıçta sorduğumuz büyük soruya geldik: “1.024 GPU aldık ama MFU %38. Kalan performans nerede?”

Artık elimizde birkaç olası cevap var: yeterince büyük GEMM yok olabilir, ağırlıklar HBM'den yavaş taşınıyor olabilir, collective haberleşme uzun sürüyor olabilir, pipeline stage'leri boş kalıyor olabilir veya sistem checkpoint sırasında duruyor olabilir.

Fakat bu ihtimallerin hiçbirini ölçmeden gerçek neden olarak ilan edemeyiz. MFU bir **sonuç göstergesidir**; kendi başına teşhis raporu değildir.

### 19.1. GPU Utilization, MFU ve HFU aynı metrik değildir

Önce bir tuzağı temizleyelim. Telemetri ekranında GPU utilization %95 görmek, “teorik Tensor Core kapasitesinin %95'ini faydalı model hesabına çevirdik” demek değildir.

Bu metrikleri, bir fabrikanın “ışıkları ne kadar süre açık?”, “makineler kaç ürün üretti?” ve “kaç ürün gerçekten müşteriye yaradı?” soruları gibi düşünün. Benzer görünürler ama farklı gerçekleri ölçerler.

Üç kavramın birbirine karıştırılması tehlikelidir:

**GPU Utilization**, belirli örnekleme aralığında GPU'nun aktif kernel çalıştırmasıyla ilgilenen donanım telemetrisidir. Yüksek olması, matris birimlerinin tepe verime ulaştığını göstermez.

**Model FLOPs Utilization (MFU)**, modelin faydalı FLOP/s hızının teorik donanım FLOP/s tavanına oranıdır. Model FLOP hesabı çoğunlukla mimari ve token throughput'una dayanan bir tahmindir; hangi operasyonların sayıldığı belirtilmelidir.

**Hardware FLOPs Utilization (HFU)**, gerçek yürütülen donanım FLOP/s miktarını tavanla karşılaştırmaya çalışır. Örneğin aktivasyon recomputation gibi tekrar hesaplamaları HFU'yu artırabilir, ancak faydalı model FLOP hesabını aynı oranda artırmadığı için MFU'yu yükseltmeyebilir.

| Metrik | Sorduğu soru | Tipik tuzak |
|---|---|---|
| GPU utilization | GPU çalışıyor mu? | 'Çalışıyor' ≠ 'verimli model hesabı' |
| SM active / achieved occupancy | İş yürütme kaynakları ne kadar kullanılıyor? | Yüksek occupancy tek başına yüksek TFLOP/s değildir |
| Tensor Core throughput | MMA birimleri ne kadar hesap yapıyor? | Sadece GEMM'i ölçer, tüm step'i değil |
| MFU | Teorik tepenin ne kadarı faydalı model FLOP'una dönüştü? | Model FLOP tanımı ve step süresi önemli |
| HFU | Donanımın gerçek hesaplama işi tavanın ne kadarına ulaştı? | Faydalı olmayan recomputation da sayılabilir |
| Effective training time | Geçen toplam sürenin ne kadarı eğitimde geçti? | Checkpoint/arıza kesintileri ayrıca hesaba katılmalıdır |

### 19.2. 1.024 H100 ve %38 için kontrol hesabı

BF16 dense teorik tavanı 990 TFLOP/s/GPU olarak kullanırsak:

$$
C_{\mathrm{peak,total}}=1024\times990\ \mathrm{TFLOP/s}
\approx1.014\ \mathrm{EFLOP/s}
$$

MFU %38 ise:

$$
C_{\mathrm{useful}}\approx1.014\times0.38
\approx0.385\ \mathrm{EFLOP/s}
$$

Bu **%62 GPU'ların fiziksel olarak kapalı olduğu** anlamına gelmez. %62, tepe model hesaplama kapasitesine göre faydalı modele dönüşmeyen potansiyeldir. Bellek, kolektif iletişim, pipeline, kernel verimi ve yazılım overhead'i farklı paylarla buna katkıda bulunabilir.

<svg viewBox="0 0 720 226" role="img" aria-label="1.024 H100&#x27;ün yaklaşık 1,014 EFLOP/s&#x27;lik teorik BF16 yoğun tepe değerinin çubuğu. %38&#x27;i, yaklaşık 0,385 EFLOP/s, faydalı model FLOP&#x27;una dönüşür. Kalan %62 faydalı model hesabına dönüşmeyen potansiyeldir; GPU&#x27;ların %62&#x27;sinin kapalı olduğu anlamına gelmez. Payları profil çıkarılmadan bilinmeyen olası nedenler: memory-bound kernel&#x27;lar, kolektif beklemeler, pipeline bubble, küçük veya birleşmemiş kernel&#x27;lar, HFU&#x27;da sayılıp MFU&#x27;da sayılmayan recomputation ve veri ya da checkpoint duraklamaları." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">%38 MFU&#x27;da 1.024 H100: %62 nedir, ne değildir?</text>
<text x="16" y="50" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">teorik tepe: 1.024 × 990 TFLOP/s ≈ 1,014 EFLOP/s (BF16 yoğun)</text>
<rect x="16" y="60" width="261.44" height="40" rx="4" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="146.72" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">faydalı model FLOP&#x27;u</text>
<text x="146.72" y="94" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%38 ≈ 0,385 EFLOP/s</text>
<rect x="277.44" y="60" width="426.56" height="40" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="490.71999999999997" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">%62 faydalı model hesabına dönüşmeyen</text>
<text x="490.71999999999997" y="94" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">GPU&#x27;ların %62&#x27;si kapalı DEMEK DEĞİL</text>
<text x="277.44" y="124" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">olası nedenler; paylar profil çıkarılmadan bilinmez:</text>
<rect x="277.44" y="134" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.44" y="148.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">memory-bound kernel</text>
<rect x="491.44" y="134" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594.44" y="148.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">kolektif bekleme</text>
<rect x="277.44" y="162" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.44" y="176.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">pipeline bubble</text>
<rect x="491.44" y="162" width="206" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:.1;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="594.44" y="176.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">küçük / birleşmemiş kernel</text>
<rect x="277.44" y="190" width="206" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:.1;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="380.44" y="204.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">recomputation (HFU&#x27;da, MFU&#x27;da değil)</text>
<rect x="491.44" y="190" width="206" height="22" rx="4" style="fill:var(--c-danger);fill-opacity:.1;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="594.44" y="204.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">veri / checkpoint duraklaması</text>
</svg>

### 19.3. MFU %38 ise önce hangi veriyi toplamalıyız?

Bir performans sorununu düzeltmek için ilk refleksimiz GPU sayısını veya batch'i artırmak olmamalı. Önce zamanın nerede harcandığını görmeliyiz. Bunu bir hastalığa yalnızca termometreyle teşhis koymaya benzetmeyin; ek ölçümlere ihtiyaç var.

Bir training step'i zaman çizelgesine bölelim: hangi aralıkta matris kernel'ı, hangi aralıkta kolektif iletişim, hangi aralıkta dataloader, hangi aralıkta depolama yazma var? Sıra sıra beklemeler görünür hale gelmeye başladığında çözüm de belirginleşir.

Bir performans sorununu düzeltmeden önce **GPU ve host zaman çizelgesini** görmemiz gerekir.

Önerilen gözlem katmanları:

1. **Model KPI:** token/s, step/s, MFU, sequence length, global batch, microbatch, gradient accumulation.
2. **Kernel KPI:** GEMM shape, Tensor Core instructions, achieved FLOP/s, DRAM throughput, cache hit, register spill.
3. **Communication KPI:** NCCL AllReduce/AllGather/ReduceScatter/AllToAll süreleri, byte miktarı, overlap yüzdesi.
4. **Pipeline KPI:** stage başına compute, transfer ve bubble; *straggler* stage dağılımı.
5. **I/O KPI:** data wait, CPU parse/decode süresi, checkpoint pause, durable write süresi, dosya sistemi p95.
6. **Reliability KPI:** interruption sayısı ve nedeni, restart/recovery time, son kullanılabilir checkpoint yaşı.

Pratik araç aileleri arasında **NVIDIA Nsight Systems**, **Nsight Compute**, **DCGM**, **NCCL tests**, PyTorch Profiler, sistem I/O ve network telemetrisi bulunur. Bunların her biri başka ölçekte bilgi sağlar:

- Nsight Systems: GPU/CPU kernel, stream ve communication zaman çizgisi.
- Nsight Compute: Bir CUDA kernel'ının compute, cache, memory ve occupancy ayrıntısı.
- DCGM: GPU health ve operasyonel performans telemetrisi.
- NCCL tests: İletişim topluluk işlemlerinin kapasitesine ilişkin baz çizgisi.

### 19.4. Belirti → olası sebep → deney

| Gözlenen belirti | Muhtemel darboğaz | Yapılabilecek kontrollü deney |
|---|---|---|
| HBM throughput yüksek, TFLOP/s düşük | Bellek sınırlı kernel | Batch ve weight precision değiştir; Roofline karşılaştır |
| Düşük Tensor Core kullanım, küçük GEMM | Küçük/parçalı matris şekli | Tiling, kernel fusion, paralellik derecesini test et |
| NCCL AllReduce süresi büyük | Ağ, collective veya mesaj düzeni | Aynı-node ve cross-node NCCL microbenchmark |
| TP artırınca step yavaşlıyor | TP sınırı aşırı büyümüş olabilir | TP=4/8/16 topoloji eşlemesini kıyasla |
| PP stage'lerinin bir kısmı boş | Pipeline bubble / dengesizlik | Microbatch ve schedule'ı değiştir |
| GPU'lar belirli aralıklarla duruyor | Checkpoint veya dataloader | Snapshot pause ile persist süresini ayrı ölç |
| Tek GPU tüm step'i uzatıyor | Straggler, ECC, termal veya process sorunu | Rank bazında step / clock / hata kaydı |
| Decode yavaş, prefill güçlü | Ağırlık okuma veya KV trafiği | Concurrency, continuous batching, quantization karşılaştır |
| OOM, düşük kullanıcı sayısı | Ağırlık + KV + fragmentation | KV cache/GQA/paged allocation analizi |

**Altın kural:** Aynı anda on ayarı değiştirmeyin. Testte tek değişkeni izole edin; optimizasyonun throughput, latency, kalite, maliyet ve dayanıklılık üzerindeki etkisini birlikte raporlayın.

### 19.5. Training ve inference için farklı karar matrisi

| Mimari karar | Training'de etkisi | Inference'ta etkisi |
|---|---|---|
| Daha hızlı Tensor Core | Büyük GEMM ve forward/backward'ı hızlandırabilir | Özellikle prefill / yüksek batch yarar görür |
| Daha yüksek HBM bandwidth | Veri hareketi içeren kernel'lara yardımcı | Küçük batch decode için kritik olabilir |
| Daha büyük HBM kapasitesi | Aktivasyon ve model state için alan | Daha çok eşzamanlı istek/KV cache |
| NVLink/NVSwitch | TP ve yerel collectives | Çok-GPU model serving iletişimi |
| Compute fabric / RDMA | DP, PP, EP ve CP iletişimi | Ayrık serving / dağıtık inference |
| Yüksek burst storage | Checkpoint/restart | Model yükleme ve cold start |
| GDS | Uyumlu I/O pipeline'ında kopya azaltma | Büyük GPU'ya veri aktarımında koşullu yarar |
| FP8/INT8/INT4 | Hassasiyet/doğruluk ve eğitim reçetesi | Ağırlık belleği, throughput, kalite takası |

### 19.6. Bir operasyon sabahı: %38 MFU'yu nasıl teşhis ederdik?

Diyelim ki üretim ekibi sabah şu mesajı gönderdi: “Dün gece yeni checkpoint stratejisine geçtik; MFU %42'den %38'e düştü.” Bunu duyunca doğrudan GPU sayısını değiştirmek yerine bir dizi kontrollü soru sormak daha doğru olur.

**İlk soru: Gerçek iş yükü değişti mi?** Global batch, sequence length, precision ve model FLOP hesabı aynı mı? Özellikle context length artmışsa MFU değişiminin nedeni yalnızca yazılım güncellemesi olmayabilir. Karşılaştırdığımız iki koşul gerçekten aynı değilse performans farkını tek bir mimari bileşene bağlamak yanıltıcıdır.

**İkinci soru: MFU düşüşü sürekli mi yoksa periyodik mi?** Eğer zaman çizelgesinde her birkaç dakikada bir büyük boşluk oluşuyorsa checkpoint veya dataloader gibi periyodik işler şüpheli hale gelir. Eğer her adımın sonunda NCCL çağrıları uzuyorsa fabric veya collective yerleşimine bakılır. Eğer GPU kernel'ları sürekli çalışıyor ama düşük FLOP/s üretiyorsa Roofline ve GEMM verimliliği daha anlamlı başlangıç noktalarıdır.

**Üçüncü soru: Tek bir katılımcı mı herkesi bekletiyor?** Rank bazında süreleri karşılaştırın. Binlerce GPU'dan yalnızca birkaçının termal throttling yaşaması, ECC hatalarıyla yeniden denemesi veya veri yüklemede gecikmesi tüm eşzamanlı adımı uzatabilir. Ortalama GPU metriği bu durumu gizleyebilir.

**Dördüncü soru: Hangi hipotezi en küçük deneyle test edebilirim?** Örneğin aynı sekiz GPU üzerinde NCCL testi yapıp bunu farklı node'lara genişletebilir; checkpoint'i kısa süreliğine devre dışı bırakarak step time farkını ölçebilir; aynı model için batch ve TP derecesini değiştirirken yalnızca tek değişkeni oynatabilirsiniz.

**Son soru: Kullanıcı açısından kazanç var mı?** Eğitim senaryosunda faydalı token/s ve toplam tamamlanma süresine; inference senaryosunda TTFT, TPOT, throughput ve p95/p99'a bakın. MFU yükselmesi güzel bir işarettir ama tek başına nihai başarı kriteri değildir.

Böylece sistematik bir teşhis yolu oluşturmuş oluruz: **ölç → darboğazı yerleştir → tek hipotezi test et → gerçek faydayı doğrula.** Şimdi bu genel yöntemi modern Blackwell benchmark’larından gerçek rakamlarla genişletelim.


### 19.7. MFU'yu sadece okumayalım, gerçekten hesaplayalım

Şimdi ölçüm ekranından bir adım uzaklaşıp kendimiz hesap yapalım. Çünkü bir metriğin *hangi sayılardan üretildiğini* öğrenmek, o metriği yorumlamanın en güvenilir yoludur.

Üç şeye ihtiyacımız var: modelin bir token'ı eğitirken yaklaşık kaç FLOP gerektirdiği, kümenin saniyede kaç token işlediği ve GPU'ların kullanılan veri tipindeki teorik tepe işlem gücü.

$$
\mathrm{MFU}=\frac{F_{\mathrm{model/token}}\times R_{\mathrm{tokens/s}}}{N_{\mathrm{GPU}}\times C_{\mathrm{peak/GPU}}}
$$

Burada **token throughput**, tüm eğitim kümesinin saniyede işlediği *gerçek* token miktarıdır. **Peak FLOP/s** ise tek bir GPU'nun belirtilen precision ile *sparsity olmadan* ulaşabileceği teorik tepe değerdir. H100 SXM için BF16 yaklaşık 989 TFLOP/s düzeyindedir. B200, B300, GB200 ve GB300 için aynı sayı kullanılamaz; mimarinin ve veri tipinin kendi referans değerini seçmemiz gerekir. [K16]

Dense decoder-only Transformer için popüler ilk yaklaşım şöyledir:

$$
F_{\mathrm{model/token}}\approx 6N_{\mathrm{param}}
$$

**Neden 6?** Bir ağırlığın forward işlemindeki çarpma-toplama maliyetini kabaca **2 FLOP**, backward sırasında hem giriş gradyanları hem de ağırlık gradyanları için yapılan iki benzer hesabı **4 FLOP** düşünün. Böylece yaklaşık 6 FLOP/parametre/token elde ederiz. Bu öğretici yaklaşım attention'ın dizi uzunluğuyla büyüyen ek maliyetlerini, embedding/output operasyonlarını ve bazı mimari ayrıntıları tam kapsamaz. Uzun context'te daha ayrıntılı modele ihtiyaç vardır. Lambda'nın raporu da bu nedenle basit formüle bir attention düzeltmesi ekliyor. [K12]

**Küçük bir örnek:** Diyelim ki 70B dense bir modeli, **128 adet H100 SXM** üzerinde eğitiyoruz. Küme toplamda saniyede **100.000 token** işliyor olsun. Bunlar raporlanmış bir benchmark değil; formülün nasıl kullanıldığını göstermek için seçtiğimiz sayılar.

$$
F_{\mathrm{token}}\approx 6\times70\times10^9=420\times10^9\ \mathrm{FLOP}
$$

$$
R_{\mathrm{useful}}\approx 420\times10^9\times100000=42\ \mathrm{PFLOP/s}
$$

$$
C_{\mathrm{cluster}}\approx128\times989\ \mathrm{TFLOP/s}=126.6\ \mathrm{PFLOP/s}
$$

$$
\mathrm{MFU}\approx\frac{42}{126.6}=\mathbf{\%33.2}
$$

Bir yerde 'MFU %33' dendiğinde artık soracağınız soru yalnızca *neden düşük?* olmayacak: **Token throughput, FLOP/token hesabı, precision, GPU sayısı ve ölçüm aralığı tam olarak ne?**

> **Ölçüm kuralı:** Aynı model ve aynı global batch için, kısa bir kernel benchmark'ının MFU'sunu haftalar süren tüm eğitimin *duvar saati* verimliliğiyle karıştırmayın. Warm-up, veri yükleme, checkpoint, duraklama ve yeniden başlatma süreleri dahil edilip edilmediğini mutlaka yazın.

### 19.8. 'İyi MFU kaçtır?' sorusunda tek bir sihirli sayı yok

Bir otomobilin şehir içi yakıt tüketimini yarış pistindeki azami hızıyla değerlendiremeyiz. MFU benchmark'larında da aynı sorun var: **küçük bir dense model, yüzlerce milyar parametreli dense model ve MoE model aynı karşılaştırma sınıfında değildir.** Hatta aynı modelin 8K ve 128K context uzunluğundaki ölçümleri farklı olabilir.

Aşağıdaki sayılar bu nedenle **evrensel hedef bantları değil, belirli koşullarda yayımlanmış somut ölçümlerdir**:

| Araştırma / sistem | İş yükü ve donanım | Raporlanan sonuç | Bu sayı bize ne anlatıyor? |
|---|---|---:|---|
| **Meta Llama 3** [K5] | 405B dense, 16.384 H100; farklı context/paralellik ayarları | **%38–43 MFU** | Dev ölçekli üretimde güçlü bir referans; koşula bağlı |
| **MegaScale (NSDI 2024)** [K13] | 175B dense, **12.288 GPU** | **%55,2 MFU** | '10 bin+ GPU için %45 üstü olanaksız' iddiasını çürütüyor |
| **Lambda / Llama 3.1 8B** [K12] | 8× B200, BF16, **8K** sequence | **%55,34 MFU** | Tek sunucuya yakın ölçekte optimize edilmiş dense eğitim |
| **Lambda / Llama 3.1 8B** [K12] | 8× B200, BF16, **32K** sequence | **%60,48 MFU** | Bu deneyde daha uzun sequence ile daha yüksek hesaplama verimi |
| **Lambda / Llama 3.1 70B** [K12] | 16× B200, BF16, **8K** sequence | **%50,20 MFU** | Varsayılan ayarlar iyileştirilince belirgin kazanım |
| **Lambda / Llama 3.1 405B** [K12] | 128× GB300, BF16, **8K** sequence | **%52,69 MFU** | Büyük modelde bile modern rack-scale altyapı ve yazılım ayarı etkili |

Bir ayrıntı: MegaScale'ın **%55,2** sonucu ile Lambda'nın **%60,48** sonucu doğrudan bir 'kim daha iyi?' sıralaması değildir. Donanım, model, FLOP tanımı, sequence length ve ölçüm kapsamı farklıdır. Ayrıca bunlar **2024–2025 tarihli yayımlanmış örneklerdir**; bütün sistemler için Ekim 2026 itibarıyla kesin bir dünya rekoru tablosu oluşturmaz.

Buna rağmen güçlü bir mesajımız var: **'Büyük modelse MFU mutlaka %40'ı aşamaz' diye bir fizik kanunu yok.** İletişimi örtüştürme, uygun paralellik planı ve kernel optimizasyonu bu sınırı ciddi biçimde değiştirebilir. [K13]

### 19.9. Bazen MFU'yu ikiye katlamak için yeni GPU değil, doğru konfigürasyon gerekir

Lambda'nın deneyini bir mühendis gibi okuyalım. **Llama 3.1 70B, 16× HGX B200, BF16 ve 8K sequence** için varsayılan TorchTitan ayarlarıyla **%23,83** MFU görülüyor. Optimize edilmiş konfigürasyonda değer **%50,20** oluyor. [K12]

$$
\text{İyileşme çarpanı}=\frac{50.20}{23.83}\approx\mathbf{2.11\times}
$$

Bu sonuçta modelin mimarisi veya veri kümesi değişmiyor. Başlıca farklar arasında **tensor parallelism derecesinin 8'den 1'e düşürülmesi**, **full activation checkpointing yerine selective activation checkpointing**, batch ve kernel optimizasyonları bulunuyor. Yeni Blackwell donanımının bellek kapasitesi ve işlem özellikleri, eski donanım için hazırlanmış ayarlara körü körüne bağlı kalmamanın önemini gösteriyor.

Sezgisel düşünelim: Koca bir toplantıyı sekiz kişinin sürekli birbirine danışarak yürütmesi yerine, iş artık tek bir kişinin masasına sığabiliyorsa, gereksiz koordinasyonu kaldırabiliriz. Tabii aynı işi tek kişinin yapabilmesi için masanın ve hesaplama kapasitesinin gerçekten yeterli olması gerekir. **TP=1 her sistemde en iyi seçenektir** sonucu buradan çıkmaz.

| Aynı B200 ortamında test edilen düzen | MFU | Yorum |
|---|---:|---|
| Llama 3.1 8B, 8× B200, varsayılan | %44,46 | Temel TorchTitan düzeni |
| Llama 3.1 8B, 8× B200, optimize | %55,34 | **1,24×** göreli kazanım |
| Llama 3.1 70B, 16× B200, varsayılan | %23,83 | İletişim/bellek maliyeti yüksek |
| Llama 3.1 70B, 16× B200, optimize | %50,20 | **2,11×** göreli kazanım |

*Kaynak: Lambda, 2025 MFU teknik raporundaki BF16/8K sonuçları [K12]. Bunlar her uygulamanın otomatik yakalayacağı değerler değildir.*

Burada ince bir analiz daha yapalım: MFU'nun **%23,83'ten %50,20'ye çıkması**, eğer iş yükü ve FLOP tanımı aynıysa aynı donanımda faydalı token işleme hızının yaklaşık **2,11 katına** çıkması anlamına gelebilir. Ama eğitim hedefi değişiyorsa, yakınsama davranışı farklılaşıyorsa veya precision farklıysa toplam *time-to-quality* kazancı ayrıca ölçülmelidir.

### 19.10. Overlap: Bekleyişi yok etmek yerine onun üzerine başka iş koymak

Makale boyunca warp scheduler'dan NVLink'e kadar aynı fikri tekrar gördük. Şimdi bunu dağıtık eğitimde somutlaştıralım.

Diyelim ki bir katmanın 8 ms hesaplaması ve 5 ms GPU'lar arası iletişimi var. Her işi sırayla yürütürsek:

$$
T_{\mathrm{serial}}=8+5=13\ \mathrm{ms}
$$

Hesaplama ile bağımsız iletişim tamamıyla örtüşebilseydi kritik yol yaklaşık:

$$
T_{\mathrm{overlap}}\gtrsim\max(8,5)=8\ \mathrm{ms}
$$

Bu, **beş milisaniyelik iletişimin fiziksel olarak yok olduğu** anlamına gelmez. Aynı zaman diliminde başka bir iş yaparak iletişimin kullanıcıya görünen maliyetini *gizlemiş* oluruz. Gerçekte bağımlılıklar, GPU kaynaklarının paylaşılması ve ağ rekabeti nedeniyle tam örtüşme her zaman sağlanmaz.

<svg viewBox="0 0 720 254" role="img" aria-label="8 ms hesap ve 5 ms iletişimi olan bir katman. Seri çalışırsa 13 ms sürer. Bağımsız iletişim hesapla tam örtüşürse kritik yol en az max(8, 5) yani 8 ms olur. Bu, step süresinde %38 kısalma ya da aynı GPU&#x27;larda 1,63 kat throughput demektir; 5 ms&#x27;lik iletişim yine yaşanır, yalnızca kritik yolun dışına çıkar. Bağımlılıklar, ortak SM ve HBM kullanımı ve ağ rekabeti yüzünden gerçek örtüşme kısmidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="om-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="om-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">8 ms hesap + 5 ms iletişim: seri ve tam örtüşmeli</text>
<text x="16" y="74" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">seri</text>
<rect x="150" y="58" width="288" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="76.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">hesap 8 ms</text>
<rect x="438" y="58" width="180" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="528.0" y="76.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">iletişim 5 ms</text>
<text x="626" y="77" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">= 13 ms</text>
<text x="16" y="136" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">örtüşmeli</text>
<rect x="150" y="120" width="288" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="138.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">hesap 8 ms</text>
<rect x="222" y="152" width="180" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="312.0" y="167.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">iletişim 5 ms (gizli)</text>
<text x="446" y="139" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">≥ max(8, 5) = 8 ms</text>
<line x1="438" y1="50" x2="438" y2="184" style="stroke:var(--c-success);stroke-width:1.2;stroke-dasharray:4 3"/>
<line x1="618" y1="50" x2="618" y2="96" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="16" y="196" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="216" text-anchor="start" style="fill:var(--c-text);font-size:12px">Step süresi −%38 (13 → 8 ms) = aynı GPU&#x27;larda 1,63× throughput; 5 ms yine yaşanır, yalnızca kritik yoldan çıkar.</text>
<text x="28" y="233" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Gerçek örtüşme kısmidir: bağımlılıklar, ortak SM/HBM ve ağ rekabeti kazancı azaltır.</text>
</svg>

Bu yaklaşımın üç çok önemli kullanım alanı var:

**FSDP prefetching:** Sonraki Transformer katmanının ağırlıklarını daha o katmanın hesaplaması başlamadan *AllGather* ile istemek. Hesaplama sürerken gelecek katmanın verisi hazırlanır. Bedeli, prefetched ağırlıkların geçici olarak daha fazla HBM kullanmasıdır. Lambda'nın 8B deneylerinde ileri prefetch ayarları %60'ın üzerindeki MFU sonucuna katkıda bulunuyor. [K12]

**Asynchronous Tensor Parallelism:** Bazı TP kolektiflerini bağımsız hesaplamalarla örtüştürmeye çalışmak. Bu, iletişim maliyetini bütünüyle kaldırmaz; kullanılabilir yazılım ve donanım desteğine bağlıdır. [K12]

**DualPipe:** DeepSeek-V3'ün MoE eğitiminde forward/backward parçalarını ve expert dispatch/combine iletişimini dikkatle zamanlayan çift yönlü pipeline düzeni. DeepSeek'in raporunda **PP=16**, **EP=64** ve **ZeRO-1 DP** kullanıldığı, ayrıca maliyetli TP'nin tercih edilmediği belirtiliyor. DualPipe'ın başarısı yalnızca algoritmadan değil, özel AllToAll kernel'ları ile NVLink/InfiniBand topolojisinin birlikte tasarlanmasından kaynaklanıyor. [K9]

> **Sezgisel ayrım:** Pipeline bubble, fabrika hattındaki bir istasyonun iş beklemesidir. Communication overlap ise o istasyonun bir kamyonu beklerken başka hazır siparişi işlemesidir. DualPipe bu fikri forward ve backward işleri arasında uygulayan daha özel bir zamanlama yaklaşımıdır. 'Bütün baloncukları her koşulda sıfırlar' demek doğru değildir.

### 19.11. FP8, BF16 ve MFU: Daha az bit bazen daha çok hız getirir; yüzde aynı kalabilir

BF16 sayılar 16 bit ile, FP8 sayılar 8 bit ile temsil edilir. İlk bakışta 'veri yarıya indi; MFU kesin iki katına çıkar' diye düşünebilirsiniz. Ancak **MFU iki farklı etkiden oluşur:** Faydalı FLOP/s artabilir, fakat donanımın o veri tipi için **teorik tepe FLOP/s** değeri de yükselir.

Örneğin NVIDIA'nın non-sparse referans tablosunda H100 için BF16 yaklaşık **989 TFLOP/s**, FP8 yaklaşık **1.979 TFLOP/s**; B200 için BF16 **2.250 TFLOP/s**, FP8 **4.500 TFLOP/s** olarak veriliyor. [K16]

Sırf veri tipi değişti diye FP8'de 1.200 TFLOP/s'ye çıktığınızı varsayın:

- BF16'da 800/989 = yaklaşık **%81**.
- FP8'de 1.200/1.979 = yaklaşık **%61**.

İkinci koşulda **gerçek hesaplama hızınız artmış**, ama **MFU yüzdeniz düşmüş** olur! Bu düşünce deneyi, neden yalnızca bir MFU yüzdesine bakarak iki precision'ı karşılaştıramayacağımızı açıklıyor.

Lambda da BF16+FP8 karma hassasiyet deneylerinde tam bu sebeple MFU yerine **elde edilen TFLOP/s** metriğini kullanmayı seçiyor. 70B/B200 karma hassasiyet deneyinde varsayılan konfigürasyonla yaklaşık **812 TFLOP/s**, optimize edilmiş düzenle yaklaşık **1.783 TFLOP/s** raporlanıyor; değerler raporun ölçüm tanımında verilmiştir. [K12]

FP8 eğitimi ayrıca ölçekleme katsayıları (*scales*), format dönüşümleri, hangi katmanın hangi precision'da çalışacağı ve sayısal stabilite kararları gerektirir. **Quantization**, sadece belleği küçültmek değil; veri hareketini, Tensor Core kernel'larını ve hata bütçesini birlikte yönetmektir. [K17]

Formatların kendisine bakmak, ölçek katsayısının neden kaçınılmaz olduğunu açıklar:

| Format | İşaret | Üs | Mantis | Yaklaşık en büyük değer | Tipik rol |
|---|---:|---:|---:|---:|---|
| FP16 | 1 | 5 | 10 | 65.504 | Eski karma hassasiyet, inference |
| BF16 | 1 | 8 | 7 | ~3,4 × 10³⁸ | Pretraining'in varsayılanı: FP32 ile aynı üs aralığı |
| FP8 E4M3 | 1 | 4 | 3 | 448 | İleri geçiş: ağırlık ve aktivasyon |
| FP8 E5M2 | 1 | 5 | 2 | 57.344 | Geri geçiş: gradyanlar (daha geniş aralık) |
| NVFP4 (E2M1) | 1 | 2 | 1 | 6 (16'lık blok başına E4M3 ölçekle) | Blackwell'de düşük bitli GEMM, ağırlıkla inference |

Şöyle okuyun: üs bitleri temsil edilebilen aralığı, mantis bitleri hassasiyeti belirler. E4M3 en fazla 448'e kadar çıkabildiği için tensörler önce bir ölçek katsayısıyla bu aralığa sıkıştırılır; ölçek yanlışsa büyük değerler taşar, küçükler sıfıra düşer. NVFP4 bunu bir adım ileri götürür: her 16 değerlik blok kendi FP8 (E4M3) ölçeğini, bütün tensör de ayrıca bir FP32 ölçeğini taşır; bu, 32'lik blok ve ikinin kuvveti ölçek kullanan MXFP4'ten daha ince bir ölçeklemedir. [K17][K57]

**Önemli sonuç:** FP8/BF16 karşılaştırırken MFU'nun yanında **faydalı token/s, model kalitesi, loss yakınsaması, kullanılan precision tanımı ve maliyet/token** raporlanmalıdır. Özellikle Dense ile MoE için aynı FLOP/token yaklaşımı kullanılmamalıdır.

### 19.12. Kernel fusion: Üç ayrı sipariş yerine tek seferde hazırlık

Bir mutfakta önce soğanı doğrayıp tezgâhı temizlediğinizi, sonra domatesi doğrayıp yeniden temizlediğinizi, sonra biberi hazırladığınızı düşünün. Her ayrı işlemde araçları taşıma ve hazırlık süresi var. Bazı işleri aynı hazırlıkta yapmak toplam zamanı azaltabilir.

GPU'daki **kernel fusion** da uygun işlemleri birleştirerek ayrı kernel başlatma (*kernel launch*) yükünü ve ara tensörlerin HBM'ye yazılıp tekrar okunması ihtiyacını azaltabilir.

Örneğin Transformer attention'da Query, Key, Value projeksiyonları aynı girdiyi kullanır. Uygun weight düzeniyle:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

üç projeksiyon yerine ağırlıklar yan yana birleştirilerek:

$$
[Q,K,V]=X[W_Q,W_K,W_V]
$$

şeklinde tek büyük GEMM yapılabilir. Bu **QKV merging** olarak bilinir. Diğer örnekler arasında fused optimizer ve ardışık pointwise işlemlerin birleştirilmesi bulunur. [K12]

**FlashAttention-3** ise aynı fikrin daha incelikli bir versiyonudur: attention hesabını bloklar halinde düzenleyerek gereksiz HBM hareketini azaltır, Hopper'ın asenkron hesaplama/veri taşıma olanaklarını değerlendirir. Fakat **bütün attention hesaplaması SRAM'de kalır** demek doğru olmaz; bloklar ve ara durumlar uygun bellek düzeyleri arasında yönetilir. [K8]

**`torch.compile`** desteklenen PyTorch işlemlerini yakalayıp derleme ve füzyon optimizasyonları yapabilir. Ancak modeli her koşulda tek statik grafa dönüştürmez; *graph break*, yeniden derleme veya donanım uyumsuzluğu görülebilir. Python overhead'ini otomatik **sıfıra** indirdiğini söylemek de doğru değildir. [K14]

Lambda'nın profil örneğinde, optimize edilen 8B çalışmasında **elementwise işlem süresinin payı yaklaşık %14,8'den %1,5'e**, matmul payı yaklaşık **%56,4'ten %68,7'ye** değişiyor. Bunlar **toplam süre içindeki oranlardır**; matmul süresinin tek başına uzadığı ya da kısaldığı sonucuna doğrudan atlamamalıyız. Batch ve derleme koşulları birlikte değişiyor. [K12]

### 19.13. Activation checkpointing paradoksu: Daha az VRAM her zaman daha yüksek MFU değildir

Eğitim sırasında forward aşamasındaki bütün ara tensörleri saklayabiliriz. Ama bunlar HBM'yi doldurur. **Activation checkpointing**, bazı aktivasyonları saklamamayı, backward sırasında gerektiğinde yeniden hesaplamayı önerir.

Sezgisel olarak öğrenci defterindeki her ara işlemi saklamak yerine yalnızca bazı kritik satırları not eder, ihtiyaç duyduğu ara hesabı sonra tekrar yapar. Defter hafifler ama tekrar çözmek zaman alır.

Bu noktada **disk checkpoint** ile **activation checkpoint** ayrımını tekrar hatırlayalım: Birincisi arıza sonrası eğitim durumunu kalıcı depoya kaydeder, ikincisi backward için gerekli GPU ara belleğinden tasarruf sağlar.

| Strateji | HBM tasarrufu | Ek hesaplama | MFU açısından beklenen yön |
|---|---|---|---|
| Aktivasyonları sakla | Daha az | Daha az | Bellek yeterliyse çoğu zaman avantajlı |
| **Selective** checkpoint | Orta | Sınırlı | İyi bir denge olabilir |
| **Full** checkpoint | Daha fazla | Daha çok | Bellek kazanımı uğruna MFU azalabilir |

Püf nokta: **Recomputation sırasında yapılan FLOP'lar MFU'daki faydalı model FLOP hesabına eklenmez.** Dolayısıyla HFU yükselirken MFU düşebilir. Ama recomputation sayesinde daha büyük mikro-batch kullanabiliyor ve OOM'u önleyebiliyorsanız tüm eğitim throughput'u yine de artabilir. PyTorch'un güncel activation checkpointing rehberi bu bellek–hesaplama takasını ayrıntılandırıyor. [K15]

Lambda çalışmasının özellikle vurguladığı nokta bu: Önce mümkünse yeniden hesaplama ihtiyacını azalt, bellek sıkışınca **selective** stratejiyi dene, yeterli olmazsa daha yoğun recomputation veya model paralelleştirme seçeneklerine geç. **Bu evrensel zorunlu sıra değildir**; bir yol haritasıdır. [K12]

### 19.14. MoE'nin MFU'sunu dense modelle neden doğrudan karşılaştıramayız?

Bir dense modelde her token için modelin büyük kısmındaki hesaplama etkinleşir. **Mixture-of-Experts (MoE)** mimarisinde ise bir router, token'ı belirli uzmanlara gönderir. Modelin **toplam parametre sayısı** ile tek token için **aktif parametre sayısı** aynı değildir.

DeepSeek-V3 örneğinde toplam **671B parametre**, token başına yaklaşık **37B aktif parametre** bulunur. Raporda 14,8 trilyon eğitim token'ı ve 2,788 milyon H800 GPU-saatlik toplam eğitim maliyeti belirtiliyor. [K9]

Bu sayılara bakıp basit dense formülüyle $6\times671B$ hesaplamak hatalı olur: Token başına bütün expert'ler çalışmaz. Aynı şekilde $6\times37B$ tek başına attention, router, paylaşılan katmanlar, çoklu eğitim hedefleri ve iletişimin bütün maliyetini tam temsil etmez.

MoE'de özellikle şu kavramları ayıralım:

- **Routing:** Token'ın hangi expert'lere gideceğine karar verilmesi.
- **AllToAll:** Token aktivasyonlarının ilgili expert'leri barındıran GPU'lara gönderilip sonuçların geri birleştirilmesi.
- **Load balancing:** Bazı expert'lerin aşırı dolup diğerlerinin boş kalmasının önlenmesi.
- **Expert Parallelism (EP):** Expert'lerin GPU'lara dağıtılması.
- **DualPipe overlap:** Bu iletişimin bağımsız forward/backward hesaplarıyla mümkün olduğunca örtüştürülmesi.

Paylaştığınız metindeki **'DeepSeek-V3 kesin %21,4–23 MFU'ya ulaşmıştır'** iddiasını **birincil teknik raporda doğrudan doğrulayamadım**. Bu oran ikincil yazılarda türetilmiş olabilir; fakat FLOP/token tanımı ve FP8 tepe FLOP/s paydası verilmeden onu resmi ölçüm gibi kullanmamak gerekir. DeepSeek'in birincil raporundan güvenle kullanabileceğimiz bilgiler mimari, FP8 sistemi, DualPipe, donanım-saat ve eğitim ölçeğidir. [K9]

Başka bir deyişle, **MoE'nin daha düşük MFU göstermesi otomatik olarak daha pahalı, daha yavaş veya kötü optimize edilmiş olduğu anlamına gelmez**. Token başına aktif hesaplama, model kalitesi, toplam eğitim süresi ve efektif maliyet birlikte yorumlanmalıdır.

### 19.15. MFU iyileştirme deneyinin arkasındaki mantık

Bütün bu çalışmaları bir mimari toplantısına götürsek, 'MFU'yu %60 yapalım' gibi bir hedef yerine ölçülebilir bir deney planıyla başlamamız daha doğru olur.

**Birinci aşama — Karşılaştırılabilir bir baseline oluşturun.** Aynı model, aynı sequence length, aynı global batch, aynı precision, aynı GPU sayısı ve aynı veri örnekleme düzeniyle en az birkaç kararlı step çalıştırın. Hem model FLOP/token hesabını hem token/s değerini kaydedin. Warm-up ve checkpoint'leri nasıl ele aldığınızı rapora ekleyin.

**İkinci aşama — Zamanın adresini bulun.** Profilde matmul, attention, pointwise kernel, optimizer, NCCL collective, pipeline bubble ve dataloader sürelerini işaretleyin. Her şeye birden dokunmadan baskın darboğazı belirleyin.

**Üçüncü aşama — Ucuz ve geri döndürülebilir değişikliklerle başlayın.** Önce `torch.compile` desteğini, uygun fused kernel'ları, QKV merging'i, pinned-memory veri hattını, CPU affinity'yi ve batch ayarlarını kontrollü test edin. Tek bir değişkeni değiştirin.

**Dördüncü aşama — Bellek ve paralellik kararlarını tekrar değerlendirin.** TP derecesini eski GPU neslindeki varsayımdan kopyalamayın. HBM yetiyorsa daha az TP, selective activation checkpointing ve veri paralelliği seçeneklerini kıyaslayın. Yetmiyorsa FSDP/HSDP, TP ve uzun context için CP kombinasyonlarını mimariye göre seçin. [K12]

**Beşinci aşama — Örtüşmenin gerçekten oluştuğunu kanıtlayın.** FSDP prefetch, async TP veya MoE için DualPipe benzeri düzenleri açmak tek başına başarı değildir. Nsight Systems zaman çizelgesinde iletişimin hangi bölümünün hâlâ görünür (*exposed*) olduğunu ölçün.

**Altıncı aşama — Precision'ı verim ve kaliteyle birlikte seçin.** BF16 ve BF16+FP8 senaryolarını yalnızca MFU yüzdesiyle kıyaslamayın. TFLOP/s, token/s, loss eğrisi ve kalite testlerini birlikte raporlayın. [K12][K17]

**Yedinci aşama — Sonucun işletmeye etkisine bakın.** MFU'nun %40'tan %50'ye çıkması güzel; fakat eğitim daha sık çöküyorsa, checkpoint daha uzun sürüyorsa veya model kalitesi değiştiyse iş hedefi aynı ölçüde iyileşmemiş olabilir. Son gösterge **time-to-quality, GPU-saat, maliyet ve dayanıklılık** olmalıdır.

> **Bir cümlede temel sezgi:** Önce hangi sürenin gerçekten boşa gittiğini ölçün; sonra veriyi yakına alın, gereksiz iletişimi azaltın, kaçınılmaz beklemeyi faydalı hesaplamayla örtüştürün ve kazanımı aynı koşullarda tekrar ölçün.


---


### 19.16. MFU'nun ötesi: İyi mimariyi hangi metrikle seçmeliyiz?

Bir üretim hattının verimliliği %90 olsun ama yanlış ürünü üretiyor olsun. Başka bir hat %65 makine kullanımına rağmen müşterilerin istediği ürünü daha ucuza ve zamanında teslim etsin. Hangi hat daha iyi tasarlanmıştır?

MFU, **eğitim hesaplama verimliliği** için değerlidir; ancak bütün AI platformunun ana başarı metriği değildir.

Önce ölçekleri ayıralım:

| Metrik | Ne ölçer? | Tek başına neyi söylemez? |
|---|---|---|
| **Job MFU** | Çalışan model işinin teorik FLOP tavanına oranı | Tüm kümenin ne kadar dolu olduğu |
| **Fleet allocation/utilization** | GPU kapasitesinin ne kadarı işlere ayrılmış/meşgul | Her işin verimli hesaplama yapıp yapmadığı |
| **Training tokens/s** | İşlenen token hızı | Modelin doğru öğrendiği veya yakınsadığı |
| **Time-to-quality** | Belirli kalite eşiğine ulaşma süresi | Tekil kernel verimliliği |
| **Serving goodput** | SLO'yu sağlayan faydalı sunum işi | SLO dışı kapasitenin israf miktarı |
| **Cost per million tokens** | Birim inference maliyeti | Yanıt kalitesi veya tail latency |
| **Tokens per joule** | Enerji başına tamamlanan token | Tenant adaleti veya operasyonel risk |

**Goodput**, genel anlamda hizmet düzeyi gereksinimini karşılayan *yararlı* çıktı hızıdır. Örneğin bir sistem saniyede çok sayıda token üretip ilk token için istenen gecikmeyi düzenli olarak aşıyorsa throughput iyi, kullanıcının yaşadığı hizmet kötü olabilir. MLPerf Endpoints yaklaşımı da farklı eşzamanlılık seviyelerindeki interactivity ve throughput eğrisini birlikte ölçmeye odaklanıyor. [K34]

**Training için mimari hedef:** Aynı değerlendirme eşiğine hangi sistemin daha düşük toplam süre, GPU-saati, enerji ve arıza riskiyle ulaştığını karşılaştırın.

**Inference için mimari hedef:** Aynı kalite ve güvenlik şartlarında, gerçek giriş/çıkış token dağılımı altında hangi topolojinin TTFT/TPOT p95-p99 hedefini sağlayarak en düşük birim maliyeti sunduğunu ölçün.

> **Düşündüren nokta:** Model sürümünü değiştirmek, daha kısa prompt kullanmak, KV cache isabetini yükseltmek veya doğru worker'a yönlendirmek; bazen daha fazla GPU almaktan daha yüksek *hizmet kalitesi* kazandırabilir. Buna karar vermek için yalnızca donanım ölçümü değil, ürün iş yükü dağılımını da bilmeliyiz.

### 19.17. Ölçüm bir seferlik benchmark değil, bir deney tasarımıdır

AI platformu performansını ölçmek için şu dört adımı sistematik yapalım:

**1 — Ölçüm koşullarını sabitleyin.** Model commit'i, precision, kernel sürümü, driver/CUDA/NCCL, GPU sayısı ve tipi, ağ topolojisi, input/output token dağılımı, dataset ve concurrency sabit değilse iki sonucu doğrudan kıyaslamak güvenilir değildir.

**2 — Katmanları ayrı test edin.** GPU için GEMM/memory microbenchmark; ağ için NCCL collective; depolama için eşzamanlı I/O; uygulama için gerçek training step veya serving request testi çalıştırın. Tek bir uçtan uca sayı, darboğazın yerini göstermeye yetmez.

**3 — Tail latency ve tekrarları kaydedin.** Ortalama step süresi çok iyi olabilir ama her yüzüncü step iki kat uzun sürüyorsa planlama ve network straggler'ları araştırılmalıdır. Inference için aynı durum p95/p99 TTFT ve TPOT açısından geçerlidir.

**4 — Optimizasyon başına karşı-olgusal deney tasarlayın.** Örneğin TP=4'ten TP=8'e geçişte diğer parametreleri değiştirmeden step time, memory, NCCL time ve loss eğrisi karşılaştırın. Birden çok değişken aynı anda değişirse gerçek kazanımın kaynağı belirsizleşir.

Pratik bir ölçüm yığını şu şekilde ayrılabilir:

<svg viewBox="0 0 720 302" role="img" aria-label="Gözlemlenebilirlik yığını. Dört sinyal kaynağı: DCGM&#x27;den güç, ECC, Xid ve saat gibi GPU ve sunucu telemetrisi; NIC, RDMA sayaçları ve NCCL RAS&#x27;tan ağ telemetrisi; step süresi, MFU, loss ve restart gibi eğitim işi metrikleri; TTFT, TPOT, kuyruk ve KV kullanımı gibi serving metrikleri. Bunlar DCGM exporter ve Prometheus ya da OpenTelemetry üzerinden, istek başına tek trace_id ile ilişkilendirilen metrik ve trace deposuna akar; oradan aksiyon doğar: SLO burn-rate alarmı, düğüm karantinası, kapasite planlama ve faydalı token maliyeti. DCGM exporter yalnızca GPU alanlarını verir; geri kalanı ayrıca enstrümante edilmelidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ob-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ob-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Gözlemlenebilirlik: dört sinyal kaynağı, ilişkilendirmek için tek yer</text>
<rect x="16" y="44" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="126.0" y="66.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU &amp; sunucu</text>
<text x="126.0" y="80.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">DCGM: güç, ECC, Xid, saat</text>
<path d="M236 68 C 262 68, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="102" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="126.0" y="124.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Ağ</text>
<text x="126.0" y="138.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NIC, RDMA sayaçları, NCCL RAS</text>
<path d="M236 126 C 262 126, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="160" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="182.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Eğitim işi</text>
<text x="126.0" y="196.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">step süresi, MFU, loss, restart</text>
<path d="M236 184 C 262 184, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="218" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="126.0" y="240.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Serving</text>
<text x="126.0" y="254.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">TTFT, TPOT, kuyruk, KV kullanımı</text>
<path d="M236 242 C 262 242, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="292" y="116" width="190" height="88" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="387.0" y="143.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Metrik + trace</text>
<text x="387.0" y="158.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">DCGM exporter →</text>
<text x="387.0" y="172.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Prometheus / OpenTelemetry</text>
<text x="387.0" y="187.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">istek başına tek trace_id</text>
<line x1="482" y1="160" x2="512" y2="160" marker-end="url(#ob-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="514" y="106" width="190" height="108" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="609.0" y="136.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Aksiyon</text>
<text x="609.0" y="150.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">SLO burn-rate alarmı</text>
<text x="609.0" y="165.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">düğüm karantinası</text>
<text x="609.0" y="179.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kapasite planlama</text>
<text x="609.0" y="194.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">faydalı token maliyeti</text>
<text x="16" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DCGM exporter yalnızca GPU alanlarını verir; ağ, uygulama ve kullanıcı deneyimi metrikleri ayrıca enstrümante edilmelidir.</text>
</svg>

NVIDIA **DCGM Exporter**, GPU telemetri alanlarını Prometheus uyumlu biçimde sunabiliyor. Ancak yazılım metriklerini, ağ switch sayaçlarını ve kullanıcı deneyimi metriklerini kendiliğinden üretmez; bunlar ayrıca enstrümante edilmelidir. [K35]

#### Somut bir kapasite kestirimi: Little's Law

Bir inference servisine saniyede ortalama 6 istek geldiğini ve isteklerin sistemde ortalama 8 saniye kaldığını düşünelim. Kararlı durum ve uygun uzun dönem ortalama varsayımlarıyla:

$$
L=\lambda W=6\;\mathrm{request/s}\times8\;\mathrm{s}=48\;\mathrm{request}
$$

Bu, sistemde ortalama **48 eşzamanlı istek** bulunduğunu anlatır; 48 GPU gerektiğini söylemez. Bir GPU worker'ı birden fazla isteği birlikte sunabilir ve iş yüklerinin KV bellek gereksinimleri farklıdır.

**Önemli:** Ortalama eşzamanlılığa göre kapasite satın almak tek başına yeterli değildir. Ani trafik artışı, kuyruk politikası ve p99 gecikme hedefleri için yük testleri ve güvenlik payı gerekir.

---



---

### MFU, GPU Utilization ve gerçek faydayı aynı grafikte hayal edelim

İki eğitim koşusu düşünelim. A koşusu %55 MFU üretiyor ama 10.000 adım sonrasında model kalitesi beklendiğinden düşük kalıyor. B koşusu %45 MFU üretiyor fakat daha iyi sayısal kararlılıkla daha hızlı hedef kayba ulaşıyor. Hangisi daha iyi? Yalnızca MFU'ya bakarsak A deriz; hedef kaliteye kadar geçen zamana bakarsak B daha değerli olabilir.

Bu nedenle üç farklı ölçüm katmanı vardır:

1. **Donanım etkinliği:** GPU meşguliyet yüzdesi, Tensor Core throughput'u, bellek ve ağ kapasite kullanımı.
2. **Model ilerleme etkinliği:** MFU/HFU, step time, eğitim token/s, kayıp eğrisi, hedef kaliteye kadar süre.
3. **Ürün etkinliği:** İstek başına maliyet, p95/p99 TTFT, başarılı çıktı oranı, kullanıcıya faydalı token/s.

Birinci katmanda mükemmel görünen sonuç, ikinci veya üçüncü katmanda aynı başarıya dönüşmeyebilir. MLPerf Training'in hedef kaliteyi koşul olarak kullanmasının amacı da benchmark'ın yalnızca ham işlem yarışına dönüşmesini engellemektir. [K50]

### Throughput iyi olduğu hâlde p99 neden patlar? Küçük bir kuyruk deneyi

Bir gişenin saniyede ortalama 10 kişiye hizmet verdiğini düşünelim. İki farklı talep var: saniyede 8 kişi ve saniyede 9,5 kişi. Basitleştirilmiş M/M/1 kuyruğunda ortalama sistemde kalma süresi:

$$
W=\frac{1}{\mu-\lambda}.
$$

$\mu=10$ için $\lambda=8$ olduğunda $W=0.5$ saniye; $\lambda=9.5$ olduğunda $W=2$ saniye. Trafik yalnızca %18,75 artmasına rağmen ortalama bekleme dört katına çıkmıştır.

Bu, GPU serving sisteminin tam modeli değildir: sürekli batching, değişken token uzunlukları, çoklu worker, KV bellek limiti ve scheduling politikasını dışarıda bırakır. Ama çok önemli bir eğilimi gösterir: **sistem kapasite sınırına yaklaştıkça küçük trafik artışları büyük gecikme patlamaları yaratabilir.** İşte bu nedenle p99, başlı başına bir mimari sinyaldir.

### Ölçümün granülerliği neden önemli?

`nvidia-smi` ile “GPU yoğun çalışıyor” sonucuna ulaşabilirsiniz; fakat “hangi kernel, hangi zaman aralığında, bellek nedeniyle mi yoksa iletişim nedeniyle mi yavaş?” sorusunu yanıtlamak için zaman çizelgesi ve kernel sayaçları gerekir. NVIDIA Nsight Systems sistem genelindeki CUDA/API, GPU, ağ ve CPU olaylarını birlikte gösterir. Nsight Compute ise kernel düzeyindeki sayaçları incelemeye yardımcı olur. [K47]

İyi bir teknik açıklamada her iddianın yanında örtük bir **kanıt seviyesi** olmalı:

| İddia | Yeterli olabilir mi? | Neden? |
|---|---|---|
| “GPU aktif” | GPU utilization | Kabaca etkinlik belirtir |
| “Tensor Core etkinliği düşük” | Kernel profili | Gerçek yürütme birimine bakar |
| “Ağ sınırına çarpıyoruz” | NCCL zamanı + NIC/fabric ölçümleri | Bekleyişin iletişime bağlı olduğunu gösterir |
| “Depolama eğitimi durduruyor” | Checkpoint timeline + storage telemetry | I/O süresi ile pause süresini ayırır |
| “Daha iyi bir sistem” | Kalite + SLO + maliyet + tekrarlanmış ölçüm | Teknolojiyi kullanıcıya etkisiyle ilişkilendirir |


## 20. Kaynaklarla doğrulanmış büyük ölçek hikâyeleri

Teori çok şey öğretir, ama büyük ölçekli sistemleri tasarlarken raporlanmış gerçek deneylere de bakmak gerekir. Aşağıdaki iki örnek, aynı soruna iki farklı yaklaşımı gösteriyor: Çok büyük dense model eğitiminde sistem bütünlüğünü korumak ve MoE modelinde iletişim ile hesaplamayı akıllıca örtüştürmek.

### 20.1. Meta Llama 3 405B: Çok sayıda GPU'da dengeli ilerleme

Meta'nın teknik raporu 405 milyar parametreli modelin 16.384 H100 üzerinde farklı bağlam uzunluklarında eğitildiğini gösterir. Örnek yapılandırmalar içinde 8K context'te yaklaşık **%41 MFU**, 128K context'te yaklaşık **%38 MFU** raporlanmıştır. Sistem, **Grand Teton** tabanlı sunucular, **RoCE** compute ağı, **Tectonic** depolama ve otomatik kurtarma mekanizmalarından yararlanmıştır. [K5]

Bu örnek bize tek bir yöntemden değil, **birlikte çalışan mühendislik kararlarından** oluşan başarıyı anlatır: model parçalaması, ağ topolojisi, checkpoint ve arıza otomasyonu aynı büyük hedefe hizmet eder.

### 20.2. DeepSeek-V3: MoE'de hesaplamayı ağ iletişiminin üzerine bindirmek

MoE modellerinde bir token'ın seçilen expert'e ulaşması için ağda hareket etmesi gerekebilir. Expert'lerin kendisi küçük olduğunda compute miktarı ile token dispatch iletişimi birbirine yaklaşabilir.

DeepSeek-V3 raporu, **DualPipe** ile forward/backward hesaplamaları, AllToAll dispatch/combine ve pipeline trafiğini örtüştürmeyi ele alır. Raporda 671B toplam parametre, token başına 37B etkin parametreli MoE ve 2.788 milyon H800 GPU-saatlik eğitim maliyeti bilgisi vardır. Bunlar modelin eğitim düzenine bağlı raporlanmış sonuçlardır; farklı kümeye doğrudan aktarılacak sabit süreler değildir. [K9]

Raporun eğitim düzeni somuttur: 16 yönlü PP, 8 düğüme yayılan 64 yönlü EP ve ZeRO-1 DP; maliyetli TP hiç kullanılmaz. DualPipe microbatch'leri pipeline'ın iki ucundan birden besler ve her parçada ileri/geri hesabı dispatch/combine iletişimiyle örtüştürür; rapor, ince taneli expert'lerle bile düğümler arası AllToAll yükünün böylece **sıfıra yakın** kaldığını belirtir. Global batch 15.360 dizi × 4.096 token ≈ 62,9 milyon token olduğundan 2.048 H800'ün her biri adım başına ~30.720 token işler: MoE'nin büyüttüğü kritik batch eşiğine (§13.6) böyle yaklaşılır. [K9]

**Öğretici karşıtlık:** Dense modelde büyük GEMM iyi çalışır ama ağırlık boyutu ağırdır. MoE aktif hesaplama miktarını azaltabilir ancak expert routing ve AllToAll ağ sorununu güçlendirebilir. Bir darboğazı azaltırken başka bir darboğaz doğar.

---


### 20.3. Lambda ve MegaScale: Aynı GPU'yu daha iyi kullanmak ne kadar fark yaratır?

Meta örneği bize güçlü bir büyük ölçek baz çizgisi verdi. Şimdi ikinci tür başarıya bakalım: Donanımı değiştirmeden, aynı modelde daha iyi çalışma düzeni kurmak.

Lambda'nın **Kasım 2025** tarihli raporu, Llama 3.1'in 8B/70B/405B varyantlarını **HGX B200** ve **GB300 NVL72** ortamlarında karşılaştırıyor. BF16 ve 8K sequence ile 70B modelin MFU'su varsayılan TorchTitan ayarlarında %23,83 iken düzenlenmiş eğitim reçetesiyle %50,20'ye çıkıyor. 405B modelin 128× GB300 ölçümünde de %28,62'den %52,69'a gelişme raporlanıyor. Lambda, **8B/32K** senaryosunda **%60,48** ölçüyor. [K12]

Bu sonuçların en öğretici bölümü yüksek rakamlar değil, **önce neden yavaş olduğunu açıklayabilmeleri**: Küçük mikro-batch, fazla TP, yoğun activation checkpointing, bekleyen FSDP AllGather ve gereksiz pointwise/optimizer kernel maliyeti. Çözüm daha fazla GPU satın almak yerine bu maliyetleri hedefli olarak azaltmak.

MegaScale ise başka bir soruyu yanıtlıyor: *Bu optimizasyonlar on binlerce GPU'ya ulaştığında da işe yarıyor mu?* NSDI 2024'te yayımlanan çalışma, **175B model ve 12.288 GPU** için **%55,2 MFU**, Megatron-LM karşısında **1,34×** iyileşme raporluyor. Hesaplama–iletişim overlap, operatör optimizasyonu, veri hattı ve straggler teşhisi birlikte ele alınıyor. [K13]

Burada sonuç açık: **MFU yalnızca GPU modelinin özelliği değil, bütün sistemin ortak tasarımının çıktısıdır.** Aynı donanım üzerinde yazılım değişikliği iki katı aşkın fark oluşturabilir; yüksek ölçek de tek başına %40'ın üzerine çıkmayı yasaklamaz.


## 21. Beş büyük yanlış genelleme ve bunların daha doğru halleri

Bu noktada ezberlemesi kolay ama mühendislik kararını yanlış yönlendiren bazı cümleleri de açıkça ayırmak istiyorum. Çünkü teknik sunumlarda çarpıcı ifadeler çok akılda kalır; fakat yanlış genelleme yapıldığında yanlış donanım yatırımlarına veya yanlış optimizasyonlara yol açabilir.

| Hatalı veya aşırı kesin iddia | Daha doğru ifade |
|---|---|
| **'H100 decode'da MFU kesin %1–2'dir.'** | Tek-token GEMV'nin teorik aritmetik yoğunluğu çok düşük olabilir; gerçek MFU, model FLOP hesabı, concurrency ve diğer kernel'lardan etkilenir. |
| **'TP sekiz GPU sınırını asla aşamaz.'** | Sekizli NVLink alanı H100 DGX için güçlü yerleşimdir; TP'nin evrensel fiziksel üst sınırı değildir. |
| **'SHARP her AllReduce'u tam iki kat hızlandırır.'** | Desteklenen toplama operasyonlarında trafik ve gecikme azalabilir; uçtan uca fayda koşulludur. |
| **'GDS CPU'yu tamamen ortadan kaldırır.'** | Belirli veri kopyalarını atlayabilir; CPU I/O kontrolü, metadata ve uyumluluk yolu varlığını sürdürür. |
| **'7,5 TB checkpoint bir saniyede yazılırsa her şey çözülür.'** | Sonuca ulaşmak için etkin disk/ağ kapasitesi, checkpoint formatı, dayanıklılık garantisi ve job pause süresi gerekir. |

Ek iki önemli düzeltme:

- **405B state = 7,5 TB** evrensel formül değildir. Sayı, ağırlık/gradient/optimizer/master ağırlık/snapshot içeriklerine göre değişir. 16 byte/parametre varsayımında **6,48 TB** temel model durumu çıkar.
- **419 / 466 duraklama** birbirinin alternatifi değildir. Llama 3 raporunda 466 toplam olayın 419'u **beklenmedik** olarak kaydedilmiştir. [K5]

---


## 22. Gözlemlenebilirlik: Metrikler ve izler bir sistem hakkında ne söyler?

Buradaki örnek olaylar bir teknik destek prosedürü değildir; aynı belirtiye yol açabilecek farklı mekanizmaları karşılaştırmak için hazırlanmıştır.

Bir GPU cluster'ın sağlığına tek pencereden bakılamaz. **Telemetry** size "neler oluyor" sorusunu yanıtlar; **profiling** kernel seviyesinde "neden oluyor" sorusuna yaklaşır; **tracing** bir kullanıcı isteğinin veya eğitim step'inin hizmetler arasında nasıl dolaştığını gösterir.

### 22.1. Üç katmanlı görünürlük

| Katman | Araç örneği | Cevap verdiği soru |
|---|---|---|
| Donanım | DCGM, NVML, NIC counters | GPU ısınıyor mu; HBM/NVLink/PCIe ne durumda? |
| Runtime / kernel | Nsight Systems, Nsight Compute, PyTorch Profiler | Hangi kernel, launch gap veya collective zaman harcıyor? |
| Hizmet | Prometheus, OpenTelemetry, distributed trace | Bu isteğin TTFT'si neden 3 saniye arttı? |

DCGM'ın sampling profili her kernel'ı ayrıştırmaz; Nsight Systems trace'i de her koşulda düşük overhead'li sürekli üretim metriği olarak kullanılmaz. **Sürekli telemetri** ile **hedefli profil incelemesi** farklı araçlardır. [K36][K37]

### 22.2. Bir inference trace'inin asgari alanları

```text
trace_id, request_id, tenant_id_hash, model_version, runtime_version,
prompt_token_count, output_token_count, image_count, context_length,
auth_ms, safety_ms, data_access_ms, queue_ms, prefill_ms,
first_decode_ms, inter_token_p95_ms, e2e_ms,
cache_hit, gpu_pool, worker_id, result_status, cancel_reason
```

Bu, ölçümleri yapılandırmak için **örnek şema**dır. Tenant kimliği ve request metadata'sı dahi kişisel veya hassas bilgi taşıyabileceğinden kurumun veri minimizasyonu ve saklama politikasına uyarlanmalıdır.

### 22.3. Teşhis mantığı: Aynı belirti hangi katmandan kaynaklanabilir?

<svg viewBox="0 0 720 364" role="img" aria-label="Kök neden teşhis ağacı. SLO veya MFU&#x27;nun düştüğü belirtisinden başlayın. Önce hata oranı arttı mı diye sorun: Xid, ECC, yeniden başlatma. Evetse bunu donanım veya sürücü arızası sayıp düğümü karantinaya alın. Hayırsa profil zaman çizelgesinde beklemenin yerini bulun: kuyruktaysa kabul politikası ve havuz kapasitesi; GPU hesabındaysa kernel belleği ve batch boyutu; ağdaysa NCCL kolektifleri ve NIC sayaçları; depolama ve I/O&#x27;daysa dataloader ve checkpoint yazımı. Ardından tek değişkenle tek hipotezi test edin." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dt-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dt-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Kök neden teşhisi: bir şey bozuk mu, yoksa bekliyor mu?</text>
<rect x="16" y="128" width="150" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="91.0" y="154.0" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SLO / MFU düştü</text>
<text x="91.0" y="169.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">belirti</text>
<rect x="212" y="52" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="307.0" y="78.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Hata oranı arttı mı?</text>
<text x="307.0" y="92.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Xid, ECC, yeniden başlatma</text>
<rect x="212" y="204" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="307.0" y="230.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Bekleme nerede?</text>
<text x="307.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">profil zaman çizelgesi</text>
<path d="M166 146 C 190 146, 190 80, 210 80" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M166 166 C 190 166, 190 232, 210 232" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="52" width="254" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="577.0" y="78.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Donanım / sürücü arızası</text>
<text x="577.0" y="92.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">düğümü karantinaya al (§17.5)</text>
<line x1="402" y1="80" x2="448" y2="80" marker-end="url(#dt-b)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<text x="425" y="74" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px;font-weight:600">evet</text>
<path d="M307 108 V198" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="315" y="160" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">hayır</text>
<rect x="450" y="136" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="154.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Kuyruk</text>
<text x="577.0" y="169.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kabul politikası, havuz kapasitesi</text>
<path d="M402 232 C 426 232, 426 157, 448 157" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="186" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="577.0" y="204.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU hesap</text>
<text x="577.0" y="219.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kernel belleği, batch boyutu</text>
<path d="M402 232 C 426 232, 426 207, 448 207" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="236" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="577.0" y="254.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Ağ</text>
<text x="577.0" y="269.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NCCL kolektifleri, NIC sayaçları</text>
<path d="M402 232 C 426 232, 426 257, 448 257" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="286" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="577.0" y="304.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Depolama / I/O</text>
<text x="577.0" y="319.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">dataloader, checkpoint yazımı</text>
<path d="M402 232 C 426 232, 426 307, 448 307" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Sonra her seferinde aynı disiplin: tek hipotez, tek değişken, önce/sonra ölçümü.</text>
</svg>

### 22.4. Örnek olay A: MFU %38’den %22’ye neden düşebilir?

**İlk kontrol:** Aynı model mi, aynı batch ve sequence length mi, aynı precision mı, aynı GPU sayısı mı? MFU'nun paydası değişmiş olabilir. Yeni bir kütüphane veya ölçüm hatası da farkın kaynağı olabilir.

**İkinci kontrol:** Profiler timeline'da eğitim adımlarının compute, network, dataloader, checkpoint ve optimizer fazlarını ayırın.

**Üçüncü kontrol:** Bir grup GPU geride kalıyorsa straggler node tespiti yapın. Eğer bütün GPU'lar benzer şekilde yavaşsa ortak network/storage veya yazılım sürümü etkisini düşünün.

**Dördüncü kontrol:** Önce son sağlıklı image/config kombinasyonuna A/B dönerek regresyonu doğrulayın. Hiçbir değişkeni kaydetmeden birden fazla NCCL tuning knob'unu değiştirmeyin.

### 22.5. Örnek olay B: `nvidia-smi` %100 iken token/s neden düşük olabilir?

**Olası nedenler:** B=1 decode'un bellek sınırlı olması, batch'in küçük kalması, KV cache baskısı, gereksiz token preprocessing veya quantization kernel'ının verimsizliği. GPU utilization tek başına karar verdirmez. İş yükünü kısa/uzun prompt ve output boyunca ayırın; output token/s, TPOT, HBM throughput ve profiling'i aynı zaman diliminde ölçün.

### 22.6. Örnek olay C: NCCL `all_reduce` neden takılabilir?

Önce uygulama ilerlemesinin gerçekten durup durmadığını ve hangi rank'ların eksik olduğunu doğrulayın. NCCL'nin **RAS** altyapısı, sürüme bağlı olarak aktif iş durumunu ve yanıt vermeyen rank'ları gösterebilir. Yeni sürümlerde aktif diagnostics de bulunur; pasif RAS, veri yolunu otomatik olarak test etmez. [K38]

Sonra şu sırayı izleyin: rank-env sürüm tutarlılığı → NIC/GPU görünürlüğü → doğru network interface → GPU–NIC topolojisi → fiziksel link/counter → iki düğümlü nccl-tests → rack ölçeği. Bu yol, sorunun parametrede mi, ağ yolunda mı yoksa belirli node'da mı olduğunu ayırmaya yardımcı olur.

```bash
# Salt-okunur başlangıç kontrolü. Tam komutlar kurulum sürümüne göre değişebilir.
nvidia-smi -L
nvidia-smi topo -m
# nccl-tests ortamı varsa tek node AllReduce sweep
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

NCCL RAS endpoint'ini kurum ağına kontrolsüz şekilde açmayın; tanılama bilgilerini yetkili ağ/host sınırında tutun.

### 22.7. Örnek olay D: Ortalama iyi görünürken p99 TTFT neden kötüleşir?

Kuyruğun ani patlamalarını, bekleyen token sayısını, soğuk model yüklemelerini, en uzun prompt'ları, retrieval service timeout'larını ve prefix cache hit oranını karşılaştırın. p99'u "ortalama sıcaklığı" ölçer gibi ortalamayla tedavi edemezsiniz. İstek uzunluğu kırılımı ve tenant başına trafik sınıfları gerekir.

### 22.8. Örnek olay E: Checkpoint yazma süresi neden dakikalara uzar?

Önce yazılan **gerçek byte** miktarını, shard sayısını ve paralel writer sayısını ölçün. Sonra dosya sistemi metadata darboğazı, network oversubscription, storage server load, eşzamanlı işler ve checkpoint sıkıştırma davranışını inceleyin. GDS'nin etkin olup olmadığını doğrulamak yararlı olabilir ama sorunu otomatik çözdüğü varsayılmamalıdır.

### 22.9. Bir performans deneyini yorumlarken altı önemli ayrım

1. Baseline ölçümünü saklayın; sürümler ve konfigürasyon hash'ini kaydedin.
2. Aynı workload trace ve mümkünse aynı random seed'i kullanın.
3. Aynı anda yalnızca sınırlı sayıda değişkeni değiştirin; mümkünse tek hipotez.
4. Warmup, cache durumu ve arka plan trafiğini sabitleyin veya raporlayın.
5. Ortalama ile birlikte p95/p99 ve confidence/tekrar ölçümlerini karşılaştırın.
6. Kazancı kalite, güvenlik ve maliyet regresyon testinden geçirin.

---


## 23. FinOps ve sürdürülebilir kapasite: MFU yüksek ama neden fatura da yüksek?

Bir GPU sisteminin işletme başarısı, `nvidia-smi`'da %100 görmeniz değildir. Günde 24 saat çalışan bir altyapı, müşteriye anlamlı sonuç üretmiyorsa aslında pahalı bir elektrikli ısıtıcıya dönüşebilir.

### 23.1. Toplam maliyet neleri kapsar?

**TCO (Total Cost of Ownership)**, yalnızca GPU sunucu faturası değildir. Donanım amortismanı veya bulut kira bedeli, NIC/switch, kablolama, rack, elektrik, soğutma, depolama, lisans, bakım, yedek parça, internet/backbone ve ekip operasyon maliyetleri birlikte ele alınır.

Basitleştirilmiş yıllık model:

$$
C_{\mathrm{annual}}=C_{\mathrm{capex/amort}}+C_{\mathrm{power}}+C_{\mathrm{network}}+C_{\mathrm{storage}}+C_{\mathrm{software}}+C_{\mathrm{ops}}
$$

Elektrik için:

$$
E_{\mathrm{annual}}\text{ (kWh)}\approx P_{\mathrm{IT}}\text{ (kW)}\times PUE\times8760
$$

Bu, yıl boyunca ortalama güç sabitmiş gibi yapılan bir yaklaşımdır. Gerçekte dinamik power cap, idle güç, mevsimsel PUE ve çalışma oranı hesaba katılmalıdır.

### 23.2. GPU-saat yerine hangi maliyeti izlemeli?

İki modeli düşünün:

- Model A çok hızlı ama maliyeti yüksek; yüksek kalite, düşük gecikme sunuyor.
- Model B daha küçük ve ucuz; çoğu kolay soruyu yeterli kalitede yanıtlıyor.

Yalnızca `$/GPU-hour` kıyaslamak yetmez. **$/1M başarılı çıktı token**, **$/1000 SLO uyumlu istek**, **$/hedef kaliteye erişmiş training run** gibi metrikler iş kararıyla daha uyumludur. Bir modelden yüzde beş daha yüksek throughput almak için yüzde yirmi daha fazla enerji yakıyor ve başarısız istek oranı artıyorsanız kazanım tartışmalıdır.

### 23.3. Model routing ekonomik bir mimari kararıdır

Her isteği en büyük modele göndermek yerine kalite değerlendirmesinden geçmiş küçük/büyük model routing stratejileri toplam maliyeti azaltabilir. Fakat sınıflandırıcı/route kararının kendisinin gecikme ve hata maliyeti vardır; yanlış yönlendirme müşteri kalitesini düşürebilir. Bu yüzden **routing politikası kalite kapısının bir parçası** olmalıdır.

### 23.4. On-prem mi cloud mu? Kestirme cevap yok

On-prem'de yüksek sabit başlangıç maliyeti, uzun kapasite planlama döngüsü ve veri egemenliği/operasyon kontrolü olabilir. Cloud'da daha hızlı deneme ve geçici kapasite avantajı bulunabilir, ama egress, rezervasyon ve sürekli GPU kiralamanın maliyeti önemlidir. En doğru yöntem:

1. İş yükünü sürekli/periodik/ani patlama olarak sınıflandırın.
2. 12–36 aylık gerçekçi kullanım tahmini yapın.
3. On-prem ve cloud senaryolarında aynı hizmet SLO ve kaliteyi baz alın.
4. Network, depolama, destek, ekip, veri transferi ve arıza rezervini hesaba katın.
5. Doluluk oranı varsayımını değiştirerek duyarlılık analizi yapın.

**Mimari ders:** %80 doluluk varsayımıyla avantajlı görünen bir satın alma, gerçek doluluk %20'de kalırsa dezavantajlı olabilir. Belirsizliği tek bir rakama gizlemek yerine en iyi, temel ve kötü senaryoları birlikte göstermeliyiz.

### 23.5. Carbon-aware ve power-aware scheduling

İşin esneklik durumuna göre toplu işlerin düşük enerji fiyatı veya uygun elektrik/soğutma zamanlarında çalıştırılması mümkün olabilir. Ancak bunu eğitim kalitesi, teslim süresi, veri mevzuatı ve ek veri taşıma maliyetiyle birlikte değerlendirmek gerekir. **Daha az watt** ile **daha az kWh/iş** aynı şey değildir: hızlanan iş, anlık güç yükselse bile daha az toplam enerji harcayabilir.

---




---

**KISIM VII — Platformun bütünü: güvenlikten veri ürünlerine**

*Artık GPU kablolarından daha geniş bir resimdeyiz: iş yükleri, güvenlik sınırları, kullanıcı deneyimi ve ekonomi.*


## 24. AI Architect bakış açısı: Teknik parçalar aynı sistemde nasıl birleşir?

Şimdi bir düşünce deneyi yapalım. Bir şirket sizden "1.024 H100 GPU'luk bir AI platformu tasarla" dedi. Elinizde teknik özellikler, network portları ve GPU fiyatları var. Hemen rack yerleşimi mi çizersiniz?

Ben çizmezdim. Önce beş soru sorardım: **Bu platform hangi işi yapacak? Başarı hangi metrikle ölçülecek? Veri ve model kimlere ait? Ne kadar kesintiyi kaldırabiliriz? Bütün bunların toplam maliyet sınırı nedir?** Çünkü bunlar yanıtlanmadan çizilen mimari, güzel bir kablo diyagramı olabilir ama doğru bir AI platformu olmak zorunda değildir.

### 24.1. Önce gereksinim: Aynı GPU'lar, üç farklı ürün

| Gereksinim | Büyük model pretraining | Fine-tuning / araştırma | Kurumsal online inference |
|---|---|---|---|
| Birincil çıktı | Hedef kaliteye ulaşmış checkpoint | Deney sonucu ve özelleştirilmiş model | Güvenli ve zamanında yanıt |
| Ana performans metriği | Time-to-quality, tokens/s | Deney çevrim süresi, maliyet | TTFT/TPOT, goodput, p95/p99 |
| Kaynak stratejisi | Homojen, topolojiye yakın, gang scheduling | Esnek kota, küçük iş kuyruğu | Her zaman hazır kapasite, admission control |
| Kritik bellek kalemi | Weights, gradients, optimizer, activations | Model + optimizer/adapters | Weights + KV cache + geçici buffer |
| Kritik hata senaryosu | Büyük eğitimin yeniden başlaması | Deney kaybı, tekrar edilememe | Canlı kullanıcı isteğinin kesilmesi |
| Başlıca kontrol | Checkpoint + health/RAS | Deney izleme + veri/model sürümü | Rate limit, failover, tenant isolation |

Bu tablo bize neden tek bir "GPU utilization" göstergesinin bütün ürünleri yönetmek için yetmediğini gösterir.

### 24.2. Control plane ile data plane'i neden ayırıyoruz?

Havaalanını düşünün. Bir tarafta hangi uçağın hangi kapıdan kalkacağını belirleyen kontrol odası vardır; diğer tarafta yolcuları ve bagajı fiilen taşıyan operasyon. Kontrol odası yolcuları tek tek omzunda taşımaz; sistemi yönetir.

**Control plane (kontrol düzlemi)**, kaynakların kime, ne zaman, hangi politika ve yapılandırmayla tahsis edileceğini belirler. Scheduler, iş kuyruğu, model registry, izinler, deployment politikaları ve rollout kontrolü bu sınıfa girer.

**Data plane (veri/işleme düzlemi)**, GPU üzerinde gerçek eğitim veya inference hesaplamalarının yürütüldüğü, KV bloklarının aktarıldığı ve kullanıcı yanıtlarının üretildiği çalışma yoludur.

Bu iki alanın ayrılması, "her istek kontrol sistemini dolaşmalı" anlamına gelmez; kontrol düzlemi kararları uygular, data plane gecikmeye duyarlı hesaplama yolunu yürütür.

<svg viewBox="0 0 720 318" role="img" aria-label="Üç düzlem. Üstteki control plane IAM ve politika, scheduler ve kuyruk, model registry, rollout kontrolü ve FinOps&#x27;u barındırır; her isteğin yolunda durmadan yerleşim, kota, yönlendirme politikası ve rollout kararlarını verir. Training data plane veri seti shard&#x27;larını FSDP ve NCCL çalıştıran GPU kümesinden checkpoint deposuna taşır; throughput odaklıdır, saatlerden haftalara sürer. Serving data plane istekleri gateway ve router, prefill havuzu ve decode havuzu üzerinden kullanıcıya akıtır; gecikme odaklıdır, milisaniyelerle ölçülür. İki data plane&#x27;in telemetrisi control plane&#x27;e geri akar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="pl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Control plane karar verir, iki data plane işi yapar</text>
<rect x="16" y="40" width="688" height="74" rx="10" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">Control plane · karar, kota ve orkestrasyon</text>
<rect x="28" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="91.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">IAM &amp; policy</text>
<rect x="162" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="225.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Scheduler / queue</text>
<rect x="296" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="359.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Model registry</text>
<rect x="430" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="493.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Rollout kontrolü</text>
<rect x="564" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="627.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">FinOps &amp; maliyet</text>
<path d="M190 114 V140" marker-end="url(#pl-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M530 114 V140" marker-end="url(#pl-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="198" y="132" text-anchor="start" style="fill:var(--c-accent-2);font-size:10.5px">yerleşim, kota</text>
<text x="538" y="132" text-anchor="start" style="fill:var(--c-accent-2);font-size:10.5px">yönlendirme politikası, rollout</text>
<rect x="16" y="142" width="338" height="132" rx="10" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-warn);font-size:12.5px;font-weight:700">Training data plane</text>
<rect x="28" y="174" width="98" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="77.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Veri seti</text>
<text x="77.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">shard&#x27;lar</text>
<rect x="140" y="174" width="100" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="190.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU kümesi</text>
<text x="190.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">FSDP / NCCL</text>
<rect x="254" y="174" width="88" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="298.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Checkpoint</text>
<text x="298.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">deposu</text>
<line x1="126" y1="201" x2="138" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="240" y1="201" x2="252" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="28" y="254" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">throughput odaklı · saatler–haftalar</text>
<rect x="366" y="142" width="338" height="132" rx="10" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="378" y="162" text-anchor="start" style="fill:var(--c-success);font-size:12.5px;font-weight:700">Serving data plane</text>
<rect x="378" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="413.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Gateway</text>
<text x="413.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">router</text>
<rect x="458" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="493.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Prefill</text>
<text x="493.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">havuz</text>
<line x1="448" y1="201" x2="457" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="538" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="573.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Decode</text>
<text x="573.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">havuz</text>
<line x1="528" y1="201" x2="537" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="618" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="653.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Kullanıcı</text>
<text x="653.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">stream</text>
<line x1="608" y1="201" x2="617" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="378" y="254" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">gecikme odaklı · milisaniyeler</text>
<path d="M190 274 V292 H530 V276" marker-end="url(#pl-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="360" y="306" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">telemetri control plane&#x27;e geri akar</text>
</svg>

Bu yaklaşım, NVIDIA Dynamo gibi sistemlerdeki worker, router ve yönetim bileşenlerinden de esinlenen **öğretici bir referans şemasıdır**; belirli bir ürünün birebir kurulum kılavuzu değildir. [K20]

Küçük sistemlerde prefill ve decode aynı worker üzerinde birleşebilir; büyük eğitim sistemlerinde Slurm, Kubernetes veya ayrı kümeler kullanılabilir. Kimlik/politika katmanının somut uygulanması da kurumsal güvenlik modeline bağlıdır.

### 24.3. 1.024 H100 için öğretici bir yerleşim hesabı

128 adet sekiz GPU'lu sunucumuz olduğunu varsayalım. Büyük bir dense modeli eğitirken **TP=8, PP=8, DP=16** seçtiğimizi düşünelim:

$$
N_{\mathrm{GPU}}=8\times8\times16=1024
$$

- **TP=8:** Aynı katmanın matris parçalarını sekiz GPU'lu tek sunucu içinde tutmaya çalışırız; böylece yüksek hızlı scale-up bağlantısından faydalanırız.
- **PP=8:** Her model kopyasının katmanlarını sekiz pipeline aşamasına ayırırız. Bu aşamalar birden fazla sunucuya yayılabilir.
- **DP=16:** Aynı eğitim işini farklı batch dilimleri üzerinde çalışan 16 model-paralel grup olarak çoğaltırız; gradyan stratejisiyle tutarlılığı sağlarız.

Bu yalnızca **GPU sayısına uyan bir topoloji örneğidir**. Gerçekte katman sayılarının PP aşamalarına dengeli paylaştırılması, aktivasyon boyutları, CP ihtiyacı, network oversubscription, optimizer durumu, checkpoint ve HBM kapasitesi doğrulanmadan bu düzenin en hızlı veya hatta doğrudan çalıştırılabilir seçenek olduğu söylenemez.

Örneğin yalnızca BF16 ağırlıkları 405B yoğun model için yaklaşık 810 GB tutar. TP×PP=64 model shard'a ideal/eşit bölündüğünü varsayarsak ağırlık başına **yaklaşık 12,7 GB/GPU** düşer. Ama optimizer durumları, gradyanlar ve aktivasyonlar bu hesaba henüz girmedi. "Modelin ağırlığı sığdı, training kesin çalışır" sonucunu buradan çıkaramayız.

**Topoloji kararı:** Aynı rakamdaki GPU'ları rastgele node'lara dağıtmak yerine TP gruplarını hızlı yerel bağlantıda, PP geçişlerini uygun fabric üzerinde ve DP gruplarını ağ kapasitesi ile fault-domain risklerini dengeleyerek yerleştirmeliyiz.

### 24.4. Güvenlik ve yönetişim performanstan ayrı değildir

Kurumsal AI'da bir kullanıcının modelden yanıt alabilmesi, yalnızca GPU'nun hazır olduğu anlamına gelmez. İsteğin yetkili kullanıcıdan geldiğini, izin verilen modele yönlendirildiğini ve gerekli veri erişim sınırlarını aşmadığını da bilmeliyiz.

**Model registry**, hangi model sürümünün onaylı olduğunu ve nerede yayımlandığını izler. **RBAC (Role-Based Access Control)**, kullanıcıların ve servislerin rollerine göre erişim sınırlarını belirler. **Secrets management**, API anahtarları ve erişim belirteçlerini uygulama koduna gömmeden saklama ve döndürme pratiğidir.

Kurumsal bir data plane tasarımında şu korumaları netleştirmek gerekir:

- **Model ve veri kökeni:** Hangi kaynak, lisans ve değerlendirme sonrasında bu model kullanılabilir hale geldi?
- **Yetkilendirme:** Kullanıcı hangi modele, tenant'a ve veri ürününe erişebilir?
- **Veri minimizasyonu:** Gereksiz kişisel veya gizli veriyi prompt'a ekliyor muyuz?
- **İzolasyon:** Cache, request log, debug dump, GPU belleği, depolama ve model artefact sınırları nasıl ayrılıyor?
- **Gözlemlenebilirlik ile mahremiyet:** Olayı araştırmak için gereken metadata tutulurken hassas ham içerik gereksiz yere loglanıyor mu?
- **Tedarik zinciri:** Container image, model ağırlığı, tokenizer ve bağımlılıklar sürümlü/doğrulanabilir mi?

Güvenliği tek bir DLP kontrolüne indirgememek gerekir. Prompt injection, hassas veri sızıntısı, model çıktı riski, eğitim verisi hakları ve kurumsal erişim kontrolü farklı güvenlik katmanlarıdır. NIST'in Generative AI Profile yaklaşımı, veri kaynağı ve mahremiyet riskleri dahil olmak üzere bu alanların yaşam döngüsü boyunca yönetilmesini önerir. [K18]

#### AI Gateway, AI Safety ve veri erişimini karıştırmayalım

Bir kurumsal inference isteğini havaalanındaki yolcu gibi hayal edebilirsiniz: Önce kimliği ve yetkisi doğrulanır, sonra güvenlik politikasına uygunluğu kontrol edilir, ardından doğru modele yönlendirilir. Gerektiğinde kurumsal veriye erişilir; **her isteğin mutlaka bir veri sorgusu yapması gerekmez**.

- **API Gateway / erişim katmanı:** Kimlik doğrulama, yetkilendirme, kota, hız sınırlama ve hizmet yönlendirme kararlarını uygular.
- **AI Safety katmanı:** Girdi/çıktı politikaları, hassas veri sızıntısı kontrolleri, içerik güvenliği ve prompt-injection riskini azaltmaya yönelik kontrolleri uygular. Hiçbir tekil filtre tam güvenlik garantisi vermez.
- **Inference engine:** Modeli çalıştırır; batching, KV cache, prefill/decode, tensor parallelism gibi performans sorumluluklarını üstlenir.
- **Data Product Gateway / veri hizmeti:** Yalnızca ihtiyaç duyulduğunda, kurumsal veriye yetkili, gözlemlenebilir ve denetlenebilir erişim sağlar. Bu katman AI Gateway'in alt bileşeni olarak düşünülmek zorunda değildir.

Bu ayrım bir de **gecikme bütçesi** meselesidir. Örneğin uçtan uca TTFT hedefi belirlenirken giriş kontrolü, güvenlik taraması, gerekiyorsa veri sorgusu, kuyruk ve prefill süreleri birlikte ölçülmelidir. Güvenlik kontrollerinin hizmet veremediği durumda isteğin engellenmesi mi, sınırlandırılmış biçimde sürmesi mi gerektiği risk sınıfına göre önceden belirlenmelidir; erişim politikasını otomatik olarak esnetmek güvenli bir varsayılan değildir.

**Veri Product Gateway / data service notu:** Kurumsal sistemlerde modelin veya agent'ın ihtiyaç duyduğu veriye yetkili ve denetlenebilir erişimi sağlamak, model inference'ını çalıştırmaktan ayrı bir mimari sorumluluktur. Bu veri servisi yalnızca gerektiğinde çağrılabilir; her inference isteği mutlaka bir veri kaynağına gitmez. Bu ayrım hem erişim denetimini hem de gecikme bütçesini netleştirir.

### 24.5. Bütçe: GPU fiyatı değil, toplam sahip olma maliyeti

Bölüm 23.1'de modellediğimiz TCO yapısı, mimari kararların finansal temelidir. Bir AI mimarı için asıl görev, bu toplam maliyet havuzunu iş hedefleriyle ilişkilendirilmiş **birim metriklere** dönüştürmektir:

- **1 milyon token başına maliyet:** Yalnızca üretilen ham token hacmine değil; SLO sınırları içinde başarıyla teslim edilen *faydalı goodput* token'larına bölünerek hesaplanmalıdır.
- **Make-or-buy (on-prem vs bulut) kırılımı:** Yıllık ortalama kullanım oranı (% utilization) %60'ın altındaysa bulut elastikiyeti; %75'in üzerinde ve öngörülebilir iş yüklerinde ise on-prem altyapı TCO açısından avantajlı hale gelir.
- **SLA ihlal maliyeti:** Yetersiz yedeklilik veya darboğazlar nedeniyle yaşanan kesintilerin ve p99 gecikme patlamalarının kurumsal maliyeti, tasarruf edilen GPU bütçesini hızla aşabilir.

Bu nedenle bütçe tartışması "Hangi GPU daha ucuz?" sorusundan çıkıp "Hangi mimari, taahhüt edilen SLO altında 1 milyon faydalı token'ı en düşük birim maliyetle üretir?" sorusuna odaklanmalıdır.


### 24.6. Mimari tercihleri anlamak: Hangi ölçümler hangi sorulara cevap verir?

Bir mimari diyagram, bileşenlerin ilişkisini gösterir; fakat performans ve güvenilirlik hakkında tek başına kanıt oluşturmaz. Bu yüzden mimari çalışmalarda farklı ölçüm düzeylerinin hangi soruya cevap verdiğini görmek yararlıdır. Aşağıdaki tablo, örnek bir değerlendirme akışını açıklar; evrensel bir devreye alma prosedürü değildir.

| İnceleme boyutu | Soru | Örnek gözlem veya kanıt |
|---|---|---|
| **1. Workload tanımı** | Hangi model, veri, SLA/SLO? | ISL/OSL, concurrency, model/dataset sürümleri, kalite hedefi |
| **2. GPU mikrobenchmark** | İlgili GEMM, attention ve HBM davranışı ne? | Roofline yorumuyla ölçülmüş kernel profili |
| **3. Fabric doğrulama** | NIC ve collective yolları beklenen şekilde mi? | `nccl-tests`, RDMA/latency ve topoloji raporu |
| **4. I/O ve checkpoint** | Storage burst ve recovery yeterli mi? | Gerçek veri boyutunda save/load ve restart tatbikatı |
| **5. Küçük ölçek pilot** | Model doğru ilerliyor mu? | Loss/kalite ve time-to-quality tahminleri |
| **6. Kademeli ölçekleme** | 8 → 64 → 256 → 1024 GPU eğrisi nasıl? | Strong/weak scaling, straggler, MFU, toplam maliyet |
| **7. Yük ve dayanıklılık testi** | Arıza, trafik artışı, node kaybında ne oluyor? | SLO uyumu, hata bütçesi, geçiş ve kurtarma süresi |
| **8. Operasyona devir** | Sorun çıktığında kim neye bakacak? | Dashboard, runbook, incident/recovery prosedürü |

Burada **strong scaling**, toplam iş miktarı sabitken daha fazla GPU ile işi ne kadar hızlandırabildiğimizdir. **Weak scaling**, GPU sayısıyla birlikte iş miktarını da artırdığımızda işlem süresini veya verimi ne kadar koruduğumuzdur. İki test de gereklidir; çünkü "GPU'ları iki katına çıkardık" sorusu hangi ölçeklendirme rejiminde olduğumuza göre farklı cevaplar verir.

### 24.7. Mimari kararların arkasındaki temel sorular

Farklı mimari tasarımları karşılaştırırken aşağıdaki sorular, kararların hangi teknik varsayımlara dayandığını görünür kılar:

| Mimari konu | Anlamayı kolaylaştıran soru |
|---|---|
| Model / precision | Hangi kalite hedefi ve numerik doğrulama ile seçildi? |
| GPU / HBM | Weight + KV/activation + çalışma belleği + headroom sığıyor mu? |
| Paralellik | TP/PP/DP/CP/EP neden tam bu topolojiye yerleştirildi? |
| Network | NCCL/RDMA sonucu ile teorik port hızı arasındaki fark ne? |
| Storage | Checkpoint'in bloklayıcı süresi, restore süresi ve bütünlüğü ölçüldü mü? |
| Orkestrasyon | Gang scheduling, quota, preemption ve fragmentation nasıl yönetiliyor? |
| Reliability | Kötü node nasıl tespit, izole ve yeniden doğrulanıyor? |
| Serving | p95/p99 TTFT, TPOT, KV cache, concurrency ve goodput hedefi ne? |
| Veri | Dataset/model lineage ve hassas veri erişimi denetlenebilir mi? |
| Güvenlik | Identity, data policy, audit ve cache isolation yeterli mi? |
| Tesis | Rack güç/soğutma/yedeklilik planı doğrulandı mı? |
| Ekonomi | Birim faydalı çıktı maliyeti ve büyüme varsayımı nedir? |

**Bütün yazının mimari sonucu:** En iyi AI platformu, en yüksek teorik FLOP değeri olan veya sadece en yüksek MFU'ya ulaşan platform değildir. **İstenen kaliteyi, gecikmeyi, güvenliği ve sürekliliği kabul edilebilir toplam maliyetle sağlayan platformdur.** Donanımı, algoritmayı, veriyi ve operasyonu aynı tasarımın parçaları olarak görmeye başladığımızda gerçek ölçeklendirme mühendisliği başlar.

---



---

### Aynı hizmetin iki mimari resmi: "donanım haritası" ve "sorumluluk haritası"

Bir AI platformu yalnızca GPU'ların nasıl kablolandığından ibaret değildir. Donanım haritası bize **verinin nereden nereye taşındığını** gösterir. Sorumluluk haritası ise **hangi kararın nerede verildiğini** anlatır: Kimlik doğrulama nerede yapılıyor? Model seçimi ve routing kimin görevi? Dış veri kaynağına hangi yetkiyle gidiliyor? Modelin bir araç çağrısını kim denetliyor?

<svg viewBox="0 0 720 260" role="img" aria-label="Kurumsal bir AI isteğinin sorumluluk haritası. Kullanıcı veya uygulama yetkili isteği kimlik, kota ve hız sınırını denetleyen AI Gateway&#x27;e gönderir; güvenlik ve DLP katmanı prompt ve verileri filtreler; model servisi GPU&#x27;da LLM veya agent&#x27;ı çalıştırır. Model servisi yalnızca veri gerekirse yetkili ve denetimli RAG ile araç çağrıları yapan veri erişim katmanını çağırır. Çıktı araç çağrısı, sızıntı ve politika açısından çıktı denetiminden geçer ve yanıt kullanıcıya akıtılır. Bunlar ürün sınırları değil, mantıksal sorumluluklardır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="en-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="en-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="en-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Kurumsal bir AI isteğinin sorumluluk haritası</text>
<rect x="16" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Kullanıcı / uygulama</text>
<text x="96.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yetkili istemci</text>
<rect x="192" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="272.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">AI Gateway</text>
<text x="272.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kimlik, kota, hız sınırı</text>
<line x1="176" y1="72" x2="191" y2="72" marker-end="url(#en-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="448.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Güvenlik &amp; DLP</text>
<text x="448.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">prompt / veri filtreleri</text>
<line x1="352" y1="72" x2="367" y2="72" marker-end="url(#en-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="624.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Model servisi</text>
<text x="624.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">GPU&#x27;da LLM / agent</text>
<line x1="528" y1="72" x2="543" y2="72" marker-end="url(#en-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="380" y="150" width="220" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="490.0" y="176.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Veri erişim katmanı</text>
<text x="490.0" y="190.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yetkili RAG + araçlar, denetimli</text>
<path d="M654 100 V178 H602" marker-end="url(#en-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="646" y="143" text-anchor="end" style="fill:var(--c-accent-2);font-size:10.5px">yalnızca veri gerekirse</text>
<rect x="150" y="150" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="245.0" y="176.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Çıktı denetimi</text>
<text x="245.0" y="190.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">araç çağrısı, sızıntı, politika</text>
<path d="M564 100 V124 H245 V148" marker-end="url(#en-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M150 178 H96 V102" marker-end="url(#en-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<text x="100" y="196" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">akıtılan yanıt</text>
<text x="16" y="232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Mantıksal sorumluluklar, ürün sınırları değil: veri katmanı yalnızca gerektiğinde çağrılır;</text>
<text x="16" y="248" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">agent araç çağrıları modelden sonra da denetim isteyebilir.</text>
</svg>

Bu **mantıksal** ayrımdır; zorunlu ürün sınırları veya tek bir request sırası değildir. Örneğin bir agent için araçların güvenlik kontrolü model çıktısından sonra da gerekebilir. Veri erişim katmanı her istekte devreye girmez; yalnızca ihtiyaç olduğunda, kurumsal veri üzerinde yetkili ve denetlenebilir erişim sağlar. Uygulama ve GPU altyapısı aynı veri merkezinde olsa bile bu sorumlulukların karıştırılması tasarımı zorlaştırır.

### Kurumsal güvenlik katmanı GPU performansından neden bağımsız olamaz?

Bir RAG isteğinde model token üretmeden önce retrieval, reranking, erişim kontrolü ve gerekirse hassas veri taraması yapabilir. GPU mükemmel performans verirken kullanıcı yine yavaş yanıt alabilir. Çünkü toplam süre yalnızca GPU süresi değildir:

$$
T_{request}=T_{auth}+T_{retrieval}+T_{queue}+T_{prefill}+T_{decode}+T_{postprocess}.
$$

Bu terimler kimi sistemlerde kısmen örtüşebilir; ifade yalnızca kavramsal ayrıştırmadır. Fakat şu çok nettir: **GPU'yu iki kat hızlandırmak, request süresini mutlaka iki kat azaltmaz.** Güvenlik ve veri erişim adımlarının latency/maliyet etkisini de doğru anlamak gerekir.

### Tek bir 70B modelin iki yüzü: Toplam belleğe değil, sunulan işe bakmak

70B parametreli bir modelin BF16 ağırlıkları (~140 GB) iki 80 GB GPU'ya sığsa da, bu yerleşim yalnızca *statik ağırlık barındırma* kısıtını çözer. Servis mimarisi açısından asıl belirleyici olan, kullanıcı trafiği altındaki dinamik çalışma zarfıdır:

- **Statik sınır:** Ağırlıkların HBM'e yerleşmesi (örneğin INT4 ile ~35 GB veya BF16 ile TP=2'de 70 GB/GPU).
- **Dinamik sınır:** Eşzamanlı isteklerin KV cache hacmi, aktivasyon bellekleri ve CUDA runtime tamponları.

Bir mimar "Model belleğe sığdı mı?" sorusunu geride bırakıp "Hedeflenen concurrency ve bağlam uzunluğunda sistem hangi hızda cevap üretebiliyor?" sorusuna odaklanmalıdır. Modelin nereye sığdığı statik bir donanım sorusudur; hangi hız ve kararlılıkla çalıştığı ise uçtan uca bir sistem mühendisliği problemidir.


### Son sentez: Ölçüm sıralaması bir reçete değil, nedensellik zinciridir

Bir kullanıcı “Yanıtlar yavaş” dediğinde doğru soru her zaman “GPU yetersiz mi?” değildir. Önce *yavaşlığın ne olduğuna* bakmak gerekir: İlk token mı geç geliyor, sonraki token'lar mı seyrek akıyor, yalnızca uzun görsellerde mi bozuluyor, yoksa kimlik/arama katmanında mı bekleniyor?

Bir veri merkezi yöneticisi “GPU'lar boşa çalışıyor” dediğinde de tek bir çözüm yoktur. Örneğin bir kernel memory-bound ise NVLink optimizasyonu hiçbir şeyi değiştirmeyebilir. NCCL bekleyişi ağırsa daha yüksek Tensor Core FLOP değeri yetersiz kalabilir. Depolama kesintisi baskınsa düşük precision kullanmak tek başına yeterli olmaz.

Bu yüzden makalenin boyunca gördüğümüz fiziksel ve mantıksal katmanlar aynı gerçeğin parçalarıdır: **Büyük yapay zekâ sistemleri, hesaplama kapasitesinden çok bağımlılıkların ve veri hareketinin nasıl düzenlendiğiyle anlaşılır.**


## 25. Teknik kapasiteden iş yüküne: Bir AI mimarisi hangi sorularla anlaşılır?

Buraya kadar GPU’nun içindeki hesaplamadan kümeler arası iletişime uzandık. Şimdi aynı teknik bileşenlere farklı açıdan bakalım: Neden iki sistem aynı sayıda GPU kullansa bile birbirinden tamamen farklı davranabilir? Çünkü sunulan hizmet, kullanıcı trafiği, veri erişimi ve gecikme beklentisi mimarinin davranışını değiştirir. Aşağıdaki örnekler bir kurulumun nasıl yapılacağını dikte etmek için değil, **teknik kararların arkasındaki neden-sonuç ilişkilerini** açıklamak için var.

Bir şirket size "300 GPU alalım" dediğinde mimar olarak ilk cevap GPU modeli değildir. Önce istenen **hizmetin kontratını** çıkarırsınız. Bir otel tasarlayan mimar, bina çizmeden önce kaç oda gerektiğini, yoğun sezonda kaç kişinin geleceğini, yangın çıkışlarını ve işletme bütçesini düşünür. AI altyapısında da "kaç GPU?" sorusu, hizmet tanımı olmadan eksik bir sorudur.

### 25.1. Aynı GPU, farklı gereksinimler: Yedi örnek soru

| Soru | Örnek cevap | Tasarıma etkisi |
|---|---|---|
| Ürün ne yapacak? | 70B modelle sohbet, RAG ve belge analizi | Serving, retrieval ve güvenlik katmanları gerekir |
| Trafik şekli nedir? | Ortalama 8, zirvede 25 istek/sn; ani patlamalar var | Kuyruk, autoscaling ve kapasite rezervi gerekir |
| Girdi/çıktı uzunluğu? | p50: 1k/200 token, p95: 8k/1k token | Prefill, decode ve KV belleği farklı etkilenir |
| Başarı ölçüsü? | p95 TTFT ≤ 2 sn, p95 TPOT ≤ 60 ms, başarısız istek ≤ %0,5 | Benchmark kabul kriteri oluşur |
| Veri nerede? | On-prem, belirli sınıflandırılmış veri havuzları | Kimlik, ağ segmentasyonu, veri erişim tasarımı |
| Arıza toleransı? | Tek GPU/node kaybında sınırlı kapasite düşüşü | N+1, çoklu replika, yönlendirme ve kurtarma |
| Bütçe ve zaman? | Belirli kWh, rack, lisans ve hizmet bütçesi | TCO ve mimari alternatif kıyaslaması |

Bu sayıların tamamı **temsili gereksinimdir**. Mimarın başarısı, örnek bir tabloyu ezberlemek değil, gerçek iş yükünden bu tabloyu çıkarmaktır.

İhtiyacı dört iş yüküne ayırmak kararları kolaylaştırır:

- **Batch training:** Saatler/günler süren büyük işler; öncelik time-to-quality, checkpoint, topoloji ve GPU-saat maliyetidir.
- **Interactive inference:** Kullanıcı bekler; öncelik p95/p99 gecikme, goodput, kuyruk ve güvenilirliktir.
- **Offline inference/embedding:** Gece milyonlarca belge işlenebilir; büyük batch ve ucuz kapasite daha anlamlı olabilir.
- **Research/fine-tuning:** Kaynak boyutları çok değişir; kısa bekleme süresi, deney tekrarlanabilirliği ve paylaşımlı havuz önemlidir.

### 25.2. SLI, SLO ve SLA: Üçü neden farklı?

**SLI (Service Level Indicator)** ölçtüğümüz şeydir: p95 TTFT, hata oranı, zamanında tamamlanan iş oranı gibi. **SLO (Service Level Objective)** bu ölçü için iç hedefimizdir: "son 30 günde isteklerin %99'u beş saniye altında tamamlanmalı" gibi. **SLA (Service Level Agreement)** ise müşteriyle yapılmış sözleşmesel taahhüt ve olası yaptırımları içerebilir. Bir SLO'nun ihlali her zaman sözleşmesel SLA ihlali anlamına gelmez.

İyi SLO yalnızca ortalama gecikmeye bakmaz. Ortalama 1 saniye olsa bile yüz kullanıcıdan beşi 15 saniye bekliyorsa kurumsal bir uygulama başarısız sayılabilir. Bu nedenle **p95** (isteklerin yaklaşık %95'inin altında kaldığı değer) ve **p99** (en yavaş %1'e yaklaşan kuyruk) önemlidir.

### 25.3. Architecture Decision Record (ADR) neden kullanılır?

Mimarinin dokümantasyonu yalnızca kutular ve oklar değildir. Altı ay sonra "neden bu sistemi seçtik?" sorusuna cevap verebilmelidir.

```text
ADR-007 — Production inference için aggregated vs disaggregated serving
Durum: Önerildi / Deneyle doğrulanacak
Bağlam: Uzun prompt p95 12k token; eşzamanlılık artıyor.
Alternatif A: Tek worker içinde prefill + decode
Alternatif B: Prefill ve decode için ayrı worker havuzları
Karar ölçütü: Aynı kalite ve trafik altında p95 TTFT/TPOT,
               başarılı istek/sn/GPU, KV transfer maliyeti, işletme riski
Deney: A/B yük testi; soğuk/sıcak prefix; 4 trafik profili
Geri dönüş: Kazanç kanıtlanmazsa A'ya dön
Sahip: AI Platform ekibi
```

**Kavramsal ayrım:** Ölçülmemiş avantaj bir **hipotez** olarak kalır. Örneğin “RDMA kesin hızlandırır” demek yerine “KV transferi gecikmenin ölçülmüş %18’ini oluşturuyorsa RDMA alternatifini incelemek anlamlı olabilir” demek, kanıta dayalı düşünmeyi daha iyi yansıtır.

### 25.4. Referans mimarinin sınırları

Bir çözümün üç ayrı ölçeği vardır: **mantıksal mimari** (hangi yetenekler var?), **fiziksel mimari** (hangi sunucuda, NIC'te, rafta?) ve **operasyonel mimari** (kim izler, kim müdahale eder, kim geri döndürür?). Çoğu AI şeması ilkini anlatır; üretimde sorun genellikle diğer ikisinin arasında çıkar.

---


## 26. RAG, agent ve kurumsal veri mimarisi: GPU'nun dışındaki darboğaz

Bir şirketin çalışanı "Geçen çeyrek ürünümüz neden daha az satıldı?" diye sorduğunda modelin parametreleri sorunun cevabını içermeyebilir. Kurumun veri kaynaklarına erişmek gerekebilir. İşte bu noktada **Retrieval-Augmented Generation (RAG)** devreye girer.

Sezgisel örnek: Bir doktorun her şeyi ezberlemek yerine gerektiğinde onaylı kılavuza bakması gibi, bir RAG sistemi de önce ilgili kanıtları toplar; ardından modelden kanıta dayalı yanıt ister. Ama "ilgili doküman bulundu" demek otomatik olarak "doğru doküman bulundu" anlamına gelmez.

### 26.1. RAG'de gerçekte kaç katman var?

<svg viewBox="0 0 720 260" role="img" aria-label="RAG&#x27;de iki hat vardır. Çevrimdışı ingestion, ACL etiketli izinli kaynakları alır, anlamsal parçalara böler, vektörlere gömer ve vektörleri ACL metadata&#x27;sıyla vektör dizinde saklar. Çevrimiçi tarafta her soru kullanıcının kimliği, tenant&#x27;ı ve rolüyle gelir; filtreli retrieval dizini kullanıcının yetkisiyle, ACL filtresi ve hibrit aramayla sorgular; reranking kanıtı seçer; LLM bu kanıta dayalı, atıflı yanıt verir. Yalnızca modelin cevabını filtrelemek geç kalır, çünkü yetkisiz metin prompt&#x27;a çoktan girmiş olur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rg2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rg2-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="rg2-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">RAG&#x27;de iki hat var; yetkilendirme retrieval&#x27;dan önce gelir</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:700">çevrimdışı: ingestion</text>
<rect x="16" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">İzinli kaynaklar</text>
<text x="96.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">belge + ACL etiketi</text>
<rect x="192" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="272.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Chunking</text>
<text x="272.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">anlamsal parçalar</text>
<line x1="176" y1="85" x2="191" y2="85" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="448.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Embedding</text>
<text x="448.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">vektörler</text>
<line x1="352" y1="85" x2="367" y2="85" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="624.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Vektör dizin</text>
<text x="624.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">vektör + ACL metadata</text>
<line x1="528" y1="85" x2="543" y2="85" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:700">çevrimiçi: her soru</text>
<rect x="16" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Soru + kimlik</text>
<text x="96.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tenant, rol</text>
<rect x="192" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="272.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Filtreli retrieval</text>
<text x="272.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">ACL filtresi + hibrit arama</text>
<line x1="176" y1="175" x2="191" y2="175" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="448.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Reranking</text>
<text x="448.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kanıtı seç</text>
<line x1="352" y1="175" x2="367" y2="175" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="624.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LLM + atıf</text>
<text x="624.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kanıta dayalı yanıt</text>
<line x1="528" y1="175" x2="543" y2="175" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M624 110 V124 H272 V148" marker-end="url(#rg2-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<text x="448" y="139" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">dizin kullanıcının yetkisiyle sorgulanır</text>
<rect x="16" y="214" width="688" height="34" rx="6" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="28" y="236" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">Yalnızca modelin cevabını filtrelemek geç kalır: yetkisiz metin prompt&#x27;a çoktan girmiştir.</text>
</svg>

**Embedding**, bir metin parçasını benzerliği hesaplanabilen sayısal vektöre dönüştürür. **Vector index**, bu vektörler arasında yakın komşu aramasını hızlandırır. **Hybrid retrieval**, yalnızca vektör yakınlığı yerine örneğin BM25 gibi sözcüksel aramayla birlikte çalışabilir. **Reranker**, ilk getirilen adayları daha pahalı ama daha hassas bir modelle yeniden sıralar.

### 26.2. En tehlikeli RAG hatası: Retrieval sonrası yetkilendirme

Bir kullanıcı bir belgeye yetkili değilse o belgenin retrieval sonucuna hiç girmemesi gerekir. Sadece modelin cevabını filtrelemek, yetkisiz içeriğin prompt'a sokulması riskini ortadan kaldırmaz. Tenant, departman, doküman etiketi, satır/sütun bazlı veri erişimi gibi koşullar **retrieval veya data-service katmanında** uygulanmalıdır.

### 26.3. RAG kalite ölçümü

Bir RAG sistemini yalnızca modelin verdiği cevapla değerlendirmek eksiktir. Önce **retrieval recall@k** ile gerekli kanıtın ilk $k$ sonuç içinde gelip gelmediğine bakın. Sonra **context precision**, kaynak atfı doğruluğu, faithfulness, cevap kalitesi, latans ve token maliyetini ölçün. Bu metriklerin bazıları değerlendirme yöntemine göre öznel veya model-judge temelli olabilir; kritik senaryolarda insan doğrulaması gerekir.

### 26.4. Agent mimarisi: Model yalnızca düşünmez, eylem de tetikleyebilir

Agent; arama, kod çalıştırma, veritabanı sorgulama, bilet açma gibi araç çağrılarını planlayabilir. **Tool authorization**, **approval gates**, **idempotency**, **audit trail** ve **minimum privilege** bu noktada performans optimizasyonu kadar önemlidir. Model bir API çağrısı önerdi diye o çağrının yetkili hale gelmesi mümkün değildir.

### 26.5. AI Gateway ile Data Product Gateway'in doğru sınırı

- **AI Gateway:** Kim, hangi AI hizmetini çağırabilir? Kota, routing, authentication ve request politikaları.
- **AI Safety:** Girdi/çıktı risk kontrolleri, hassas veri politikası, belirli içerik testleri ve gerektiğinde human review.
- **Data Product Gateway:** Kurumsal verilere ihtiyaç oluştuğunda, kullanıcı/servis yetkisi ve veri ürünü sözleşmesine göre kontrollü erişim.
- **Serving engine:** Model hesaplama, KV cache, batching ve inference paralelleştirme.

Bunlar mantıksal sorumluluklardır; tek ürün veya dört ayrı ürün olmak zorunda değillerdir. Ancak politika sınırları belirsiz kalırsa "model çalışıyor ama şirket verisini kim verdi?" sorusuna yanıt verilemez.

---


## 27. Güvenlik ve izolasyon: Paylaşılan kaynaklar hangi yeni riskleri doğurur?

İki farklı ekibin aynı GPU'yu paylaşması teknik olarak mümkün olabilir, ama **kaynak paylaşımı** ile **güvenlik izolasyonu** aynı şey değildir. Özellikle kurumsal LLM/VLM sistemlerinde veri modelin eğitim örneklerinden, prompt'larından, araç çağrılarından, cache'inden, log'larından ve üretilen cevaplarından geçebilir.

### 27.1. Threat model: Neyi, kimden koruyoruz?

| Varlık | Olası risk | Mimari kontrol |
|---|---|---|
| GPU hesaplama kapasitesi | Bir tenant'ın diğerlerini yavaşlatması | Kota, scheduler, ayrı havuz veya donanım partition |
| KV/prefix cache | Tenant içerik karışması | Tenant/salt sınırı, cache policy, test |
| Model weight ve adapter | Yetkisiz erişim/tampering | İmzalı artefact, RBAC, sürüm ve hash |
| Dataset / RAG index | Yetkisiz retrieval | Data-level authorization, encryption, audit |
| Agent tools | Yetki dışı eylem | Tool allowlist, scoped credentials, approval |
| Container ve sürücüler | Supply-chain veya ayrıcalık hatası | Image scanning, pinning, least privilege |
| Telemetry ve debug | Gizli prompt'ların loglara sızması | Redaction, retention, role-based access |

### 27.2. MIG, MPS ve time slicing güvenlik eşanlamlısı değildir

- **Time slicing:** İşler GPU zamanını paylaşır; gecikme ve hafıza bant genişliği için katı izolasyon varsayılmamalıdır.
- **MPS:** Çok süreçli CUDA işlerini birlikte planlayabilir; bazı kaynak sınırlamaları destekler ama fiziksel partition izolasyonuyla karıştırılmamalıdır.
- **MIG:** Desteklenen GPU'larda işlem, bellek ve cache kaynaklarının ayrı GPU instance'larına fiziksel bölünmesine olanak verir. Bununla birlikte host, sürücü, kontrol düzlemi ve ağ politikası güvenliği ayrıca tasarlanmalıdır. [K45]

Bu yüzden regülasyonlu veya hassas tenant senaryosunda "MIG açtık, bütün riskler bitti" demeyiz. Her ortamın izolasyon garantisini üreticinin destek koşulları ve tehdit modeliyle birlikte doğrularız.

### 27.3. Prompt injection ile data access control neden birbirinin alternatifi değildir?

Prompt injection, bir metnin modele yetkisiz davranış yaptırmaya çalışmasıdır. Modelin erişebildiği araç ve veriler zaten **least privilege** ile sınırlandıysa, manipüle edici içeriklerin yaratabileceği hasar da azalır. Sadece prompt'a "talimatları görmezden gelme" yazmak teknik güvenlik kontrolü sayılmaz.

Örnek: Bir RAG kaynağındaki belge, sistem prompt'unu taklit eden metin taşıyabilir. Bu metin kullanıcı yetkisini genişletmemelidir. Belge içeriği veri olarak işlenmeli; modelin aracı kullanma yetkisi harici politika motoru ve tool gateway üzerinde doğrulanmalıdır.

### 27.4. Veri yaşam döngüsü boyunca güvenlik

Veri sisteme girdiğinde ingestion tarafında izin ve sınıflandırma, index'te tenant ayrımı, prompt assembly'de minimizasyon, API çağrısında kimlik, model çıktı katmanında güvenlik testi, loglarda veri saklama ve silme politikası uygulanmalıdır. **Silme isteği**, yalnızca orijinal dosyayı silmek değildir: türetilmiş embedding, cache, snapshot, analitik kopya ve yedeklerin politikaya uygun yaşam döngüsü ayrıca tanımlanmalıdır.

### 27.5. Güvenlik açısından örnek inceleme başlıkları

```text
[ ] Model / tokenizer / adapter hash ve kaynağı onaylı
[ ] Training ve serving container image imzalı veya doğrulanabilir
[ ] Secrets merkezi servisten kısa ömürlü credential ile alınıyor
[ ] Runtime host ayrıcalıkları ve GPU device görünürlüğü kısıtlı
[ ] Tenant sınırları KV / prefix cache / RAG index üzerinde test edildi
[ ] Tool invocation dış yetkilendirme katmanından geçiyor
[ ] Prompt injection / data exfiltration risk testleri yapıldı
[ ] Hassas ham içerik logging ve tracing politikasına uygun
[ ] Incident response, erişim iptali ve anahtar döndürme denendi
```

---


## 28. 70B model üzerinden düşünelim: Kurumsal bir AI sisteminin bileşenleri

Şimdi tüm parçaları örnek bir problem üzerinde birleştirelim. Bir kurum, 70B dense dil modelinden hem sohbet hem RAG hizmeti almak istiyor. Bazı kullanıcılar uzun belgeler gönderiyor, bazıları kısa sorular soruyor. Hassas veri on-prem kalmak zorunda. Bu bölümdeki bütün yük rakamları **öğretici varsayımdır**; gerçek ürün için benchmark gerekir.

### 28.1. Gereksinimler

- Model: 70B yoğun Transformer; ilk deneme BF16 ve alternatif INT8/FP8 ağırlık sunumları.
- Trafik: Normal 5 istek/sn, yoğun saatte 15 istek/sn; girdi p50 1k, p95 8k token; çıktı p50 250, p95 1k token.
- Kullanıcı deneyimi: p95 TTFT ≤ 2 sn, p95 TPOT ≤ 70ms; RAG kullananlar için ayrı bütçe.
- Güvenlik: Model hizmeti yalnızca yetkili erişim; kurumsal veri erişimi ihtiyaç halinde ayrı data service üzerinden.
- Süreklilik: Tek serving node kaybında kısmi kapasite düşüşü kabul; tüm hizmetin kesilmesi kabul değil.

### 28.2. İlk bellek kontrolü

BF16 ağırlıklar yaklaşık 140 GB. İki 80 GB GPU toplam 160 GB ham HBM sunsa da runtime overhead ve KV için alan az kalabilir. Bu yüzden **ilk ölçülebilir alternatif** olarak dört 80 GB GPU üzerinde model-paralel inference yerleşimini düşünelim. Bu dört GPU'nun birbirine nasıl bağlı olduğu (NVLink/PCIe) doğrulanmalıdır; TP=4 seçilmesi GPU sayısı tek başına optimum demek değildir.

8.192 token başına 2,5 GiB mantıksal KV örneğini hatırlayalım. Eşzamanlı 32 böyle istek, tek model replika genelinde yaklaşık 80 GiB mantıksal KV oluşturabilir. Bu KV, uygun runtime ve model yerleşimiyle GPU'lara shard edilebilir; block allocator ve prefix paylaşımı fiili tahsisi değiştirir. Asıl soru "80 GiB toplam var mı?" değil, **cache hangi GPU'ya ne kadar dağılıyor ve p95 istek uzunluğunda güvenle kaç sequence kabul edebiliriz?** sorusudur.

### 28.3. Başlangıç referans mimarisi

<svg viewBox="0 0 720 344" role="img" aria-label="Tek bir 70B servisi için başlangıç referans mimarisi. İstemci kimlik ve hız sınırı için AI Gateway&#x27;e, ardından girdi ve çıktı denetimi için güvenlik katmanına, sonra kuyruk ve kabulü yöneten model router&#x27;a gider. Router, farklı sunuculardaki 70B serving replikalarını besler; böylece bir düğüm kaybı kesinti değil kısmi kapasite kaybıdır; her replika NVLink üzerinden TP = 4 ile dört H100 80 GB GPU&#x27;da çalışır. Yaklaşık 140 GB&#x27;lık BF16 ağırlıklar GPU başına ~35 GB tutar; 320 GB&#x27;ın kalanı KV cache ve runtime tamponlarına kalır. Data gateway yalnızca gerektiğinde yetkili RAG erişimi sağlar; control plane ve FinOps telemetri, rollout ve token başına maliyeti yönetir. Test edilecek varyantlar: BF16 birleşik TP 4, düşük hassasiyetli ağırlık ve yalnızca hızlı KV ağı varsa ayrıştırılmış serving." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rf7-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rf7-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Tek bir 70B servisi için başlangıç referans mimarisi</text>
<rect x="16" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">İstemci</text>
<text x="96.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">uygulama / kullanıcı</text>
<rect x="192" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="272.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">AI Gateway</text>
<text x="272.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kimlik, hız sınırı</text>
<line x1="176" y1="70" x2="191" y2="70" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="448.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Güvenlik katmanı</text>
<text x="448.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">girdi / çıktı denetimi</text>
<line x1="352" y1="70" x2="367" y2="70" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="624.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Model router</text>
<text x="624.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kuyruk, kabul</text>
<line x1="528" y1="70" x2="543" y2="70" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="624.0" y1="96" x2="624.0" y2="122" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="124" width="688" height="104" rx="10" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="28" y="144" text-anchor="start" style="fill:var(--c-accent-2);font-size:12.5px;font-weight:700">Farklı sunuculardaki 70B serving replikaları: bir düğüm kaybı = kısmi kapasite, kesinti değil</text>
<rect x="28" y="154" width="324" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">replika 1: 4 × H100 80 GB, TP = 4</text>
<rect x="40" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="65.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 0</text>
<rect x="96" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="121.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 1</text>
<rect x="152" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="177.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 2</text>
<rect x="208" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="233.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 3</text>
<text x="268" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">NVLink</text>
<rect x="368" y="154" width="324" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="380" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">replika 2: 4 × H100 80 GB, TP = 4</text>
<rect x="380" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="405.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 0</text>
<rect x="436" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="461.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 1</text>
<rect x="492" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="517.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 2</text>
<rect x="548" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="573.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 3</text>
<text x="608" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">NVLink</text>
<text x="28" y="248" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">BF16 ağırlık ≈ 140 GB → GPU başına ~35 GB; 320 GB&#x27;ın kalanı KV cache + runtime tamponları içindir.</text>
<rect x="16" y="264" width="336" height="46" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="184.0" y="284.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Data gateway</text>
<text x="184.0" y="299.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yetkili RAG erişimi, yalnızca gerektiğinde</text>
<rect x="368" y="264" width="336" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="536.0" y="284.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Control plane &amp; FinOps</text>
<text x="536.0" y="299.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">telemetri, rollout, token başına maliyet</text>
<text x="16" y="332" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Aynı trafikte varyantları test edin: A = BF16 birleşik TP 4 · B = düşük hassasiyetli ağırlık · C = ayrıştırılmış, yalnızca hızlı KV ağı varsa.</text>
</svg>

Bu şema mantıksaldır. Data Product Gateway, AI Gateway'in zorunlu alt modülü değildir. Retrieval çıktısının istek akışına nasıl eklendiği, güvenlik sınırı ve veri erişim sözleşmesi tasarımda ayrıca tanımlanır.

### 28.4. Kapasite deneyi

Aynı trafikte üç varyant test edilsin:

- **A:** BF16, aggregated prefill/decode, TP=4.
- **B:** Daha düşük hassasiyetli weights, aggregated; kalite regresyon testiyle.
- **C:** A'nın disaggregated prefill/decode mimarisi, yalnızca gerekli yüksek hızlı KV fabric sağlanıyorsa.

Her varyant için sıcak/soğuk cache, kısa/uzun prompt ve artan concurrency koşullarında **TTFT, TPOT, p99, request/s, toplam output token/s, istek başına peak KV, güç ve başarısızlık oranı** ölçülür. C mimarisi daha pahalı ve daha karmaşıksa ancak SLO içinde anlamlı kazanım sağladığında seçilmelidir.

### 28.5. Arıza testi ve kurtarma

Önce tek worker durdurulduğunda yük dengeleyicinin sağlıksız replika'ya yeni istek göndermediğini doğrulayın. Ardından eski worker'ın streaming işlemlerinin nasıl sonuçlandığını ve hata mesajının kullanıcıya nasıl yansıtıldığını test edin. Son olarak model rollout esnasında yeterli sıcak kapasite kalıp kalmadığını ölçün.

### 28.6. Bu örnek için kararların nasıl belgelendirilebileceği

```text
Ürün: Kurumsal LLM + RAG
İş yükü: p50/p95 prompt-output dağılımı ve peak QPS ektedir
Seçilen model/runtime: Sürüm hash'leriyle kaydedildi
GPU ve topology: Benchmark sonucu ve kapasite rezervi ektedir
Quality gate: Türkçe, domain, uzun context, hallucination, güvenlik testleri
Performance gate: p95 TTFT, p95 TPOT, goodput, cost/token
Reliability: Failover testi, restore ve rollback kanıtı
Security: IAM, tenant isolation, data-product policy, logging retention
Operasyon: Dashboard, alarm, on-call, runbook, bakım takvimi
Karar tarihi / sahip / tekrar değerlendirme tetikleyicisi
```

Gerçek bir tasarım incelemesinin çıktısı bir ürün logosu değil, bu kanıt setidir.

---


## 29. AI platformlarının zaman içindeki gelişimi: Organizasyon ve operasyon

AI altyapısının fiziksel bileşenlerinin yanında bir de organizasyonel boyutu vardır. Sistem büyüdükçe GPU, ağ, veri, model, güvenlik ve ürün ekipleri arasındaki ilişki önem kazanır. Aşağıdaki roller ve zaman aralıkları, gerçek bir kurumun planı değil; organizasyonel karmaşıklığı anlaşılır kılmak için kurulmuş **temsili bir örnek**tir.

### 29.1. Farklı ekipler hangi sorumluluklarla karşılaşır?

| Sorumluluk | Öncelikli sahip | Birlikte çalıştığı ekip |
|---|---|---|
| GPU/driver/fabric sağlığı | Platform/SRE | DC/network |
| Dataset ve ingestion kalitesi | Data Engineering | ML Research |
| Model training ve kalite | ML Engineering/Research | GPU Platform |
| Serving p95/p99 SLO | Inference Platform | Ürün/SRE |
| Kimlik, veri erişimi, denetim | Security/Data Governance | Platform/Ürün |
| TCO, kapasite forecast | Platform/FinOps | Ürün/Finans |

Bu tablo **örnek RACI başlangıcıdır**, her kuruma uyan tek organizasyon modeli değildir. Esas olan her olay, kontrol ve karar için birincil sorumlunun yazılı olmasıdır.

> **Zaman çizelgesi notu:** Aşağıda 30/60/90 gün biçiminde kalan dönem adları, sıralı bir kurulum takvimi önermez; kavramları aşamalı anlatmak için kullanılan varsayımsal bir zaman eksenidir.

### 29.2. İlk dönem örneği: Sistemin mevcut durumunu anlamak

Bir AI platformunu anlamanın ilk boyutu, bileşenlerin ve ölçümlerin haritalanmasıdır: GPU/NUMA/NIC topolojisi, depolama veri yolu, yazılım sürümleri, model kalitesi ve p95 gecikme gibi değerler bu resmin parçalarıdır. Buradaki “ilk dönem”, gerçek bir 30 günlük teslimat sözü değil; mevcut durumun neden ölçülmeden yorumlanamayacağını anlatan örnektir.

### 29.3. Sonraki dönem örneği: Kararlılık ve otomasyon

Sistem olgunlaştıkça golden image, scheduling politikası, model registry, CI/CD, gözlemlenebilirlik, olay kayıtları ve checkpoint geri dönüş mekanizmaları önem kazanır. Bunlar birbirinden bağımsız ürün isimleri değil; sürüm tutarlılığının, tekrarlanabilirliğin ve hata toleransının farklı yüzleridir. Örneğin bir failover testi, donanım yedeğinin bulunması ile hizmetin gerçekten sürdürülebilmesi arasındaki farkı görünür kılabilir.

### 29.4. Olgunlaşma dönemi örneği: Performans ve maliyet dengesi

Daha sonraki aşamada performans optimizasyonlarının birbirleriyle ilişkisi öne çıkar. Örneğin %40 MFU'yu %45'e yükseltmek, her zaman p99 TTFT'yi azaltmaktan daha değerli değildir. Kernel fusion, quantization, caching ve ayrık prefill/decode gibi tekniklerin etkisi, trafik profili, doğruluk ve maliyetle birlikte düşünüldüğünde anlam kazanır.

### 29.5. Operasyonel gözlemlerin farklı zaman ölçekleri

- Haftalık: SLO ve arıza özeti; kapasite değişimi; açık kritik riskler.
- Aylık: Cost/goodput, GPU fault pattern, model sürüm dağılımı, teknik borç.
- Üç aylık: Network/storage darboğazı trendi, donanım yaşam döngüsü, felaket kurtarma ve güvenlik tatbikatı.

Bu zaman aralıkları bir uygulama takvimi veya sektör standardı değildir. Yalnızca günlük, haftalık ve uzun dönem ölçümlerin farklı bilgiler taşıdığını gösterir. Ayrıca her ölçümün hedefe dönüştürülmesi, metrik odaklı istenmeyen davranışlara yol açabilir.

---


## 30. Sayıları anlamlandırmak: Küçük Python deneyleri ve varsayımlar

Aşağıdaki örnek, dış bağımlılığı olmayan Python 3 koduyla **öğretici alt sınırlar** üretir. Çalışan bir GPU kümesini sorgulamaz, gerçek runtime benchmark'ının yerini tutmaz. Ama tasarım görüşmesinde söylenen sayıların boyut olarak mantıklı olup olmadığını hemen test etmeye yardımcı olur.

```python
from math import sqrt


def weights_gb(params_billion: float, bits: float) -> float:
    """Onluk GB; quantization metadata / runtime overhead dahil değil."""
    return params_billion * 1e9 * bits / 8 / 1e9


def kv_gib(layers: int, kv_heads: int, head_dim: int,
           tokens: int, bytes_per_element: int = 2) -> float:
    """Toplam mantıksal KV; TP/PP shard etmeden önce."""
    n_bytes = 2 * layers * kv_heads * head_dim * tokens * bytes_per_element
    return n_bytes / (1024 ** 3)


def checkpoint_interval_min(snapshot_tb: float,
                            write_gbs: float, mtbf_hours: float) -> float:
    """Young yaklaşık optimum; bloklayıcı checkpoint varsayımı."""
    checkpoint_s = snapshot_tb * 1000 / write_gbs
    mtbf_s = mtbf_hours * 3600
    return sqrt(2 * checkpoint_s * mtbf_s) / 60


print("70B BF16 weights (GB):", weights_gb(70, 16))
print("70B-like KV @ 8,192 tokens (GiB):",
      round(kv_gib(80, 8, 128, 8192), 2))
print("Checkpoint interval (min):",
      round(checkpoint_interval_min(7.5, 250, 3), 2))


def tp_max(ffn_dim: int, peak_flops: float = 990e12,
           link_bytes_s: float = 450e9) -> float:
    """T_hesap = T_iletişim olduğu TP derecesi (Megatron MLP, birinci mertebe)."""
    return ffn_dim * link_bytes_s / peak_flops


def dp_min_tokens_per_gpu(peak_flops: float = 990e12,
                          link_bytes_s: float = 400e9,
                          experts: int = 1, active: int = 1) -> float:
    """Gradyan senkronunu örten GPU başına token; MoE için x E/k."""
    return experts / active * peak_flops / link_bytes_s


print("TP limit, NVLink 450 GB/s:", round(tp_max(28672), 2))
print("TP limit, NIC 50 GB/s:", round(tp_max(28672, link_bytes_s=50e9), 2))
print("DP tokens/GPU across nodes:", round(dp_min_tokens_per_gpu()))
print("MoE E=128, k=4 tokens/GPU:",
      round(dp_min_tokens_per_gpu(experts=128, active=4)))
print("Decode ceiling, 8xH100 TP=8 (tokens/s):", round(8 * 3350 / 140))
```

Beklenen çıktı:

```text
70B BF16 weights (GB): 140.0
70B-like KV @ 8,192 tokens (GiB): 2.5
Checkpoint interval (min): 13.42
TP limit, NVLink 450 GB/s: 13.03
TP limit, NIC 50 GB/s: 1.45
DP tokens/GPU across nodes: 2475
MoE E=128, k=4 tokens/GPU: 79200
Decode ceiling, 8xH100 TP=8 (tokens/s): 191
```

Buradaki checkpoint sonucu, $7{,}5\,\mathrm{TB}/250\,\mathrm{GB/s}=30\,\mathrm{s}$ ve $M=3\,\mathrm{h}$ varsayımlarıyla bulunur. Sayılar tutarlı görünüyorsa bir sonraki aşama **gerçek workload profiler ve benchmark'ına geçmektir**.

### 30.1. Sekiz örnek belirti ve arkasındaki olası mekanizmalar

| Belirti | İlişkili ölçüm veya değişken | Yanlış olabilecek çıkarım |
|---|---|---|
| Training MFU aniden düştü | Konfigürasyon, profiler timeline, straggler | Rastgele TP artırmak |
| NCCL hang | RAS, rank state, GPU/NIC/topoloji | Tüm timeout'ları büyütmek |
| TTFT p99 yükseldi | Kuyruk, uzun prompt, cold start | Sadece GPU eklemek |
| TPOT kötü | Decode HBM, batch ve cache | Sadece prefill optimize etmek |
| Checkpoint yavaş | Gerçek write BW, shard ve metadata | GDS'nin her şeyi çözeceğini varsaymak |
| HBM yetmiyor | Weights, KV, activations ayrımı | Sebebi bilmeden quantization yapmak |
| MoE dengesiz | Expert load ve AllToAll dağılımı | Dense benchmark'ıyla kıyaslamak |
| Model çıktı kalitesi düştü | Model/precision/data değişikliği | MFU yüksek diye yayına devam etmek |

**Örneklerin ortak noktası:** Tek bir belirti tek bir kök nedene işaret etmez. Bu tablo, farklı donanım ve yazılım katmanlarının aynı kullanıcı deneyimini nasıl etkileyebildiğini göstermeyi amaçlar.

---




---

**KISIM VIII — Öğrendiklerimizi bir araya getirmek**

*Farklı başlıklardaki ilkeleri birkaç düşünce deneyi ve sözlük yardımıyla pekiştireceğiz.*


## 31. Öğrendiklerimizi sınayalım: Altı kısa düşünce deneyi

Şimdi bir mühendislik refleksi geliştirelim: Bir sayıyı gördüğümüzde hemen onu hangi varsayımla hesapladığımızı soralım. Aşağıdaki soruların amacı sınav yapmak değil; okuduklarımızı birkaç kısa hesap üzerinden somutlaştırmak.

### Soru 1 — 70B modeli BF16 olarak belleğe koymak

70 milyar parametre, her biri 2 byte ise yalnızca ağırlık belleği kaç GB eder?

**Çözüm:**

$70\times10^9\times2=140\times10^9$ byte = **140 GB**. Burada KV cache, aktivasyon ve çalışma tamponları yok. Sekiz adet 80 GB H100 toplamda teorik 640 GB HBM sunsa da fiziksel dağıtım yöntemi ve parçalı yerleşim önemlidir.



### Soru 2 — Roofline eşiği

990 TFLOP/s, 3,35 TB/s özellikleriyle H100 BF16 arithmetic intensity eşiği nedir?

**Çözüm:**

$990/3.35\approx\mathbf{295.5}$ FLOP/byte. Bu değer BF16 yoğun Tensor Core tepe hesabı ve HBM3 yolu için yaklaşık eşiktir.



### Soru 3 — 16 GPU'yu nasıl bölersin?

Modeli dört pipeline stage'e bölen ve her stage'de iki GPU'luk tensor parallel grup kullanan bir kurulumda geriye kaç data parallel kopyası kalır? CP=1 olsun.

**Çözüm:**

$DP=16/(TP\times PP)=16/(2\times4)=\mathbf{2}$. Her veri paralel kopyası 8 GPU kullanır.



### Soru 4 — Ring AllReduce hacmi

8 GPU ve GPU başına 8 GB veri için ideal ring AllReduce gönderim hacmi kaç GB/GPU?

**Çözüm:**

$2\times(8-1)/8\times8=\mathbf{14}$ GB/GPU. Bu, payload modeli; mesaj gecikmesi ve protokol overhead'i dahil değildir.



### Soru 5 — Depolamayı hızlandırmak neden arıza kaybını azaltır?

Küme MTBF = 3 saat, bloklayıcı checkpoint süresi 30 saniye. Yaklaşık optimum checkpoint aralığı nedir?

**Çözüm:**

$\sqrt{2\times30\times10800}=804.98$ saniye ≈ **13,4 dakika**. Bu formül ilk mertebe Young–Daly yaklaşımıdır.



### Soru 6 — En güçlü GPU'yu almak yeter mi?

Bir ekibin GPU'su iki kat fazla tepe FLOP/s sunuyor, ama bellek bant genişliği değişmiyor. Decode iş yükü bellek sınırlıysa neden iki kat hızlanmayabilir?

**Çözüm:**

Çünkü çarpım $\alpha W_{\mathrm{mem}}$ değişmiyorsa memory roof aynı kalır. Ek hesaplama birimleri veri beklemeye devam eder. İyileştirme için veri yeniden kullanımı, batching, compression veya daha yüksek efektif bellek bant genişliği gerekebilir.



---


## Sonuç: GPU’dan süper bilgisayara uzanan ortak ilkeler

GPU'nun en küçük yürütme grubu olan warp'tan başladık; binlerce GPU'luk kümelere, checkpoint tasarımına ve MFU'ya kadar geldik. Üstelik Lambda ve MegaScale çalışmalarında gördüğümüz üzere aynı sistemin yazılımı ve zamanlaması, tek başına GPU neslinden beklediğimizden çok daha büyük fark yaratabiliyor. Aslında yol boyunca tekrar tekrar aynı iki fikirle karşılaştık: **veriyi yakın tutmak** ve **kaçınılmaz beklemeyi mümkünse başka işle doldurmak**.

Biraz düşünün: SM içindeki warp scheduler veri bekleyen warp'ın yerine başka bir warp çalıştırabiliyor. GPU içindeki tiling, HBM'ye gereksiz dönüşleri azaltıyor. DGX içinde NVLink, katmanlar arası yoğun haberleşmeyi yakın tutuyor. Çok büyük kümede PP ve iletişim overlap teknikleri benzer şekilde boşta geçen zamanı azaltmaya çalışıyor. GDS ve asenkron checkpoint ise aynı fikri depolama katmanına taşıyor.

'En yavaş halka' benzetmesi faydalı ama tek başına eksiktir. Çünkü bazı aşamalar birbirinin üzerine bindirilebilir: ağ transferi sırasında hesaplama devam edebilir, checkpoint arka planda yazılabilir, bir warp veri beklerken başka warp çalışabilir. İyi AI altyapısı mühendisliği tam da bu **örtüşme ve veri yerelliği** fırsatlarını tasarlama sanatıdır.

Bir GPU'nun içindeki warp scheduler neyi çözüyorsa, çok daha büyük ölçekte pipeline scheduling benzer bir sorunu çözmeye çalışır: **bekleme süresini başka faydalı işle doldurmak**. Shared memory tiling ile HBM'e daha az gitmek neyse, NVLink alanında TP yerleşimi de aynı prensibin dağıtık sistem karşılığıdır: **veriyi mümkün olduğunca yakın tutmak**.

Bu yüzden doğru soru 'kaç GPU alalım?' değildir. Doğru sorular şunlardır: *Hangi iş yükü? Hangi precision? Hangi matris şekli? Hangi batch/concurrency? Hangi iletişim topolojisi? Hangi checkpoint ve recovery hedefi? Ve bunları hangi metriklerle doğrulayacağız?*

Bir **AI Architect** için yolculuğun son durağında bir soru daha var: *Sistem en hızlı anında değil, yoğun ve sorunlu gününde hedef kullanıcı deneyimini koruyor mu?* Çünkü iyi bir platform, yalnızca rekor kıran benchmark değil; takip edilebilir veri kaynağı, güvenli servis yolu, sağlıklı scheduler, gerçek arıza kurtarma planı, sürdürülebilir güç bütçesi ve ölçülebilir birim maliyet demektir.

**Kapanış:** 1.024 GPU’luk bir kümenin performansını anlamak için yalnızca peak TFLOP/s değerine bakmak yetmiyor. Hangi verinin hangi bellek düzeyinden geçtiğini, GPU’ların ne zaman haberleştiğini, eğitim ve çıkarımın hangi kaynakları tükettiğini birlikte düşünmek gerekiyor. Bu yazının paylaşmak istediği esas fikir de bu: Yapay zekâ altyapısı, birbirine bağlı pek çok fiziksel ve yazılımsal mekanizmanın toplamı.

---



---

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

İlk sayfadaki 1.024 H100 ve %38 MFU örneğine geri dönelim. Bu sayı bize artık “GPU'nun geri kalanı kapalı” demiyor. Bir soru işareti gösteriyor ve bu işareti çözmek için üç ayrı ölçeği ayırabiliyoruz: GPU içinde hangi kernel sınırlı; birden fazla GPU arasında hangi iletişim bağımlılığı var; sistem genelinde veri, güvenilirlik ve servis hedefleri nasıl etkiliyor?

Bunun için tek bir sihirli oran veya marka listesi öğrenmek zorunda değiliz. **Bir byte nereden geliyor, kaç kere kullanılıyor, nereye taşınıyor; bir hesaplama başka hangi sonucu bekliyor; nihayet hangi kalite ve gecikme hedefini mümkün kılıyor?** soruları çoğu mimari tartışmanın başlangıç noktasıdır.

Bu çalışma *bitmiş bir sorun çözme ansiklopedisi* değil. Yeni GPU nesilleri, farklı Transformer türleri, VLM'ler ve serving yöntemleri değişmeye devam edecek. Fakat veriyi, hesaplamayı, iletişimi ve kullanıcı deneyimini aynı zihinsel modelle okumak, bu yeni sistemleri öğrenmeyi çok daha kolaylaştırır.



---



## Bütün hikâye altı satırda

- 1.024 H100 GPU'lu bir süper bilgisayarda kâğıt üzerindeki 1 ExaFLOP teorik güç; hesaplama (SM/Tensor Core), bellek (HBM3), ağ (NVLink/InfiniBand) ve depolama (GDS/I-O) arasındaki fiziksel aktarım hızları dengelenmedikçe %38 MFU'da kilitlenir.
- H100 SXM'in 1.979 TFLOP/s BF16 tepe hesabı ile 3,35 TB/s HBM3 bant genişliği, $\alpha_{crit} \approx 295$ FLOP/baytlık kritik aritmetik yoğunluk (Roofline) dikte eder; bu eşiğin altındaki her işlem GPU'yu bellekten veri beklemeye mahkûm eder.
- Düğüm içi 900 GB/s NVLink ve NVSwitch çapraz bağlantıları 8-way Tensör Paralelliğini (TP) donanım hattında tutarken, düğüm dışına taşan TP 9 katlık bant genişliği uçurumuyla (50 GB/s InfiniBand) eğitimi durma noktasına getirir.
- Dağıtık LLM eğitiminde 4D paralellik (TP, PP, DP/FSDP, CP, EP) orkestrasyonunun nihai amacı, hesaplama yürütülürken arka planda kolektif iletişimi ve veri beslemeyi örtüştürerek (latency hiding) GPU'ların bekleme süresini sıfıra indirmektir.
- GPUDirect Storage ve asenkron checkpointing, host CPU'nun tampon bellek gecikmelerini baypas ederek terabaytlarca model ağırlığını doğrudan GPU HBM'den NVMe depolamaya yazar ve I/O duvarını ortadan kaldırır.
- Çıkarımda (serving) prefill aşaması matris çarpımlarıyla compute-bound iken, decode aşaması her yeni token için tüm KV cache'i HBM'den okuyarak memory-bound çalışır; bu iki iş yükünü ayrı GPU havuzlarında disaggregate etmek (ayrıştırmak) TTFT ve TPOT'u bağımsız optimize eder.

## Terimler sözlüğü

- **ALU (Arithmetic Logic Unit / Aritmetik-mantık birimi)** — Toplama ve karşılaştırma gibi temel işlem yapan devre
- **AllGather (Parçaları herkeste birleştirme)** — Herkes tüm sayfalara erişir
- **AllReduce (Azaltma + sonucu herkese dağıtma)** — Herkes toplamı öğrenir
- **AllToAll (Her katılımcıdan her katılımcıya farklı veri)** — Herkes herkese uygun parçayı gönderir
- **Arithmetic intensity (FLOP / taşınan byte)** — Veriyi ne kadar çok tekrar kullanıyoruz?
- **Backward (Geri yayılım)** — Hatanın parametrelere etkisini hesaplamak
- **Bandwidth (Bant genişliği)** — Bir saniyede ne kadar veri gider?
- **Bisection bandwidth (İki ağ yarısı arasındaki kapasite)** — Ağı ortadan böldüğümde ne kadar trafik geçer?
- **BMC (Baseboard Management Controller)** — Sunucunun işletim sisteminden bağımsız kontrolcüsü
- **Checkpoint (Eğitim durumunun kalıcı kaydı)** — Çökme sonrası devam etmek için güvenli kayıt
- **Collective (Çok katılımcılı veri haberleşmesi)** — GPU'ların ortak sonuç oluşturması
- **CP (Context Parallelism)** — Uzun token dizisini GPU'lara paylaştırma
- **Critical batch size (Kritik batch boyutu)** — GPU başına C/W civarındaki token eşiği; altına inince gradyan senkronu hesaptan uzun sürer, MoE'de E/k kat büyür.
- **CUDA core (Genel amaçlı GPU aritmetik birimi)** — Eleman bazlı hesap makineleri
- **Dataloader (Eğitim veri besleme bileşeni)** — GPU'ya sıradaki batch'i hazırlayan hat
- **DMA (Direct Memory Access)** — Verinin CPU ile tek tek kopyalanmadan donanımca taşınması
- **DP (Data Parallelism)** — Aynı modelle farklı veri parçaları
- **EP (Expert Parallelism)** — Uzman ağları GPU'lar arasında bölme
- **Fabric (Ağ bağlantıları bütünü)** — Sunucuların karşılıklı iletişim dokusu
- **FLOP/s (Floating-Point Operations per Second)** — Saniyedeki kayan noktalı işlem sayısı
- **FSDP (Fully Sharded Data Parallelism)** — Eğitim durumunu parçalayarak VRAM azaltma
- **GDS (GPUDirect Storage)** — Depolama–GPU DMA veri yolunu iyileştirme
- **GEMM (General Matrix–Matrix Multiplication)** — Büyük matris çarpımı
- **GEMV (General Matrix–Vector Multiplication)** — Matris ile vektör çarpımı
- **GQA (Grouped-Query Attention)** — Query head'lerin KV head paylaşması
- **HBM (High Bandwidth Memory)** — GPU'nun hızlı ana belleği
- **HFU (Hardware FLOPs Utilization)** — Gerçekte yürütülen FLOP'un donanım tavanına oranı
- **I/O (Input/Output / Girdi-çıktı)** — Veri okuma-yazma işleri
- **Kernel (GPU'da çalışan hesaplama fonksiyonu)** — GPU'nun tek iş parçası
- **KV cache (Attention Key/Value önbelleği)** — Geçmiş token'lara ait hatırlanan ara bilgiler
- **Latency (Gecikme)** — İstediğim bilgi ne kadar sonra geliyor?
- **L1/L2 (Önbellek düzeyleri)** — Sık kullanılan veriyi yakın tutan hafıza
- **MFU (Model FLOPs Utilization)** — Teorik tepenin ne kadarı faydalı model hesabı?
- **DualPipe (Çift yönlü pipeline zamanlaması)** — Forward/backward ve iletişim örtüşmesini iyileştirme
- **Exposed communication (Hesaplamayla örtüşmeyen iletişim)** — GPU'nun gözle görülür şekilde beklediği ağ süresi
- **FP8 (8-bit kayan noktalı gösterim)** — Veriyi daha az bitle işlemek; ölçekleme ve doğruluk dengesi gerekir
- **HSDP (Hybrid Sharded Data Parallelism)** — Sharding'i alt gruplarda, replikaları gruplar arasında yönetme
- **Kernel fusion (Birden fazla GPU işlemini uygun şekilde birleştirme)** — Kernel başlatma ve ara bellek trafiğini azaltma
- **MoE (Mixture-of-Experts)** — Token için seçilen az sayıda uzmanı çalıştırma
- **QKV merging (Attention projeksiyonlarını tek büyük GEMM'e dönüştürme)** — Üç ayrı hazırlığı ortak işlemle yapma
- **Selective activation checkpointing (Seçilmiş ara tensörleri backward'da yeniden hesaplama)** — Belleği kurtarırken hesaplama artışını sınırlama
- **Time-to-quality (Hedef model kalitesine ulaşma süresi)** — GPU benchmark'ı değil, gerçek eğitim başarısı
- **MTBF (Mean Time Between Failures)** — Arızalar arası ortalama süre
- **NUMA (Non-Uniform Memory Access)** — CPU'nun belleğe erişim mesafeleri eşit değil
- **NVLink (NVIDIA yüksek hızlı GPU bağlantısı)** — Aynı alandaki GPU'ların veri otoyolu
- **NVSwitch (NVLink anahtarlama donanımı)** — GPU bağlantılarını organize eder
- **NVFP4 (NVIDIA 4-bit kayan nokta)** — Blackwell'in E2M1 formatı; 16'lık blok başına FP8 ölçek ve tensör başına FP32 ölçek taşır, BF16'ya göre 4 kat az byte.
- **Occupancy (Aktif warp doluluğu oranı)** — SM'de ne kadar eşzamanlı warp barınabilir?
- **Pipeline bubble (Pipeline boşta kalma bölgesi)** — Bir aşama iş gelmesini bekliyor
- **PP (Pipeline Parallelism)** — Model katmanlarını aşamalar halinde bölme
- **Prefill (Prompt'u topluca işleme aşaması)** — İlk token üretilmeden önceki hazırlık
- **Register (Thread'e yakın küçük hızlı depolama)** — Aşçının elindeki malzeme
- **RDMA (Remote Direct Memory Access)** — Ağ üzerinden düşük CPU kopya maliyetiyle bellek erişimi
- **ReduceScatter (Birleştirip parçalara dağıtma)** — Toplam sonuç herkesin parçasına ayrılır
- **RoCE (RDMA over Converged Ethernet)** — Ethernet üzerinde RDMA
- **Roofline (İşlem–bellek performans modeli)** — Hangi fiziksel sınır bizi tutuyor?
- **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** — Bazı reduction işlerini switch'te yapma
- **SM (Streaming Multiprocessor)** — GPU'nun çok-thread'li hesaplama istasyonu
- **SP (Sequence Parallelism)** — Bazı aktivasyonları sequence ekseninde bölme
- **Tensor Core (Özel matris hesaplama birimi)** — Büyük çarpımlarda seri üretim makinesi
- **TMA (Tensor Memory Accelerator)** — GPU içi blok veri taşıma motoru
- **TP (Tensor Parallelism)** — Tek katmanın matrisini GPU'lara dağıtma
- **Warp (32 CUDA thread'den oluşan planlama grubu)** — Birlikte komut alan ekip
- **Warp divergence (Warp içindeki kontrol yolu ayrışması)** — Ekibin üyeleri farklı yollara gidiyor
- **WGMMA (Warpgroup Matrix Multiply-Accumulate)** — Hopper'ın warp grubu ölçeğindeki matris komutları
- **Young–Daly (Optimum checkpoint aralığı yaklaşımı)** — Çok sık/yavaş kayıt ile kayıp iş dengesi
- **Admission control (Kapasiteye göre iş kabulü)** — Sistem tıkanmadan yeni istek alıp almayacağına karar verir
- **Control plane (Yönetim/karar düzlemi)** — Kim hangi kaynağı, hangi kuralla kullanacak?
- **Data plane (İşin yürütüldüğü düzlem)** — GPU'nun gerçek hesabı ve veri akışı
- **Dataset lineage (Eğitim verisinin sürüm ve dönüşüm geçmişi)** — Bir model tam olarak hangi örneklerden öğrenmiş?
- **Disaggregated serving (Prefill ve decode havuzlarını ayırma)** — İki işi kendi ihtiyacına göre ayrı büyütme
- **Gang scheduling (Bağımlı çalışanları birlikte başlatabilen kabul/planlama)** — Yarım ekiple eğitim başlatmamak
- **Goodput (Hizmet gereksinimini karşılayan yararlı throughput)** — Hızlı ama geç kalan yanıtı başarı saymamak
- **GQA (Grouped-Query Attention)** — Çok sayıda query head, daha az KV head paylaşır
- **ISL / OSL (Input / Output Sequence Length)** — İsteğe gelen ve üretilen token sayıları
- **KV-aware routing (Önbellek yerelliğine duyarlı iş yönlendirme)** — Aynı prefix'i bilen worker'a gitmenin avantajı
- **Little's Law (Kararlı sistemde L = λW ilişkisi)** — Geliş hızı ve bekleme süresinden ortalama eşzamanlılık
- **MIG (Multi-Instance GPU)** — Desteklenen GPU'yu yalıtılmış parçalara ayırma
- **PagedAttention (Sayfalanmış KV bellek yönetimi)** — Uzun konuşmalar için tek parça bellek zorunlu değil
- **PUE (Power Usage Effectiveness)** — Veri merkezi enerjisi / IT enerjisi
- **RPO / RTO (Kurtarmada kabul edilen veri/ilerleme kaybı ve azami toparlanma süresi)** — Ne kadar eğitim işi kaybedebiliriz; kaç dakikada döneriz?
- **RAS (Reliability, Availability, Serviceability)** — Arızayı algıla, hizmeti sürdür, onar
- **SDC (Silent Data Corruption)** — Açık hata olmadan yanlış sonuç oluşması
- **SLO (Service Level Objective)** — Ölçülüp izlenen hizmet hedefi
- **Speculative decoding (Taslak token'ları hedef modelle doğrulama)** — Token üretimindeki seri darboğazı azaltma denemesi
- **Strong scaling (Sabit işi daha çok hızlandırıcıda çalıştırma)** — Aynı eğitim işi daha hızlı bitiyor mu?
- **TCO (Total Cost of Ownership)** — Donanım + enerji + ağ + işletim toplam maliyeti
- **Tenant isolation (Müşteri/ekip sınırlarının ayrılması)** — Bir kullanıcının verisi/işi diğerine karışmasın
- **Time-to-quality (Hedef kaliteye ulaşana kadar geçen zaman)** — Hızlı adım değil, doğru sonuç önemli
- **Weak scaling (GPU ve iş miktarını beraber büyütme)** — İş yüküyle birlikte ölçek büyüyünce verim korunuyor mu?

## Kaynaklar ve doğrulama bağlantıları

Aşağıdaki kaynakların bir bölümü **orijinal teknik bağlam**, bir bölümü **bağımsız üretici dokümantasyonu veya makale doğrulaması**dır. Kaynakların hiçbirindeki reklam amaçlı üst sınır, gerçek uygulama throughput'u ile otomatik olarak eşit kabul edilmemiştir.

- **[K1] Ana kaynak:** Engin Bozaba, [*Understanding GPUs and LLM Scaling*](https://muhendis.github.io/en/blog/post.html?slug=understanding-gpus-and-llm-scaling) — [doğrudan Markdown kaynağı](https://github.com/muhendis/muhendis.github.io/blob/master/en/blog/posts/understanding-gpus-and-llm-scaling.md). Compute–network–storage eksenleri, SM → DGX → SuperPOD hiyerarşisi ve öğretici donanım bağlamı.
- **[K2] Donanım mikro-mimarisi:** NVIDIA, [*Hopper Architecture In-Depth*](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — H100 SXM SM/Tensor Core/L2/HBM/NVLink bileşenleri.
- **[K3] Dağıtık model stratejileri:** NVIDIA, [*Megatron Core Parallelism Strategies Guide*](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) ve [*Context Parallelism*](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html). TP, PP, CP, EP ve FSDP.
- **[K4] DGX SuperPOD referansı:** NVIDIA, [*Network Fabrics*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html) ve [*Major Components*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/components.html). Rail topology, 127 node, 32 leaf + 16 spine, ayrı depolama ağı.
- **[K5] Gerçek ölçekli eğitim:** Dubey ve diğerleri, [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783), 2024. 16.384 H100, 4D paralellik, MFU aralıkları, Tectonic ve 54 günlük duraklama istatistikleri.
- **[K6] GPU–storage veri yolu:** NVIDIA, [*GPUDirect Storage Design Guide*](https://docs.nvidia.com/gpudirect-storage/design-guide/) ve [*Overview Guide*](https://docs.nvidia.com/gpudirect-storage/overview-guide/). CPU bounce buffer ve compatibility mode ayrıntıları.
- **[K7] Ağ içinde hesaplama:** NVIDIA, [*SHARP In-Network Computing*](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing). Collective acceleration.
- **[K8] Attention kernel optimizasyonu:** Shah ve diğerleri, [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/abs/2407.08608), 2024; [PyTorch teknik açıklaması](https://docs.pytorch.org/blog/flashattention-3/).
- **[K9] MoE ve DualPipe:** DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) ve [DualPipe kodu](https://github.com/deepseek-ai/DualPipe).
- **[K10] Checkpoint teorisi:** [*A survey on checkpointing strategies: Should we always checkpoint à la Young/Daly?*](https://www.sciencedirect.com/science/article/pii/S0167739X24003777), 2024; klasik $\sqrt{2MC}$ yaklaşımının varsayımları.
- **[K12] MFU optimizasyonunda somut Blackwell sonuçları:** Lambda, [*Reproducible MFU Optimization Techniques to Boost Your Training Efficiency*](https://lambda.ai/hubfs/4.%20Resources/White%20Papers/Lambda%20MFU.pdf), Kasım 2025; [benchmark özeti](https://lambda.ai/llama-3-1-blackwell-mfu-benchmarks). Llama 3.1 8B/70B/405B, TorchTitan, TP, selective activation checkpointing, torch.compile, prefetch ve BF16/FP8 karşılaştırmaları.
- **[K13] 10 bin+ GPU'da MFU:** Jiang ve diğerleri, [*MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs*](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng), NSDI 2024; [teknik makale PDF](https://www.usenix.org/system/files/nsdi24-jiang-ziheng.pdf). 175B/12.288 GPU üzerinde %55,2 MFU.
- **[K14] `torch.compile` sınırları:** PyTorch, [*torch.compile API*](https://docs.pytorch.org/docs/stable/generated/torch.compile) ve [*Introduction to torch.compile*](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). Graph breaks, kernel optimizasyonu ve dinamik boyutlar.
- **[K15] Aktivasyon belleği ve recomputation:** PyTorch, [*Current and New Activation Checkpointing Techniques*](https://pytorch.org/blog/activation-checkpointing-techniques/). Selective/full checkpointing ve bellek-hesaplama takası.
- **[K16] Precision'a göre teorik FLOP tepe değerleri:** NVIDIA, [*Exemplar Performance — Peak Theoretical Throughput*](https://github.com/NVIDIA/exemplar-performance/blob/main/README.md). H100, B200, B300, GB200, GB300 için non-sparse BF16/FP8 referansları.
- **[K17] FP8 ve karma hassasiyetin sınırları:** NVIDIA, [*Transformer Engine Data Type Support Matrix*](https://nvidia.github.io/TransformerEngine/support_matrix.html) ve [*GEMM Profiling Tutorial*](https://docs.nvidia.com/deeplearning/transformer-engine/examples/gemm_profiling/gemm_profiling.html). FP8 destek matrisi, GEMM dışı operasyonlar ve düşük precision ek maliyetleri.
- **[K11] Ölçeklendirme için ilave okuma:** Austin ve diğerleri, [*How to Scale Your Model: How to Think About GPUs*](https://jax-ml.github.io/scaling-book/gpus/). GPU mimarisi, kolektif iletişim ve model paralelleştirme üzerine hesaplı öğretici örnekler.

- **[K18] Kurumsal risk ve veri yönetişimi:** NIST, [*AI Risk Management Framework: Generative AI Profile*](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), 2024. Veri kökeni, mahremiyet, model riskleri ve yaşam döngüsü yönetişimi.
- **[K19] Decode optimizasyonu:** vLLM, [*Speculative Decoding*](https://docs.vllm.ai/en/latest/features/speculative_decoding/). Kabul oranı, yöntemler ve performansın trafik koşullarına bağlılığı.
- **[K20] Üretim inference bileşenleri:** NVIDIA, [*Dynamo Documentation*](https://docs.nvidia.com/dynamo/latest/). Router, prefill/decode dağıtımı, KV yönetimi ve ölçekleme.
- **[K21] KV cache boyutu için mimari parametreler:** [*Llama 3.1 70B model konfigürasyonu*](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json). 80 katman, 8 KV head, hidden size 8192 ve 64 attention head bilgileri; hesap formülü öğretici türetimdir.
- **[K22] Sayfalanmış attention ve KV cache:** vLLM, [*Automatic Prefix Caching*](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) ve [*Cache Design / Isolation*](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md) ve [*vLLM Security*](https://docs.vllm.ai/en/latest/usage/security/). KV block yönetimi, prefix yeniden kullanımı ve yalıtım.
- **[K23] Ayrık inference topolojisi:** NVIDIA Dynamo, [*Disaggregated Serving*](https://docs.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview) ve [*KV-Aware Routing*](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving). Prefill/decode worker'ları ve KV taşımanın maliyeti.
- **[K24] Eğitim işi planlama:** Kubernetes SIGs, [*Kueue All-or-Nothing Scheduling*](https://github.com/kubernetes-sigs/kueue/blob/main/site/content/en/v0.20/docs/concepts/all_or_nothing.md). Birden fazla worker'ın beraber kaynak kabulü.
- **[K25] GPU paylaştırma:** NVIDIA, [*MIG User Guide*](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/introduction.html). GPU bölümleme ve destek koşulları.
- **[K26] Sistem güç referansı:** NVIDIA, [*DGX H100/H200 System Specifications*](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). 10,2 kW azami sistem güç spesifikasyonu.
- **[K27] Fiziksel tesis tasarımı:** Uptime Institute, [*AI embraces liquid cooling*](https://journal.uptimeinstitute.com/ai-embraces-liquid-cooling-but-enterprise-it-is-slow-to-follow/), 2025. Rack yoğunluğu, soğutma ve güç planlaması.
- **[K28] Eğitim veri kalitesi:** Meta, [*Introducing Meta Llama 3*](https://ai.meta.com/blog/meta-llama-3), 2024. Veri filtreleme, deduplication ve kalite temelli seçim.
- **[K29] Eğitim veri ve model yönetimi:** NVIDIA, [*Megatron Bridge Documentation*](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html) ve [*Data Preparation*](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/training/data-preparation.md). Eğitim reçeteleri ve indexed dataset yaklaşımı.
- **[K30] Ölçekleme kuralları:** Hoffmann ve diğerleri, [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556), 2022. Parametre sayısı, token miktarı ve sabit hesaplama bütçesi ilişkisi.
- **[K31] Ölçüm metodolojisi:** MLCommons, [*MLPerf Training*](https://mlcommons.org/benchmarks/training/) ve [*Training Rules*](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc). Hedef kaliteye kadar eğitim süresinin ölçülmesi.
- **[K32] Kolektif iletişim sağlığı:** NVIDIA, [*NCCL RAS Documentation*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html). Çalışan dağıtık işte RAS ve sorun teşhisi.
- **[K33] GPU node izolasyonu:** NVIDIA, [*DGX SuperPOD Health Checks*](https://docs.nvidia.com/dgx-superpod/administration-guide-dgx-superpod/latest/system-health.html). Test etme, node'u havuzdan çıkarma ve NCCL validasyonu.
- **[K34] Üretim inference benchmark'ı:** MLCommons, [*What is MLPerf Endpoints?*](https://docs.mlcommons.org/endpoints-submission-docs/understand/what-is-mlperf-endpoints/). Concurrency–throughput–latency Pareto eğrisi.
- **[K35] GPU telemetri:** NVIDIA, [*DCGM Exporter*](https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html). GPU metriklerinin Prometheus formatında toplanması.

- **[K36] DCGM sağlık kontrolleri:** NVIDIA, [*Health Monitoring*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/health-monitoring.html) ve [*Diagnostics*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/dcgm-diagnostics.html). Xid, ECC, page retirement, PCIe ve stres teşhisi.
- **[K37] DCGM telemetri/alan sınırları:** NVIDIA, [*Profiling*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html) ve [*DCGM Exporter Metrics*](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html). Metrik destek sınırı, sampling, multiplexing ve exporter ayarları.
- **[K38] NCCL teşhis:** NVIDIA, [*NCCL RAS*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html) ve [*Diagnostics*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/diagnostics.html). Pasif/aktif teşhis, tutarsız rank ve iletişim sorunları.
- **[K39] NCCL mikrobenchmark:** NVIDIA, [*nccl-tests*](https://github.com/NVIDIA/nccl-tests). Bir-node ve multi-node AllReduce performans kıyaslama örnekleri.
- **[K40] Disaggregated inference kararı:** NVIDIA Dynamo, [*Disaggregated Serving Overview*](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/disaggregated-serving/overview.md). Ayrık prefill/decode havuzları, KV taşıma maliyeti, ne zaman tercih edilmeyeceği.
- **[K41] Eğitim performansı / kalitesi:** MLCommons, [*MLPerf Training Reference Implementations*](https://github.com/mlcommons/training). Hedef kaliteye kadar eğitim kriteri ve tekrarlanabilir benchmark akışı.
- **[K42] Inference benchmark yöntemleri:** MLCommons, [*MLPerf Inference Benchmarks*](https://docs.mlcommons.org/inference/benchmarks/) ve [*Submission Guide*](https://docs.mlcommons.org/inference/submission/). Latans ve doğruluk şartları, farklı yük senaryoları.
- **[K43] İstek-temelli serving benchmark:** SGLang, [*Serving Benchmark Guide*](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx). TTFT/TPOT/ITL, request throughput, input-output token dağılımı.
- **[K44] Çok modlu model işlemcisi:** Hugging Face, [*Multimodal Chat Templates*](https://huggingface.co/docs/transformers/chat_templating_multimodal). Görüntü/video preprocessor, multimodal processor ve frame seçim parametreleri.
- **[K45] GPU paylaşımı / izolasyon:** NVIDIA, [*MIG Concepts*](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/concepts.html) ve [*Introduction*](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html). Streams/MPS/MIG işlev ve izolasyon farkları.
- **[K46] Kubernetes hızlandırıcı atama mimarisi:** Kubernetes, [*Dynamic Resource Allocation*](https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/) ve [*v1.34 DRA GA*](https://kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/). DeviceClass/ResourceClaim ve sürüme göre değişen cihaz atama olanakları.


- **[K47] Profiling ve ölçüm sınırları:** NVIDIA, [*Nsight Systems*](https://developer.nvidia.com/nsight-systems) ve [*User Guide*](https://docs.nvidia.com/nsight-systems/UserGuide/). CUDA trace, GPU/NVLink/DRAM sayaçları, multi-node timeline; kernel ayrıntılarında [*Nsight Compute Profiling Guide*](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- **[K48] FSDP2'nin gerçek mekanizması:** PyTorch, [*Fully Sharded Data Parallel — `fully_shard`*](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) ve [*Getting Started with FSDP2*](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html). DTensor sharding, katman öncesi AllGather, backward ReduceScatter ve prefetch.
- **[K49] PagedAttention birincil çalışma:** Kwon ve diğerleri, [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://doi.org/10.1145/3600006.3613165), SOSP 2023. KV cache bloklama, bellek fragmentation ve vLLM throughput deneyleri.
- **[K50] Hız/kalite ayrımı:** MLCommons, [*MLPerf Training Rules*](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc). Hedef kaliteyi karşılama ve wallclock eğitim süresi.
- **[K51] Rack-scale NVLink alanı:** NVIDIA, [*GB300 NVL72 specifications*](https://www.nvidia.com/en-us/data-center/gb300-nvl72/). 72 Blackwell Ultra GPU, Grace CPU ve NVLink bağlantı alanı; üretici tepe rakamlarında sparsity koşullarını ayrı okuyun.
- **[K52] Hopper'da ölçülmüş gecikmeler:** Luo ve diğerleri, [*Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*](https://arxiv.org/abs/2402.13499), 2024. Yükleme gecikmeleri: SMEM ~29, L1 ~41, L2 ~263, global bellek ~479 döngü.
- **[K53] NVSwitch ağının açılması:** NVIDIA, [*Fabric Manager for NVIDIA NVSwitch Systems*](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). Her GPU'nun 18 NVLink'inin dört NVSwitch'e 5 + 4 + 4 + 5 dağılımı ve NVLink ağının neden Fabric Manager gerektirdiği.
- **[K54] InfiniBand switch:** NVIDIA, [*QM97X0 NDR 400Gb/s InfiniBand Switch Systems User Manual*](https://docs.nvidia.com/networking/display/qm97x0um/introduction). QM9700 başına 64 NDR port ve SHARP ağ içi azaltma.
- **[K55] SuperPOD ölçekleme:** NVIDIA, [*DGX SuperPOD Architecture (DGX H100)*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html). 8 SU'ya kadar iki katman, 16 SU'dan itibaren core katmanı; 2.048 DGX için 512 leaf, 512 spine ve 256 core switch.
- **[K56] GPU nesli veri sayfaları:** NVIDIA, [V100](https://www.nvidia.com/en-us/data-center/v100/), [A100](https://www.nvidia.com/en-us/data-center/a100/), [H200](https://www.nvidia.com/en-us/data-center/h200/) ve [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/). HBM kapasitesi ve bant genişliği, yoğun tensor tepe değerleri.
- **[K57] 4-bit kayan nokta:** NVIDIA, [*Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/). E2M1 değerler, E4M3 ölçekli 16'lık bloklar, FP32 tensör ölçeği ve MXFP4 karşılaştırması.

### Yayın ve kullanım notu

Bu makale, farklı teknik kaynaklardan edinilen bilgileri **Türkçe, özgün ve öğretici bir anlatımla bir araya getiren bilgi paylaşımıdır**. Bir üretim sistemine özel kurulum, güvenlik, arıza müdahalesi veya optimizasyon prosedürü sunmaz. Sayılar üç türde değerlendirilmelidir: **üretici teknik özellikleri**, **bağımsız çalışmalarda raporlanmış ölçümler** ve **varsayım içeren öğretici modeller**. Çalıştırılan platform, firmware, sürücü, sayısal hassasiyet ve workload'a bağlı değişebilen ölçümler için gerçek sistem benchmark'ı gereklidir.