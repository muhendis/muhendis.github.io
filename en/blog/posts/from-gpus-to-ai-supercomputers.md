*From 32 threads inside a single GPU to a 16,384-GPU training cluster, and onward to LLM/VLM systems: let's explore compute, memory, networking, storage, security, and scaling through intuitive examples, mathematics, and published technical papers.*

Imagine walking into a data center. Before you stand row upon row of server racks, each housing GPUs capable of executing thousands of parallel operations. Suppose you have purchased a total of **1,024 NVIDIA H100** GPUs. Looking purely at dense BF16 Tensor Core compute, you possess on paper a peak throughput of approximately **1 exaFLOP** per second.

The team excitedly kicks off training for a large language model. The first metrics arrive: **Model FLOPs Utilization (MFU) = 38%**.

Someone asks: "We bought all these GPUs—where did the remaining 62% go?"

This is an excellent question. But the answer is not as simple as "the GPUs are idle." A GPU might be stalling while waiting for data from memory. Another GPU may be waiting on gradients arriving from across the network. Yet another stage might be blocked while a checkpoint is being committed to storage. In fact, a GPU can appear 100% busy while performing far less useful mathematical work than expected.

**The central thesis of this article is this:** Large-scale AI is not merely a *compute* problem. It is equally a problem of **delivering data to the right place, at the right time, within the right security boundaries, and at the lowest possible cost.**

To truly understand this, we must first look inside the GPU itself. What is a thread? Why is a warp grouped into 32 threads? What does a Tensor Core do, and why is HBM so critical? Next, we will see how these same foundational concepts scale to an eight-GPU DGX server, and subsequently to a multi-thousand-GPU data center. Finally, we will trace a single training step from start to finish and seek evidence-based answers to the question: "Why is MFU so low?"

> **How to read this article:** When a concept is introduced for the first time, I will first provide an intuitive analogy and explain its purpose. Then, I will lay out its technical definition along with concrete calculations where applicable. The tables and numbers are not intended for rote memorization; they exist to help you build a clear mental model of how the system actually behaves.

**The purpose of this article is knowledge sharing.** This is not an installation guide for a specific product, an off-the-shelf architecture recipe, or an exhaustive troubleshooting runbook. Instead, we explore the foundational mechanisms spanning from a single GPU to massive clusters, model workloads, and their direct relationships to capacity, reliability, security, user experience, and cost—all within a single cohesive learning journey. The tables, sample test workflows, and commands are designed to make theoretical concepts tangible; they should not be treated as production-validated procedures to be executed blindly on a live system.

### Our Learning Roadmap

1. **Inside the GPU:** Thread → Warp → SM → Tensor Core; how does the GPU organize compute?
2. **GPU Memory Hierarchy:** Register → Shared Memory → L2 → HBM; why do operations stall waiting for data?
3. **How LLMs Execute:** Training, prefill, decode, KV cache, quantization; why does the same model hit completely different bottlenecks across phases?
4. **From a Single Server to Massive Clusters:** NVLink, NVSwitch, NICs, InfiniBand/RoCE, network topologies; how do GPUs communicate?
5. **Distributed Training:** TP, PP, DP, FSDP, CP, EP; where and how do we partition the model?
6. **End-to-End Execution:** A full training step, storage, checkpointing, failure recovery, and diagnosing MFU.
7. **Achieving Real Performance Gains:** Published MFU benchmarks on B200/GB300, FP8, kernel fusion, communication-compute overlap, and a practical measurement playbook for your own cluster.
8. **Understanding the Broader Architecture:** Production inference, orchestration, data governance, reliability, SLOs, energy consumption, and total cost of ownership (TCO).
9. **Engineering Case Studies:** Calculating GPU serving capacity; measuring network fabrics; architectural trade-offs in VLM/VQA, RAG/agents, and system security.
10. **A Concrete End-to-End Walkthrough:** Analyzing a hypothetical architecture for a 70B model, evaluating operational layers, and working through hands-on sizing calculations.

> **Numerical Accuracy Note:** Unless specified otherwise, specifications regarding the H100 refer to the **H100 SXM** form factor and its associated DGX H100 reference architecture. "Theoretical ceiling", "pedagogical assumption", and "reported empirical measurement" represent distinct categories and should not be conflated. You will find references [K1]–[K51] at the end of the article. This piece has been re-architected and expanded by synthesizing the context of [Understanding GPUs and LLM Scaling](https://muhendis.github.io/en/blog/post.html?slug=understanding-gpus-and-llm-scaling) with prior technical drafts.

> [!NOTE]
> **Part 1 of the Series: Hardware Anatomy**
> To explore the physical hardware layers of an AI supercomputer (SM microarchitecture, HBM3 memory hierarchy, in-chassis NVLink crossbars, and SuperPOD rail-optimized fat-tree fabrics) via detailed structural maps, see the first article in this series: [AI Infrastructure (1): Anatomy of an AI Supercomputer — From a Single SM to 16,384 GPUs on One Map](post.html?slug=understanding-gpus-and-llm-scaling).

---

### The Methodology: A Mental Model Instead of Rote Memorization

When reading about large GPU clusters, one often ends up with a disjointed vocabulary of buzzwords: *warp, SM, HBM, NCCL, MFU, FSDP, KV cache...* Defining them in isolation is straightforward; understanding **why they depend on one another** is far more valuable. Throughout this article, we will repeatedly return to a single guiding question: **Where does a byte of data originate, what computation does it enable, where does it travel next, and who is waiting on whom along that journey?**

We will follow two distinct narratives. In the first, we train a model: data is ingested, the forward pass is computed, the loss is evaluated, gradients are calculated in the backward pass, GPUs synchronize, and model parameters are updated. In the second, the model serves an end-user request: the query enters a queue, the prompt undergoes the *prefill* phase, and subsequent tokens are generated sequentially during the *decode* phase. The exact same GPU hits vastly different operational bottlenecks in each of these two scenarios.

To navigate this complexity, we will evaluate every core concept through a four-stage framework:

1. **Intuition:** Build an intuitive feel for the problem using a familiar analogy.
2. **Mechanism:** Identify the exact underlying hardware or software components at work.
3. **Mathematics:** Expose the bottleneck using straightforward analytical equations.
4. **Boundaries:** Discuss where the simplified formula or analogy breaks down in real systems.

Throughout this article, **vendor peak specifications**, **empirically reported benchmarks from research papers**, and **pedagogical calculation examples** are kept strictly distinct. A given technology may yield vastly different results across hardware generations or model architectures. The goal is not to prescribe a single "dogmatic" architecture, but rather to establish a rigorous, shared vocabulary for analyzing system behavior.

### A One-Sentence Map of the Architecture



<svg viewBox="0 0 720 240" role="img" aria-label="Cause-and-effect map. A byte travels from storage through the CPU and host RAM into HBM, then into the SMs and Tensor Cores, and produces a loss or the next token. Between storage and the CPU lies the I/O wall; between HBM and the SMs lies the memory wall. If more than one GPU is involved, results also travel over NVLink or InfiniBand collectives back into HBM: that is the fabric wall." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="mp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="mp-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker><marker id="mp-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">One byte&#x27;s journey — and the three walls it can hit</text>
<rect x="16" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="76.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Storage</text>
<text x="76.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dataset, checkpoint</text>
<rect x="158" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="218.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU + host RAM</text>
<text x="218.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">DataLoader, PCIe</text>
<rect x="300" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">HBM3</text>
<text x="360.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80 GB · 3.35 TB/s</text>
<rect x="442" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="502.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">SM + Tensor Core</text>
<text x="502.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">~990 TFLOP/s BF16</text>
<rect x="584" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="644.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Result</text>
<text x="644.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">loss / next token</text>
<line x1="136" y1="71" x2="156" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="278" y1="71" x2="298" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="420" y1="71" x2="440" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="562" y1="71" x2="582" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="113.896" y="104" width="66.208" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="147.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">I/O Wall</text>
<rect x="387.732" y="104" width="86.536" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="431.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Memory Wall</text>
<line x1="522" y1="98" x2="522" y2="134" marker-end="url(#mp-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="330" y="136" width="210" height="46" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="435.0" y="156.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">NVLink / InfiniBand</text>
<text x="435.0" y="171.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">collectives (AllReduce …)</text>
<line x1="360" y1="136" x2="360" y2="100" marker-end="url(#mp-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="530" y="122" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">more than one GPU?</text>
<rect x="391.732" y="190" width="86.536" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="435.0" y="202.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Fabric Wall</text>
<text x="16" y="228" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Every chapter asks the same question: where does this byte wait, and who waits for it?</text>
</svg>



This diagram is not an exhaustive hardware topology; rather, it is a **cause-and-effect map** to reference as you read. By the end of this article, you will understand the precise technical reality behind every single arrow.




---

**PART I — Why Are GPUs Fast, and Why Do They Stall?**

*First, we will examine the fundamental units of computation, and then trace how data flows inside the GPU.*


## 1. The Three Physical Walls: Compute, Network, and Storage

Consider a restaurant chain. To increase output, management sends twice as many chefs to every branch. However, the ingredient storage rooms remain unchanged; the delivery trucks remain the same; the physical doors through which ingredients pass into the kitchen are identically sized. Will the kitchen produce twice as many meals?

Almost certainly not. In this system, the bottleneck is no longer the number of chefs. Even the world's most talented chef sits idle waiting for ingredients to arrive. In a massive GPU cluster, the core challenge is often just as mundane.

Think of an AI compute cluster as a distributed culinary network:

- **Streaming Multiprocessor (SM):** A self-contained kitchen prep station.
- **Tensor Core:** A specialized industrial appliance performing large volumes of identical math operations simultaneously.
- **Register / Shared Memory:** The chef’s immediate countertop and prep surface.
- **HBM (High Bandwidth Memory):** The restaurant’s fast but capacity-constrained on-site pantry.
- **NVLink / NVSwitch:** The wide, high-speed service corridor connecting kitchens within the same building.
- **InfiniBand / RoCE:** The inter-facility highway network linking separate buildings.
- **Distributed Storage:** The central warehouse where raw bulk ingredients and long-term business ledgers are stored.

This analogy highlights a crucial reality: raw GPU count alone does not dictate overall system throughput. Different layers of the production pipeline operate at vastly different speeds, and whenever a mismatch arises between these rates, one component inevitably stalls waiting on another.

Adding more kitchens will not yield more meals if the supply chain breaks down. If ingredients cannot pass through the doorway quickly enough, you hit the **Memory Wall**. If kitchens spend all their time coordinating with one another, you hit the **Fabric Wall**. If raw supplies or the logging system cannot keep up, you hit the **I/O Wall**.

| Wall | Core Question | Typical Symptom | Primary Remedy |
|---|---|---|---|
| **Memory Wall** | Is moving data more expensive than computing on it? | High HBM bandwidth utilization, low FLOP/s | Increase data reuse; batching, tiling, kernel fusion, lower-precision weights |
| **Fabric Wall** | How much time do GPUs spend communicating? | Elongated NCCL collectives, idle pipeline bubbles | Topology-aware TP/PP/DP layouts, communication-computation overlap |
| **I/O Wall** | Can token pipelines or checkpoint I/O keep pace? | Dataloader stalls, periodic checkpoint freezes | Prefetching, sharded I/O, asynchronous checkpointing, GPU Direct Storage (GDS) |

### 1.1. The Hierarchy of Physical Scales

**First, frame the problem at the right scale.**

If a CUDA kernel is throttling due to register pressure, buying a newer InfiniBand switch will not speed it up. Conversely, if training stalls waiting on an inter-node network bottleneck, shaving off a few registers inside an SM will not resolve the issue. We must clearly delineate which layer of the system is responsible for which class of behavior.



<svg viewBox="0 0 720 388" role="img" aria-label="Scale ladder from bottom to top: a thread; a warp of 32 threads; an SM holding up to 64 warps or 2,048 threads; an H100 SXM with 132 SMs, 80 GB HBM3 and 3.35 TB/s; a DGX H100 with 8 GPUs on NVSwitch; a scalable unit of 32 DGX or 256 GPUs; a SuperPOD of 4 SUs and 127 DGX or 1,016 GPUs; and a multi-pod cluster of 16,384 H100s as used for Llama 3 405B. Beside each rung is where its typical bottleneck lives: branch divergence, latency hiding, registers and shared memory, HBM bandwidth, TP collectives over NVLink, leaf switches and rails, DP and PP over InfiniBand, and finally storage, failures and power." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Scale ladder: each rung multiplies the one below</text>
<text x="704" y="24" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">where the typical bottleneck lives</text>
<rect x="70" y="44" width="386" height="32" rx="6" style="fill:var(--c-danger);fill-opacity:.10;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="82" y="64" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Multi-pod cluster</text>
<text x="217.55" y="64" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">16,384 H100 (e.g. Llama 3 405B, RoCE)</text>
<text x="60" y="64" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)"></text>
<line x1="462" y1="60" x2="494" y2="60" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12px">storage, failures, power</text>
<rect x="80" y="84" width="376" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="92" y="104" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">SuperPOD</text>
<text x="163.2" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">4 SU · 127 DGX = 1,016 GPUs</text>
<text x="70" y="104" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×4</text>
<line x1="462" y1="100" x2="494" y2="100" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="104" text-anchor="start" style="fill:var(--c-text);font-size:12px">DP/PP over InfiniBand</text>
<rect x="90" y="124" width="366" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="102" y="144" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Scalable Unit</text>
<text x="208.95" y="144" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">32 DGX = 256 GPUs</text>
<text x="80" y="144" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×32</text>
<line x1="462" y1="140" x2="494" y2="140" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="144" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf switch, rails</text>
<rect x="100" y="164" width="356" height="32" rx="6" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="112" y="184" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX H100</text>
<text x="183.2" y="184" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">8 GPUs · NVSwitch · 900 GB/s NVLink</text>
<text x="90" y="184" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×8</text>
<line x1="462" y1="180" x2="494" y2="180" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="184" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP collectives (NVLink)</text>
<rect x="110" y="204" width="346" height="32" rx="6" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="122" y="224" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">H100 SXM</text>
<text x="193.2" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">132 SMs · 80 GB HBM3 · 3.35 TB/s</text>
<text x="100" y="224" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×132</text>
<line x1="462" y1="220" x2="494" y2="220" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="224" text-anchor="start" style="fill:var(--c-text);font-size:12px">HBM bandwidth (Roofline)</text>
<rect x="120" y="244" width="336" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="132" y="264" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">SM</text>
<text x="160.3" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">≤ 64 warps = 2,048 threads</text>
<text x="110" y="264" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×64</text>
<line x1="462" y1="260" x2="494" y2="260" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="264" text-anchor="start" style="fill:var(--c-text);font-size:12px">registers, shared memory</text>
<rect x="130" y="284" width="326" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="142" y="304" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Warp</text>
<text x="184.6" y="304" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">32 threads, one instruction stream</text>
<text x="120" y="304" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×32</text>
<line x1="462" y1="300" x2="494" y2="300" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="304" text-anchor="start" style="fill:var(--c-text);font-size:12px">latency hiding</text>
<rect x="140" y="324" width="316" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152" y="344" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Thread</text>
<text x="208.9" y="344" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">one execution instance</text>
<text x="130" y="344" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">1</text>
<line x1="462" y1="340" x2="494" y2="340" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="344" text-anchor="start" style="fill:var(--c-text);font-size:12px">branch divergence</text>
<text x="16" y="376" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">1,024 = 128 × 8 is the design capacity of 4 SUs; the reference SuperPOD gives one slot to UFM, leaving 127 DGX = 1,016 GPUs.</text>
</svg>



**1,024 or 1,016 GPUs?** Four Scalable Units (SUs) of 32 servers each represent a nominal design capacity of 128 × 8 = **1,024 GPUs**. However, in certain NVIDIA DGX H100 SuperPOD reference architectures, one server slot is reserved for network management / UFM services, yielding **127 DGX nodes = 1,016 GPUs**. Conflating these two numbers as if they describe identical deployment configurations is a common pitfall. [K4]

Once we internalize this physical hierarchy, we can answer the fundamental question: *at what level must a given problem be solved?* Register pressure lives strictly within an SM; Tensor Parallelism (TP) overhead lives in the intra-node NVLink domain; gradient synchronization predominantly impacts the inter-node network fabric; and checkpoint stalls manifest within the distributed storage subsystem.

---


## 2. What Is Inside a GPU? Threads, Warps, SMs, and Tensor Cores

### 2.1. Why Don't CPUs Work Like GPUs?

Calling a GPU an "extra-powerful processor" is not strictly wrong, but it misses the point. A GPU's true strength does not stem from finishing an isolated task at blinding single-thread speed; rather, it lies in its ability to **execute an enormous number of similar operations simultaneously**. This is where its fundamental divergence from the CPU becomes apparent.

A CPU is engineered with a relatively small number of highly sophisticated, low-latency cores optimized for branch-heavy, irregular, and latency-sensitive instruction streams. In contrast, a GPU devotes the vast majority of its silicon area to execution units, prioritizing the high-throughput processing of thousands of concurrent arithmetic operations.

Consider this difference as assigning a task to two different workforces. On a CPU, a few world-class generalist engineers rapidly tackle complex, shifting assignments sequentially. On a GPU, an army of thousands of specialized workers processes uniform sub-tasks simultaneously. The latter paradigm is phenomenally effective for dense matrix multiplication; however, if the task is tiny or highly irregular, coordination overhead quickly erodes the parallelism advantage.

The same comparison with Google's TPU is instructive. A TPU chip has a few large TensorCores built around 128 × 128 systolic matrix units (MXUs): when the work arrives as big, regular matrices these giant cutting boards are extremely efficient, and when it is small or irregular they sit idle. An H100 is instead a crew of 132 independent SMs, each scheduling its own warps. Between chips, TPUs are wired into fixed 2D/3D tori, whereas a GPU cluster is a **layered** hierarchy of NVLink domains, a leaf–spine fabric and a storage network; the rest of this article is about what waits on which layer. [K11]

### 2.2. CUDA Threads, Warps, and SIMT

Let us first examine the smallest logical unit of work. Imagine applying a blur filter across every pixel of a high-resolution image. You write the operation as a single logical program; the GPU instantiates thousands of execution instances across the image. Each instance computes a single pixel. One worker, one pixel; 32 workers grouped together, a warp...

A **Thread** is the finest logical execution instance of a program on a GPU. A **Warp** is a tightly coupled group of 32 threads scheduled and dispatched together in lockstep on NVIDIA architectures. **SIMT (Single Instruction, Multiple Threads)** is the execution model where individual threads maintain their own private registers, execution state, and data paths, while instruction fetch and dispatch are amortized across the entire 32-thread warp.

<svg viewBox="0 0 720 392" role="img" aria-label="Top: a 128-thread block is split into four warps of 32 threads each, warp 0 through warp 3. The block runs on a single SM, and an SM can host several blocks as long as its registers, shared memory and warp slots last. Bottom: warp divergence. For if x less than 0 then A else B, the same warp first runs path A with 16 lanes active and 16 masked, then path B with the other 16 active; in this equal-cost case half the lanes idle in each pass, but this is not a universal 50 percent loss." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="bw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="bw-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">A thread block is cut into warps of 32</text>
<rect x="16" y="38" width="430" height="176" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">thread block (128 threads)</text>
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
<text x="92" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">each square = one thread · 32 per warp</text>
<line x1="448" y1="126" x2="478" y2="126" marker-end="url(#bw-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="480" y="38" width="224" height="176" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="492" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">one SM</text>
<rect x="492" y="70" width="200" height="34" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="592" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">block 0</text>
<rect x="492" y="112" width="200" height="34" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="592" y="133" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">block 1</text>
<rect x="492" y="154" width="200" height="34" rx="5" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="592" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">+ more, while resources last</text>
<text x="592" y="204" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">registers · shared memory · warp slots</text>
<text x="16" y="252" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Warp divergence: one warp, two paths, taken in turn</text>
<text x="16" y="276" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">if (x &lt; 0) { A } else { B }</text>
<text x="16" y="307" text-anchor="start" style="fill:var(--c-text);font-size:12px">pass 1: if → A</text>
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
<text x="16" y="341" text-anchor="start" style="fill:var(--c-text);font-size:12px">pass 2: else → B</text>
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
<text x="176" y="362" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">16 lanes masked off in each pass → in this equal-cost case, half the lanes idle</text>
<text x="176" y="378" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(not a universal 50% loss: it depends on branch costs and scheduling)</text>
</svg>

**Analogy:** Imagine a team of 32 clerks executing the exact same formula across 32 distinct rows of a spreadsheet in unison. Same instruction, different data: this is the intuition behind SIMT.

Why is **warp divergence** so costly? If 16 threads within a warp evaluate an `if` branch as true while the remaining 16 take the `else` branch, the hardware cannot execute both paths concurrently for that warp. Instead, it must serialize execution: it executes the `if` path while masking (disabling) the second half, and then executes the `else` path while masking the first. In a simplified scenario where both branches have equal instruction counts, throughput drops significantly. However, stating that "every divergence causes an exact 50% performance drop" is inaccurate; branch length, instruction-level independence, and compiler scheduling dictate the actual penalty.

### 2.3. The Streaming Multiprocessor (SM): The GPU's Workstation

Where do these 32-thread warps actually execute? There is no unstructured, monolithic pool of execution units where every thread freely roams. Instead, the GPU partitions its compute fabric into modular, hardware-managed processing units called **Streaming Multiprocessors (SMs)**.

Think of an SM as an autonomous workstation on a factory floor. Inside each SM reside warp schedulers that dispatch instructions, arithmetic units (ALUs), Tensor Cores that accelerate matrix multiply-accumulate operations, and high-speed local storage (registers and shared memory) immediately adjacent to the execution pipelines. When we state that "an H100 GPU features 132 SMs," we mean it contains 132 of these physical workstations—not that each SM hosts a fixed pool of 132 threads.

The H100 SXM features **132 SMs**, with each SM partitioned into **4 processing blocks (subpartitions)**. In aggregate, this architecture delivers 528 fourth-generation Tensor Cores, 16,896 FP32 CUDA cores, and extensive auxiliary functional units. Note that these figures differ for the PCIe variant of the H100 (which features 114 SMs). [K2]

The familiar "989 TFLOP/s" comes straight from this layout. At the 1.83 GHz clock used for the peak figure, each Tensor Core performs 1,024 dense BF16 FLOPs per cycle (one 8 × 8 × 8 multiply-accumulate block):

$$
C_{\mathrm{peak}}=528\times1024\times1.83\times10^{9}\approx989\ \mathrm{TFLOP/s}
$$

Read it as: 528 Tensor Cores × 1,024 FLOPs per cycle × 1.83 billion cycles per second. The same arithmetic explains why no real kernel reaches the peak: every one of those 528 units has to be fed on every cycle. On Blackwell B200 the dense BF16 peak rises to 2,250 TFLOP/s. [K2][K16]

Inside an SM, four primary functional roles operate in concert:

| Unit | Primary Function | Analogy |
|---|---|---|
| **Warp Scheduler** | Issues instructions for warps that are eligible to execute | The dispatch manager |
| **CUDA Core / ALU** | Addition, multiplication, elementwise ops, and general arithmetic | General-purpose workbench |
| **Tensor Core** | Accelerates fused matrix multiply-accumulate (MMA) operations | Specialized mass-production press |
| **Register / Shared Memory** | Stores ultra-low-latency data during active computation | The immediate countertop surface |

<svg viewBox="0 0 720 374" role="img" aria-label="Inside one H100 SM there are four sub-partitions. Each has its own warp scheduler, a 64 KB register file, 32 FP32, 16 INT32 and 16 FP64 units, and one Tensor Core. All four share a 256 KB L1 data cache and shared memory, and the TMA unit issues asynchronous bulk copies to and from L2, which is 50 MB and shared by all SMs, and HBM. With 132 SMs, an H100 SXM has 528 Tensor Cores and 16,896 FP32 CUDA cores." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Inside one H100 SM: four sub-partitions + shared memory</text>
<rect x="16" y="38" width="688" height="300" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="28" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="108.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">sub-partition 0</text>
<rect x="36" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="108.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="36" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="108.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register file 64 KB</text>
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
<text x="108.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1 per sub-partition</text>
<rect x="196" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="276.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">sub-partition 1</text>
<rect x="204" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="276.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="204" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="276.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register file 64 KB</text>
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
<text x="276.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1 per sub-partition</text>
<rect x="364" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="444.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">sub-partition 2</text>
<rect x="372" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="444.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="372" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="444.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register file 64 KB</text>
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
<text x="444.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1 per sub-partition</text>
<rect x="532" y="50" width="160" height="196" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="612.0" y="68" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">sub-partition 3</text>
<rect x="540" y="78" width="144" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="612.0" y="94.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">warp scheduler</text>
<rect x="540" y="108" width="144" height="26" rx="4" style="fill:var(--c-border);fill-opacity:.10;stroke:var(--c-border);stroke-width:1.2"/>
<text x="612.0" y="125.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">register file 64 KB</text>
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
<text x="612.0" y="230" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1 per sub-partition</text>
<rect x="28" y="256" width="664" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="275.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">L1 data cache / shared memory · 256 KB per SM</text>
<rect x="28" y="292" width="328" height="30" rx="4" style="fill:var(--c-border);fill-opacity:.08;stroke:var(--c-border);stroke-width:1.2"/>
<text x="192.0" y="311.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">TMA (async bulk copies)</text>
<rect x="364" y="292" width="328" height="30" rx="4" style="fill:var(--c-border);fill-opacity:.08;stroke:var(--c-border);stroke-width:1.2"/>
<text x="528.0" y="311.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">→ L2 (50 MB, shared by all SMs) → HBM</text>
<text x="16" y="360" text-anchor="start" style="fill:var(--c-text);font-size:12px">× 132 SMs on H100 SXM = 528 Tensor Cores and 16,896 FP32 CUDA cores.</text>
</svg>

The role of the Tensor Core is paramount in modern deep learning. The linear projections and feed-forward networks of Transformer models are fundamentally composed of dense matrix multiplications. Dedicated hardware capable of executing fused multiply-add operations of the form $D = A \times B + C$ at staggering parallel density accounts for the vast majority of the H100's theoretical BF16 compute throughput.

**Technical nuance:** A "CUDA core" and a "Tensor Core" are distinct physical functional units. Just because `nvidia-smi` reports 99% GPU utilization does not mean Tensor Cores are operating efficiently—or being utilized at all.

### 2.4. How the Warp Scheduler Hides Memory Latency

This brings us to a fundamental question: when a warp issues a read request to HBM for data that is hundreds of clock cycles away, do all 32 threads simply sit idle, leaving the SM stalled?

Not necessarily. This is where the genius of hardware-managed multithreading and the warp scheduler comes into play. If another warp on the same SM has all its operands ready, the scheduler immediately switches context and dispatches instructions for that warp. Zero-overhead context switching allows useful computation to fill the idle pipeline bubbles of stalled warps. We do not "eliminate" physical memory latency; rather, we **hide latency behind concurrent computation**.

Consider a warp requesting operands from high-bandwidth memory. Until those bytes arrive at the register file, dependent instructions cannot advance. As long as the SM hosts other active, eligible warps, the scheduler effortlessly rotates to them.

<svg viewBox="0 0 720 252" role="img" aria-label="Timeline of 8 time slots for three warps. Warp A computes in slots 1–2 and 7–8 and waits in 3–6. Warp B computes in slots 3–4 and waits otherwise. Warp C computes in slots 5–6 and waits otherwise. The SM&#x27;s issue row reads A A B B C C A A, so the SM is busy in all 8 slots although no warp is busy more than half the time. The memory wait is not shorter; it is hidden behind other ready warps." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ws-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Latency hiding: while one warp waits, another one computes</text>
<text x="183.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">1</text>
<text x="249.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">2</text>
<text x="315.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">3</text>
<text x="381.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">4</text>
<text x="447.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">5</text>
<text x="513.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">6</text>
<text x="579.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">7</text>
<text x="645.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">8</text>
<text x="138" y="46" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">time →</text>
<text x="16" y="73" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Warp A</text>
<rect x="152" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="183.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="218" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="249.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="284" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="315.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="350" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="381.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="416" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="447.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="482" y="56" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="513.0" y="72.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="548" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="579.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="614" y="56" width="62" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="645.0" y="72.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<text x="16" y="107" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Warp B</text>
<rect x="152" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="183.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="218" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="249.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="284" y="90" width="62" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="315.0" y="106.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="350" y="90" width="62" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="381.0" y="106.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="416" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="447.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="482" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="513.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="548" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="579.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="614" y="90" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="645.0" y="106.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<text x="16" y="141" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Warp C</text>
<rect x="152" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="183.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="218" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="249.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="284" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="315.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="350" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="381.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="416" y="124" width="62" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="447.0" y="140.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="482" y="124" width="62" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="513.0" y="140.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="548" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="579.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<rect x="614" y="124" width="62" height="26" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="645.0" y="140.96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">wait</text>
<line x1="16" y1="162" x2="704" y2="162" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="16" y="189" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">SM issues</text>
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
<text x="16" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">No single warp is busy more than half the time, yet the SM issues work in all 8 slots.</text>
<text x="16" y="240" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The memory wait did not get shorter — it was covered by other ready warps.</text>
</svg>

This mechanism is known as **latency hiding**. Crucially: the physical latency of the memory subsystem does not magically decrease. Rather, because other useful work is overlapped during that waiting window, the **aggregate wall-clock execution time** is shielded from the penalty.

A simplified analytical intuition:

$$
\text{Required independent work}\propto
\frac{\text{Memory latency (stall cycles)}}{\text{Compute time between independent instructions}}
$$

This is not a hard hardware specification or an exact warp-scheduling equation; rather, it formalizes why achieving high concurrency is essential for saturating GPU throughput.

We can put measured numbers on this intuition. Luo et al. (2024) micro-benchmarked load latencies on Hopper: about **29** cycles for shared memory, **41** for L1, **263** for L2 and **479** for HBM [K52]. Now suppose a warp computes for 2 cycles and then waits 6. The number of warps needed to keep the scheduler busy every cycle is roughly:

$$
N_{\mathrm{warp}}\approx1+\frac{T_{\mathrm{wait}}}{T_{\mathrm{compute}}}=1+\frac{6}{2}=4
$$

Read it as: while one warp computes, enough other warps must be ready to fill its waiting time. With two such warps the scheduler idles half the time; with four it never does. A ~479-cycle HBM load needs far more independent work, which is why fast Hopper kernels such as FlashAttention-3 do not rely on warp count alone: they hide data movement behind math with TMA copies and asynchronous WGMMA. [K8]

### 2.5. Occupancy and Register Pressure: More Threads Do Not Always Mean Faster Execution

At this point, an intuitive idea arises: "Why not simply launch vastly more warps so the GPU never runs dry?" It sounds reasonable, but every active warp consumes dedicated hardware state. The physical registers and shared memory allocated to each thread are strictly bounded.

As the number of registers allocated per thread increases, the total number of concurrent warps that can simultaneously reside on an SM decreases. In our kitchen analogy: if every chef demands a massive countertop, fewer chefs can fit onto the prep floor at the same time.

**Occupancy** is defined as the ratio of active, resident warps on an SM to the maximum theoretical warp capacity permitted by the hardware architecture. **Register pressure** describes the degree to which a kernel's per-thread register demands constrain achievable occupancy.

An SM features a finite physical register file. If each thread consumes a high register count, fewer concurrent warps can be maintained in flight. For instance, assuming an H100 provides 65,536 32-bit registers per SM:

$$
R_{\text{warp}}=32\times R_{\text{thread}}
$$

<svg viewBox="0 0 720 288" role="img" aria-label="The same 65,536-register file of one SM is split among warps for four kernels. At 32 registers per thread at most 64 warps fit (100 percent), limited also by the SM warp ceiling and block geometry. At 64 registers, 32 warps (50 percent), where shared memory allocation also matters. At 128 registers, 16 warps (25 percent), limited also by shared memory and block granularity. At 255 registers, 8 warps (12 percent), with register allocation granularity, spilling and resource limits. These are upper bounds from the register budget only, not real occupancy." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="oc-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">One register file, four kernels: bigger threads, fewer warps</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">65,536 × 32-bit registers per SM · R_warp = 32 × R_thread · upper bound from registers only</text>
<text x="16" y="78" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">32 regs/thread</text>
<text x="16" y="95" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SM warp ceiling (64), block geometry</text>
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
<text x="660" y="82" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">64 warps</text>
<text x="660" y="97" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">100%</text>
<text x="16" y="130" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">64 regs/thread</text>
<text x="16" y="147" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">same, plus shared memory allocation</text>
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
<text x="660" y="134" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">32 warps</text>
<text x="660" y="149" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">50%</text>
<text x="16" y="182" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">128 regs/thread</text>
<text x="16" y="199" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">shared memory, block granularity</text>
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
<text x="660" y="186" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">16 warps</text>
<text x="660" y="201" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">25%</text>
<text x="16" y="234" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">255 regs/thread</text>
<text x="16" y="251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">allocation granularity, spilling, limits</text>
<rect x="236" y="220" width="416" height="26" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="236.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="288.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="340.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="444.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="496.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="548.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="600.6" y="222" width="50.8" height="22" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="660" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">8 warps</text>
<text x="660" y="253" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">12%</text>
<text x="16" y="276" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Each block in a bar = one resident warp. Real occupancy can be lower; higher occupancy is not automatically faster.</text>
</svg>

The warp counts shown above represent **theoretical upper bounds derived strictly from the register budget**; they do not represent final achievable occupancy.

Does maximizing performance always require 100% occupancy? No. Many compute-heavy kernels achieve near-peak Tensor Core utilization with surprisingly moderate occupancy. In modern architectures like Hopper, advanced techniques such as asynchronous data movement (TMA) and warp specialization—exemplified by FlashAttention-3—can keep execution units fully saturated even at low occupancy by decoupling data movement from computation. [K8]

**Sanity check:** Attempting to optimize a kernel merely by launching "more threads" is dangerous. Without profiling register spills, shared memory usage, warp divergence, and memory access patterns first, blindly increasing thread concurrency often degrades overall performance.

---


## 3. The Memory Hierarchy: Why Data Movement Matters as Much as Math

We have established that a GPU houses a massive array of arithmetic processing units. However, every single execution unit requires operands to perform computation. Where does that data actually live?

While it is theoretically possible to store everything in HBM and read operands from it before every single operation, doing so would be disastrously inefficient. Even though HBM delivers extraordinary bandwidth, its latency and access energy are orders of magnitude higher than reading directly from registers or shared memory. Consequently, modern GPUs incorporate a steep, multi-tiered hierarchy of memory structures optimized for varying capacities, latencies, and costs.

Think of this as the continuum from the chef's knife in hand, to the immediate cutting board, to the kitchen shelf, and finally down to the central warehouse. The cutting board is instantly accessible but holds very little; the central warehouse holds vast inventory, but walking there for every pinch of salt halts the entire kitchen.

Modern GPUs organize their storage into a rigorous hierarchy balancing **latency, bandwidth, capacity, and energy cost per byte**:

<svg viewBox="0 0 720 300" role="img" aria-label="Memory ladder drawn as a widening pyramid, from top to bottom: registers, 256 KB per SM, private to one thread; shared memory and L1, 256 KB per SM with measured load latencies of about 29 and 41 cycles, shared by one thread block; L2, 50 MB at about 263 cycles, shared by all 132 SMs; HBM3, 80 GB at 3.35 TB/s and about 479 cycles, holding weights, activations and the KV cache; NVLink, PCIe and NIC at 450, 63 and 50 GB/s per direction, reaching data on other GPUs and hosts; and CPU RAM and storage, 2 TB per DGX and petabyte-scale, for datasets and checkpoints. Each step down is bigger, farther and costlier to reach." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mh-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">The memory ladder: every step down is bigger, farther and costlier to reach</text>
<rect x="210" y="40" width="120" height="36" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="270" y="62" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Registers</text>
<text x="490" y="55" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">256 KB per SM</text>
<text x="490" y="70" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">private to one thread</text>
<line x1="334" y1="58" x2="482" y2="58" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="182" y="82" width="176" height="36" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="270" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Shared memory / L1</text>
<text x="490" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">256 KB per SM · ~29 / ~41 cycles</text>
<text x="490" y="112" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">shared by one thread block</text>
<line x1="362" y1="100" x2="482" y2="100" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="154" y="124" width="232" height="36" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="270" y="146" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">L2</text>
<text x="490" y="139" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">50 MB · ~263 cycles</text>
<text x="490" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">shared by all 132 SMs</text>
<line x1="390" y1="142" x2="482" y2="142" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="126" y="166" width="288" height="36" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="270" y="188" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">HBM3</text>
<text x="490" y="181" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">80 GB · 3.35 TB/s · ~479 cycles</text>
<text x="490" y="196" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">weights, activations, KV cache</text>
<line x1="418" y1="184" x2="482" y2="184" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="98" y="208" width="344" height="36" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="270" y="230" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">NVLink / PCIe / NIC</text>
<text x="490" y="223" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">450 · 63 · 50 GB/s per direction</text>
<text x="490" y="238" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">data on other GPUs and hosts</text>
<line x1="446" y1="226" x2="482" y2="226" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="70" y="250" width="400" height="36" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="270" y="272" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU RAM / storage</text>
<text x="490" y="265" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">2 TB per DGX · PB-scale</text>
<text x="490" y="280" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">datasets, checkpoints</text>
<line x1="474" y1="268" x2="482" y2="268" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<path d="M20 44 V282" marker-start="url(#mh-arr)" marker-end="url(#mh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="28" y="54" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">closer</text>
<text x="28" y="278" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">farther</text>
</svg>

**Registers** provide ultra-fast, single-cycle storage for thread-local private variables. **Shared memory** is an on-chip, software-managed SRAM bank shared among threads within the same thread block. The **L1/L2 caches** automatically retain frequently accessed cache lines to minimize round-trips to DRAM. Finally, **High Bandwidth Memory (HBM)** provides the large-capacity on-package storage holding model weights, optimizer states, and KV caches.

For the H100 SXM, the baseline architecture features 80 GB of HBM3 delivering approximately **3.35 TB/s** of peak memory bandwidth alongside a 50 MB on-chip L2 cache. One must be careful not to conflate decimal `GB` with binary `GiB`, nor theoretical peak bandwidth with sustainable, kernel-achievable memory throughput. [K2]

### 3.1. Bandwidth and Latency Are Not the Same Thing

We must distinguish between two fundamentally different questions: "How many boxes can a freight truck haul per trip?" versus "How many minutes does it take the truck to complete a round-trip to the warehouse?" The former is throughput/capacity; the latter is latency. This distinction is vital when reasoning about memory and interconnects.

**Bandwidth:** The volume of data that can be transferred per unit of time (measured in GB/s or TB/s).

**Latency:** The elapsed time between issuing a memory request and receiving the first valid byte of data (measured in nanoseconds or clock cycles).

An eight-lane highway represents high bandwidth; if that highway is 500 kilometers long, it has high latency. Both parameters govern performance. When streaming large contiguous tensors, bandwidth dominates; when evaluating pointer-chasing, irregular lookups, or fine-grained dependencies, latency dictates throughput.

**Memory coalescing** refers to aligning and organizing the memory requests of adjacent threads within a warp such that the hardware memory controller can service them via a minimal number of wide 32-byte or 128-byte transactions. Scattered or unaligned memory access patterns lead to severe bandwidth inflation, wasting valuable memory bus cycles on unused bytes.

### 3.2. The Roofline Model: The Mathematical Boundary Between Compute and Bandwidth

Suppose we know the peak compute capability of an H100, and we also know the peak transfer speed of its HBM3 memory. Can we analytically determine whether a specific algorithm will bottleneck on memory or compute?

Yes. The most powerful conceptual framework for answering this is the **Roofline Model**. While its formulation is elegant, the fundamental question it asks is simple: *"For every single byte transferred from memory, do we perform enough arithmetic operations to keep the compute units fully occupied?"*

The **arithmetic intensity** ($\alpha$) of a kernel is defined as:

$$
\alpha=\frac{\text{Total floating-point operations (FLOP)}}{\text{Total bytes transferred from memory}}
$$

The achievable execution performance ($P_{\mathrm{achievable}}$) is bounded by the simplified Roofline equation:

$$
P_{\mathrm{achievable}}\leq\min(C_{\mathrm{peak}},\alpha W_{\mathrm{mem}})
$$

Where $C_{\mathrm{peak}}$ is the theoretical peak compute capacity for the specified data type, $W_{\mathrm{mem}}$ is the bandwidth of the targeted memory tier (typically HBM), and $P_{\mathrm{achievable}}$ is the maximum sustainable performance.

- When $\alpha W_{\mathrm{mem}} < C_{\mathrm{peak}}$, the workload is strictly **memory-bound** (limited by data transfer).
- When $\alpha W_{\mathrm{mem}} \ge C_{\mathrm{peak}}$, the workload enters the **compute-bound** regime, making it possible to hit the hardware's compute ceiling.

For an H100 SXM operating on dense (non-sparse) **BF16 Tensor Core** arithmetic:

$$
C_{\mathrm{peak}}\simeq990\ \mathrm{TFLOP/s},\quad
W_{\mathrm{HBM}}\simeq3.35\ \mathrm{TB/s}
$$

$$
\boxed{\alpha_{\mathrm{crit}}=\frac{990}{3.35}\simeq295.5\ \mathrm{FLOP/byte}}
$$

This threshold is not an arbitrary rule saying "the H100 must always perform 295 FLOPs per byte." Rather, it marks the exact inflection point—the **machine balance**—between the memory-bandwidth slope and the compute roofline for dense BF16 math against HBM. For standard FP32 CUDA cores or FP8 Tensor Cores, this critical inflection shifts dramatically.

### 3.3. Why Batch Size Changes Everything

Having examined the Roofline formulation analytically, let us apply it to a concrete, real-world scenario. Imagine you have a massive weight matrix loaded in memory, and you must evaluate it to generate the next token. If you fetch that entire multi-gigabyte matrix from HBM just to multiply it against a single token vector, you perform very little computation relative to the enormous volume of data moved across the bus.

However, if you can reuse that identical weight matrix across *a batch of multiple tokens*, you pay the memory transfer cost once while performing multiplied FLOPs across all inputs. This single dynamic explains why batch size fundamentally alters the performance character of deep learning workloads.

Consider a standard linear projection:

$$
X\in\mathbb{R}^{B\times D},\quad
W\in\mathbb{R}^{D\times F},\quad
Y=XW
$$

Assuming 16-bit precision (2 bytes per element for BF16/FP16) and an *idealized baseline* where inputs and weights are read once from HBM and outputs are written once:

$$
\text{FLOP}\simeq2BDF
$$

$$
\text{Bytes}\simeq2(BD+DF+BF)
$$

$$
\boxed{\alpha(B)\simeq\frac{BDF}{BD+DF+BF}}
$$

**Definitions:** $B$ is the number of tokens processed concurrently (batch dimension); $D$ is the model's hidden dimension; $F$ is the projection output or FFN intermediate dimension.

When the weight matrix is vast and the batch size $B$ is small ($B \ll D, F$), the $DF$ weight term completely dominates the denominator, yielding $\alpha(B) \approx B$. In this regime, arithmetic intensity scales almost linearly with batch size because you stream the weights from HBM once and reuse them across only $B$ inputs.

Using the MLP dimensions of Llama 3 70B as an illustrative example ($D=8192, F=28672$), evaluating the formula yields:

| Concurrent Tokens ($B$) | Approx. Arithmetic Intensity (FLOP/byte) | H100 Roofline Implication |
|---:|---:|---|
| 1 | 1.0 | Severely memory-bound (decode-like) |
| 8 | 8.0 | Memory-bound |
| 32 | 31.8 | Memory-bound |
| 128 | 125.5 | HBM bandwidth ceiling still dominates |
| 310 | 295.6 | Near the theoretical machine balance |
| 512 | 473.9 | Compute-bound regime reachable |

<svg viewBox="0 0 720 364" role="img" aria-label="Log-log roofline for H100 SXM with BF16 dense Tensor Cores and HBM. The memory roof rises as 3.35 TB/s times arithmetic intensity until it meets the 990 TFLOP/s compute roof at about 295.5 FLOP per byte. Points for a Llama 3 70B MLP layer: batch 1 at intensity 1.0, batch 8 at 8.0, batch 32 at 31.8 and batch 128 at 125.5 all sit on the memory roof; batch 310 at 295.6 reaches the ridge, and batch 512 at 473.9 sits under the compute roof. Idealized data-movement model, not measured kernels." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rf-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">H100 SXM roofline (BF16 dense, HBM) and the batch sizes from the table</text>
<rect x="84" y="50" width="466.3" height="250" style="fill:var(--c-warn);fill-opacity:.06"/>
<rect x="550.3" y="50" width="139.7" height="250" style="fill:var(--c-success);fill-opacity:.06"/>
<line x1="84" y1="300" x2="690" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="304.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="84" y1="224.266" x2="690" y2="224.266" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="228.26606231134411" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<line x1="84" y1="148.532" x2="690" y2="148.532" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="152.5321246226882" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<line x1="84" y1="72.7982" x2="690" y2="72.7982" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="78" y="76.79818693403232" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1,000</text>
<line x1="134.644" y1="50" x2="134.644" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="134.64440287267917" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="302.881" y1="50" x2="302.881" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="302.8814676242264" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<line x1="471.119" y1="50" x2="471.119" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="471.11853237577355" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<line x1="639.356" y1="50" x2="639.356" y2="300" style="stroke:var(--c-border);stroke-width:0.8"/>
<text x="639.3555971273208" y="316" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1,000</text>
<line x1="84" y1="300" x2="690" y2="300" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="84" y1="50" x2="84" y2="300" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="387.0" y="334" text-anchor="middle" style="fill:var(--c-text);font-size:12px">arithmetic intensity α (FLOP/byte, log)</text>
<text x="22" y="175.0" text-anchor="middle" style="fill:var(--c-text);font-size:12px" transform="rotate(-90 22 175.0)">TFLOP/s (log)</text>
<path d="M84.0 283.0 L550.3 73.1 L690 73.1" style="fill:none;stroke:var(--c-accent);stroke-width:2.4"/>
<text x="214.91388229729048" y="214.10201792524697" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600" transform="rotate(-24.5 215 214)">memory roof: 3.35 TB/s × α</text>
<line x1="84" y1="73.1288" x2="550.289" y2="73.1288" style="stroke:var(--c-accent);stroke-width:1;stroke-dasharray:4 3;stroke-opacity:.5"/>
<text x="538.2892778792796" y="66.1287508344046" text-anchor="end" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">compute roof: 990 TFLOP/s</text>
<line x1="550.289" y1="73.1288" x2="550.289" y2="300" style="stroke:var(--c-accent);stroke-width:1;stroke-dasharray:4 3"/>
<text x="556.2892778792796" y="292" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">α* ≈ 295.5</text>
<text x="97.32122041842246" y="66" text-anchor="start" style="fill:var(--c-warn);font-size:12px;font-weight:600">memory-bound</text>
<text x="620.1446389396398" y="240" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:600">compute-bound</text>
<text x="620.1446389396398" y="256" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:600">possible</text>
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
<text x="16" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Points: α(B) = B·D·F / (B·D + D·F + B·F) for D = 8192, F = 28672 (Llama 3 70B MLP); idealized data-movement model, not measured kernels.</text>
</svg>

Keep in mind that these numbers represent an idealized analytical data-movement model rather than **empirically profiled kernel execution rates**. In practice, tiling strategies, L2 cache residency, register reuse, activation layouts, and quantization alter the exact inflection points.

### 3.4. Tensor Core Utilization, Achieved Occupancy, and MFU Are Fundamentally Different

GPU Utilization, SM activity, achieved occupancy, Tensor Core utilization, DRAM throughput, and Model FLOPs Utilization (MFU) cannot be used interchangeably. For example, an SM can report 100% activity while producing minimal FLOPs due to poor memory coalescing and persistent cache stalls. Conversely, another workload might execute highly optimized Tensor Core kernels at 80% arithmetic efficiency, yet spend half of every training step stalled waiting on all-reduce network collectives. The former represents a kernel-level optimization failure; the latter is a distributed system-level communication bottleneck.

### 3.5. Across generations: why the roofline knee keeps moving

The knee $\alpha_{\mathrm{crit}}=C_{\mathrm{peak}}/W_{\mathrm{HBM}}$ is not a constant of nature. In every GPU generation compute grows faster than memory bandwidth, so the knee drifts upward. The table collects vendor dense (non-sparse) peaks:

| Generation | Tensor Core | HBM capacity · bandwidth | BF16/FP16 TFLOP/s | FP8 TFLOP/s | FP4 TFLOP/s | $\alpha^{*}=C/W$ (FLOP/byte) |
|---|---|---|---:|---:|---:|---:|
| V100 SXM2 | 1st gen | 32 GB · 0.9 TB/s | 125 (FP16) | — | — | ≈139 |
| A100 SXM | 3rd gen | 80 GB · 2.0 TB/s | 312 | — (INT8: 624 TOPS) | — | ≈156 |
| H100 SXM | 4th gen | 80 GB · 3.35 TB/s | 989 | 1,979 | — | ≈295 |
| H200 SXM | 4th gen | 141 GB · 4.8 TB/s | 989 | 1,979 | — | ≈206 |
| B200 (DGX B200) | 5th gen | 180 GB · 8 TB/s | 2,250 | 4,500 | 9,000 | ≈281 |

Two lessons follow. First, the H200 adds 43% more memory bandwidth to the same compute and pulls the knee down from 295 to 206 FLOP/byte: for memory-bound decode it is a faster chip than the H100 even though its TFLOP/s are identical. Second, the B200 grows both sides, but the ratio stays near 280 FLOP/byte: below roughly 300 tokens per matrix multiply, Tensor Cores still starve. Blackwell also adds a separate 256 KB Tensor Memory (TMEM) per SM to hold Tensor Core accumulators. [K16][K56]

---



---

### Checkpoint: Three Distinct Concepts of "Parallelism" Inside a GPU

Let us clearly delineate three concepts frequently conflated in GPU performance analysis:

1. **Concurrency:** Managing multiple tasks in progress within the same execution window.
2. **Parallel Execution:** Physically executing instructions across distinct hardware engines (e.g., across multiple SMs or Tensor Cores) at the exact same clock cycle.
3. **Latency Hiding:** Interleaving ready warps while stalled warps wait for memory transactions to clear.

High achieved occupancy merely indicates that the warp scheduler has a rich pool of candidates to dispatch; it does not automatically imply that Tensor Cores are generating useful arithmetic FLOPs. Without addressing register pressure and instruction dependencies, artificially inflating occupancy yields negligible performance gains.

**Thought experiment:** When a CUDA kernel configured with 256-thread blocks can only fit a single block per SM, why might shrinking block size to 128 threads sometimes yield a higher number of active warps? Because resource allocations per block (registers and shared memory) and hard caps per SM interact non-linearly. Yet this change does not guarantee faster wall-clock execution: the modified configuration may introduce higher scheduling contention or reduce data reuse efficiency across thread blocks.


### Why Memory "Proximity" Is Not Just Physical Distance

HBM sits directly on the same substrate package as the GPU silicon; thus, calling it "local GPU memory" is geographically accurate. Yet for an active warp, accessing an operand in a register versus fetching it from HBM represents entirely different universes of latency and energy. *Proximity* in computer architecture is not merely a question of millimeters on a die: it encompasses protocol overhead, arbitration, bank conflicts, cache line hits, transfer widths, and the degree to which data movement can be hidden behind concurrent arithmetic.

This is why **latency** and **bandwidth** must never be conflated. Stalling on a single scalar dependency is dictated by latency; streaming billions of contiguous bytes across a wide bus is governed by bandwidth. A clear, empty mountain pass allows a single sports car to travel at high speed, but it cannot move the freight volume of an eight-lane highway.

### Applying Roofline Analysis to Concrete Workloads

For the H100 SXM, combining a dense BF16 Tensor Core ceiling of approximately 990 TFLOP/s with an HBM3 bandwidth of 3.35 TB/s produces a critical inflection point of roughly 296 FLOP/byte. This ratio is not an immutable natural law; it is the theoretical machine balance for that specific data type and hardware path.

Now let us examine three distinct operational patterns:

| Workload Characteristic | Weight / Data Reuse | Expected Bottleneck Profile |
|---|---|---|
| Large matrix-vector multiply (GEMV, $B=1$) | Near zero | Strictly memory-bandwidth bound (HBM dominated) |
| Batched matrix-matrix multiply (GEMM, large $B$) | High | Compute-bound (Tensor Core saturated) |
| Chained, elementwise / pointwise kernels | Low / fragmented | Launch overhead and redundant HBM round-trips dominate |

This distinction underscores why **algorithmic and kernel design** matters just as much as hardware selection. Faster HBM helps, but the most profound performance leaps come from structuring algorithms so that data fetched into on-chip SRAM is reused extensively across multiple operations before being written back. This is the exact design philosophy powering innovations like kernel fusion and FlashAttention.




---

**PART II — The Model's Workload: Training, Prefill, Decode, and the KV Cache**

*Why does the exact same Transformer hit fundamentally different performance walls across different operational phases? We will trace the execution path from prompt submission to token generation.*

## 4. The Transformer Workload: Why Training, Prefill, and Decode Utilize the Same GPU Differently

So far, we have examined the GPU architecture and memory hierarchy. But what exactly is the GPU computing? Without answering this question, statements like "LLM inference is slow" or "training is compute-bound" remain abstract assertions.

Do not think of a Transformer as a single, monolithic, incomprehensible mathematical object. Inside, a handful of core operations repeat continuously: mapping tokens into vectors, extracting cross-token contextual information from these vectors, multiplying by massive weight matrices, and passing the result to the subsequent layer. As the way these operations are packaged changes, the GPU's execution bottleneck shifts with it.

### 4.1. Visualizing What a Transformer Actually Computes

At a high level, a standard Transformer block executes the following computations:

<svg viewBox="0 0 720 428" role="img" aria-label="Flow of one Transformer block, top to bottom: token IDs; embedding lookup into activations X of size N by d; Q, K and V linear projections, which are GEMMs; attention, softmax of Q times K transposed over root d, times V, whose cost grows with sequence length; output projection plus residual and normalization; and the MLP, W2 times activation of W1 times X, another GEMM. The projections through the MLP repeat L times, 80 in a 70B model, and the final norm produces logits and next-token probabilities." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tb-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">What one Transformer block computes</text>
<rect x="150" y="40" width="360" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="162" y="57" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Token IDs</text>
<text x="162" y="73" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[t1, t2, …, tN]</text>
<line x1="330.0" y1="80" x2="330.0" y2="92" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="92" width="360" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="162" y="109" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Embedding</text>
<text x="162" y="125" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">lookup → X  (N × d)</text>
<rect x="524" y="103" width="86.536" height="18" rx="9" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="567.268" y="115.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">memory read</text>
<line x1="330.0" y1="132" x2="330.0" y2="144" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="144" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="161" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Q, K, V projections</text>
<text x="162" y="177" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Q = XW_Q,  K = XW_K,  V = XW_V</text>
<rect x="524" y="155" width="39.104" height="18" rx="9" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="543.552" y="167.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GEMM</text>
<line x1="330.0" y1="184" x2="330.0" y2="196" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="196" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="162" y="213" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Attention</text>
<text x="162" y="229" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">softmax(Q·Kᵀ / √d) · V</text>
<rect x="524" y="207" width="140.744" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594.3720000000001" y="219.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">grows with sequence</text>
<line x1="330.0" y1="236" x2="330.0" y2="248" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="248" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="265" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Output proj. + residual + norm</text>
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
<text x="118" y="238.0" text-anchor="end" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">× L blocks</text>
<text x="118" y="254.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">(80 in 70B)</text>
<path d="M330.0 340 V358" marker-end="url(#tb-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="150" y="360" width="360" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="330.0" y="382.0" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Final norm → logits → next-token probabilities</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Teal = large matrix multiplies (Tensor Cores). Norm, softmax and data movement still take real time.</text>
</svg>

**Embedding:** Maps discrete token IDs into continuous numerical vectors. **Activation:** Intermediate tensors produced by a layer. **Attention:** Enables different token positions to exchange contextual information with one another. **MLP:** Transforms each token's representation via learned projection matrices. **Logits:** Raw unnormalized scores produced before computing next-token probabilities.

The bulk of the model's compute budget is dominated by matrix multiplications; however, layer normalizations, softmax operations, data movement, and synchronization overhead also substantially impact the total wall-clock time.

### 4.2. Training: Large GEMMs and the Backward Pass

During training, the goal is not merely to generate text continuations, but to update the model's weights. This requires a bidirectional execution pipeline. First, the model generates predictions using its current weights; then, by evaluating prediction error, it calculates the direction and magnitude in which the parameters must be adjusted.

Consider an instructor grading a student's derivation who does not merely say "wrong," but explains exactly which step was incorrect and how it should be corrected. In deep learning training, the *loss function* and *backward pass* are the rigorous mathematical equivalents of this feedback loop.

During training, vast batches of tokens are processed simultaneously. The forward pass computes intermediate activations; a scalar loss is evaluated; backpropagation (the backward pass) computes gradients with respect to activations and weights; and the optimizer updates the model weights.

Multiplying matrices with large batch and sequence dimensions yields a classic **GEMM (General Matrix–Matrix Multiplication)** workload. Weights fetched in a single memory transfer are reused across thousands of tokens concurrently. As a result, dense training workloads typically operate deep within the compute-bound regime.

### 4.3. Prefill: Batch Processing the Prompt

Imagine sending a 2,000-token prompt to a chat interface. Before emitting a single response token, the model must process the entire context of those 2,000 prompt tokens. The phase that performs this initial ingestion is known as prefill.

The advantage of prefill is that many tokens are ingested concurrently. This enables large matrix multiplications and high arithmetic reuse of weights. However, one must not overlook the distinct quadratic scaling and memory overhead that attention incurs over long sequence lengths.

When a 2,000-token prompt hits an LLM serving system, the engine first processes the prompt tokens in parallel to generate the attention states and populate the **KV cache**. This stage is termed **prefill** (or context encoding).

Because prefill operates on larger activation matrices, it exhibits a high theoretical arithmetic intensity. Nonetheless, depending on sequence length, attention dimension, kernel fusion efficiency, and architectural details, one cannot unconditionally assume that every prefill operation is purely compute-bound.

### 4.4. Decode: Autoregressive Token-by-Token Generation

Once the initial token is generated, the workload fundamentally transforms. Autoregressive language models generate output sequentially from left to right, one token at a time: the model inspects the existing context, emits the next token, appends it to the context, and repeats the forward pass.

Ingesting a 2,000-token prompt en masse and streaming subsequent tokens one by one exhibit completely different computational patterns. This is precisely why serving performance cannot be reduced to a single aggregate "tokens/s" metric.

During subsequent token generation, the model retrieves the key/value tensors of prior token positions from the KV cache while propagating only the single newest token through the layers. When serving small batch sizes, the entire multi-gigabyte weight matrices must be transferred from HBM to compute cores for each individual token step.

For a **$B=1$ dense BF16 model**, the naive weight traffic derivation yields an arithmetic intensity of $\alpha\approx1\ \mathrm{FLOP/byte}$. Under low concurrency, where weights reside in HBM and few requests are batched together, token decoding is aggressively memory-bandwidth bound.

However, this does not mean that "decode is always memory-bound" or that "MFU is permanently stuck at 1%." High request concurrency, continuous batching, large context attention overhead, and cache access dynamics can shift the operational regime toward compute or latency bounds.

### 4.5. What Does the KV Cache Actually Store?

With every new token generated, recomputing all intermediate activations from the start of the conversation would be prohibitively slow. Imagine sitting in a meeting where, before speaking each new sentence, participants had to replay the entire meeting from the opening word. Keeping concise reference notes allows instant recall of prior context without redundant processing.

The KV cache acts as this computational reference notebook. Crucially, it does not store the raw text tokens themselves, but rather the projected **key** and **value** tensors required by the attention mechanism across every preceding layer and position.

The **KV cache** stores the key and value projections of all preceding token positions, avoiding the need to re-project past tokens through the input and attention weights on every decode step.

Simplified BF16 memory footprint:

$$
S_{\mathrm{KV}}\approx
2\times L\times T\times H_{\mathrm{KV}}\times d_{\mathrm{head}}\times b
$$

Where $L$ is the number of layers, $T$ is the number of stored context tokens, $H_{\mathrm{KV}}$ is the number of KV heads, $d_{\mathrm{head}}$ is the dimension per head, and $b$ is the number of bytes per scalar element (e.g., 2 for BF16/FP16). The leading factor of 2 accounts for the two distinct Key and Value tensors.

Multiplying this footprint by the number of concurrent active requests reveals why GPU memory is rapidly exhausted under production serving loads. **GQA (Grouped-Query Attention)** mitigates this by having multiple query heads share a single KV head, slashing the per-token cache footprint by an order of magnitude.

### 4.6. Quantization: Accelerating by Moving Fewer Bytes

When hitting the memory wall, one effective countermeasure is reducing the volume of data moved across the memory bus. If every model parameter can be represented using 1 byte instead of 2 bytes, the memory footprint and the bandwidth required to stream the weights are cut in half.

This technique is known as **quantization**. However, a critical caveat applies: representing numbers in fewer bits does not automatically guarantee identical inference latency, nor does it guarantee that the model will retain its original predictive accuracy.

BF16 weights require 2 bytes/parameter, INT8 weights take approximately 1 byte/parameter, and INT4 weights consume roughly 0.5 bytes/parameter. Excluding scale and zero-point metadata, a baseline comparison yields:

<svg viewBox="0 0 720 311" role="img" aria-label="Horizontal bars of raw weight memory. 8B model: 16 GB in BF16, 8 GB in INT8, 4 GB in INT4. 70B: 140, 70 and 35 GB. 405B: 810, 405 and 202.5 GB. Dashed reference lines mark one H100 at 80 GB and an 8-GPU DGX at 640 GB: a 70B model only fits on one H100 in INT4, and a 405B model does not fit on one DGX in BF16. Scales, zero points, KV cache and workspace are excluded." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="qb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Weight memory by precision vs. GPU memory</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Raw weights only (no scales, zero points, KV cache or workspace)</text>
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
<text x="277.5" y="274" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">202.5 GB</text>
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

**Why is an INT4 model not automatically faster?** Runtime on-the-fly dequantization overhead, specialized kernel support, native Tensor Core precision compatibility, unquantized KV cache overhead, and memory alignment patterns heavily dictate real-world speedups. Furthermore, low-bit quantization can induce significant degradation in model quality; downstream task accuracy and perplexity benchmarks must always be rigorously validated.

**Engineering Principle:** For training, track and report MFU and step time; for online serving, evaluate TTFT (*time to first token*), TPOT (*time per output token*), aggregate throughput, KV cache capacity limits, and p95/p99 tail latencies jointly.

### 4.7. FlashAttention, TMA, and WGMMA: Smarter Data Movement for Identical Math

Now consider reducing memory pressure not merely by shrinking numerical bit-widths, but by *rearchitecting the algorithm's data movement patterns*.

When cooking a meal, trekking down to the basement storage pantry for every single pinch of spice cripples your throughput; staging essential ingredients directly onto your kitchen counter dramatically speeds up preparation. Similarly, on a GPU, tiling large matrices into small blocks that fit into fast on-chip shared memory and registers minimizes expensive round-trips to off-chip HBM. FlashAttention and Hopper's asynchronous architectural primitives bring this intuition to hardware-level realization.

**FlashAttention** avoids materializing the massive $N \times N$ intermediate attention score matrix in HBM, computing attention block-by-block while keeping intermediate sums in fast SRAM/shared memory via online softmax. This tiling technique does not alter the exact mathematical output of attention; rather, it drastically reduces unnecessary HBM read/write traffic. [K8]

On the Hopper architecture specifically:

- **TMA (Tensor Memory Accelerator):** Dedicated hardware engine that initiates and manages asynchronous multidimensional tensor transfers between global memory and shared memory without consuming SM register or ALU cycles.
- **WGMMA (Warpgroup Matrix Multiply-Accumulate):** Hardware instruction family that executes matrix multiplications asynchronously across an entire 128-thread warpgroup.
- **Warp specialization:** Partitioning SM warps such that producer warps focus exclusively on data prefetching via TMA, while consumer warps focus purely on WGMMA compute.
- **Double buffering / overlap:** Overlapping computation on current shared memory tiles while asynchronously staging the next tile in background buffers.

<svg viewBox="0 0 720 264" role="img" aria-label="Timeline with four lanes: HBM, TMA, shared memory and WGMMA. K/V blocks 1, 2 and 3 sit in HBM. TMA loads block 1 into buffer A, block 2 into buffer B, and block 3 back into buffer A. WGMMA computes on each block one slot after it was loaded, so while it computes block 1 the TMA is already loading block 2, and while it computes block 2 the TMA loads block 3. The shaded overlap windows show data movement for block k plus 1 hiding behind the math for block k." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tm-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tm-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Double buffering on Hopper: TMA loads block k+1 while WGMMA computes block k</text>
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
<text x="214.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V block 1</text>
<rect x="286" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="350.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V block 2</text>
<rect x="422" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="486.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V block 3</text>
<rect x="150" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="214.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">load 1</text>
<rect x="154" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="214.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">buffer A</text>
<line x1="214.0" y1="128" x2="214.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="286" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="350.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">compute 1</text>
<path d="M272 178 L296 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="286" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="350.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">load 2</text>
<rect x="290" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="350.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">buffer B</text>
<line x1="350.0" y1="128" x2="350.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="422" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="486.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">compute 2</text>
<path d="M408 178 L432 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="422" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="486.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">load 3</text>
<rect x="426" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="486.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">buffer A</text>
<line x1="486.0" y1="128" x2="486.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="558" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="622.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">compute 3</text>
<path d="M544 178 L568 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="286" y="94" width="128" height="140" rx="6" style="fill:var(--c-accent-2);fill-opacity:.06;stroke:var(--c-accent-2);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="422" y="94" width="128" height="140" rx="6" style="fill:var(--c-accent-2);fill-opacity:.06;stroke:var(--c-accent-2);stroke-width:1;stroke-dasharray:4 3"/>
<text x="418.0" y="252" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">overlap: data movement for k+1 hides behind the math for k</text>
<text x="704" y="44" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">time →</text>
</svg>

**Intuition:** In a kitchen, a prep cook chops the next batch of ingredients while the chef is cooking the current dish. The stove never stands idle waiting for ingredients. However, if counter space (shared memory) is cramped or synchronization falters, adding more cooks leads to contention rather than speed.

### 4.8. Tracing an End-to-End Inference Request

In training, vast token batches are processed concurrently; during serving, prefill and decode exhibit fundamentally distinct operational behaviors. Let us now trace a production user request from gateway ingress to final token egress, clarifying exactly which architectural bottleneck governs each user-facing metric.

**Scenario:** A user submits a 2,000-token prompt and receives a 200-token completion. The model belongs to the 70B class, requiring roughly 140 GB in BF16 weights. The serving stack hosts the model across multiple GPUs, scheduled dynamically by an inference serving engine.



<svg viewBox="0 0 720 314" role="img" aria-label="A 2,000-token prompt with a 200-token answer on a 70B model. On the CPU and service side the request goes through the tokenizer and the scheduler, which queues it and adds it to a continuous batch. On the GPU, prefill processes all 2,000 tokens at once with large GEMMs and fills the KV cache. Decode then produces one token per step, 200 times, reading weights and the growing KV cache, which tends to be memory-bound; each new token loops back into decode and is detokenized and streamed. Below, the user&#x27;s timeline: queue, tokenize, prefill and the first decode make up TTFT; the gap between later tokens is TPOT." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="il-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="il-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="il-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">One request: 2,000 prompt tokens in, 200 tokens out (70B, BF16 ≈ 140 GB)</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">CPU / service</text>
<text x="16" y="148" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">GPU</text>
<line x1="16" y1="120" x2="704" y2="120" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<rect x="16" y="60" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="71.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Request</text>
<text x="71.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">prompt text</text>
<rect x="140" y="60" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="195.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Tokenizer</text>
<text x="195.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">text → IDs</text>
<rect x="264" y="60" width="140" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="334.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Scheduler</text>
<text x="334.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">queue + continuous batch</text>
<line x1="126" y1="83" x2="138" y2="83" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="250" y1="83" x2="262" y2="83" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="250" y="132" width="164" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="332.0" y="151.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Prefill</text>
<text x="332.0" y="166.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2,000 tokens at once</text>
<text x="332.0" y="181.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">big GEMMs → compute-leaning</text>
<line x1="334" y1="106" x2="334" y2="130" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="132" width="92" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="476.0" y="151.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">KV cache</text>
<text x="476.0" y="166.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">grows per</text>
<text x="476.0" y="181.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">token</text>
<line x1="414" y1="161" x2="428" y2="161" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="538" y="132" width="166" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="621.0" y="151.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Decode</text>
<text x="621.0" y="166.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 token / step × 200</text>
<text x="621.0" y="181.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">weights + KV → memory-bound</text>
<line x1="522" y1="161" x2="536" y2="161" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M690 190 V212 H560 V192" marker-end="url(#il-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="625" y="226" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px">next token</text>
<rect x="538" y="60" width="166" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="621.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Detokenize + stream</text>
<text x="621.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">token → text</text>
<line x1="621" y1="130" x2="621" y2="108" marker-end="url(#il-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="16" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">what the user feels</text>
<rect x="16" y="252" width="130" height="24" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="267.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">queue + tokenize</text>
<rect x="146" y="252" width="210" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="251.0" y="267.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">prefill + first decode</text>
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
<text x="392" y="302" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">TPOT = gap between tokens</text>
</svg>



**Stage 1 — Tokenization:** The user's prompt text is encoded into discrete token IDs according to the model's vocabulary. This phase is typically executed on the CPU host. For short prompts, tokenization, serialization, and server-side ingress queuing can constitute a noticeable fraction of total perceived latency.

**Stage 2 — Scheduling:** In production environments with concurrent traffic, the serving engine batches incoming requests. Here, **continuous batching** (iteration-level scheduling) dynamically incorporates new requests at every decode step. The objective is to maximize arithmetic reuse of model weights per memory load. Conversely, servicing higher concurrency increases KV cache pressure and queuing delays.

**Stage 3 — Prefill:** The 2,000 prompt tokens are passed in parallel through the Transformer layers. Because matrices are sufficiently large, Tensor Core compute utilization is high. This phase directly dictates **TTFT (Time to First Token)**. Note that TTFT encompasses not only the raw prefill compute time, but also ingress queuing, scheduling overhead, and the initial decode step.

**Stage 4 — Decode:** Once the first token is emitted, subsequent tokens are produced autoregressively. Rather than recomputing representations for all preceding tokens, the engine reads their key/value tensors from the KV cache. At small batch sizes, streaming model weights from HBM dominates step time. As batch size or context length expands, attention compute and KV cache memory bandwidth become the primary bottlenecks.

**Stage 5 — Streaming Completion:** As tokens are emitted, perceived generation speed is measured via **TPOT (Time per Output Token)** or per-user tokens per second. Averages alone obscure production realities: How does tail latency (p95/p99) degrade under load? When a few users submit massive context windows, are short interactive queries starved in queues?

This end-to-end flow isolates three distinct optimization regimes. To optimize **initial responsiveness**, profile prefill efficiency, queuing latencies, and first-token generation. To maximize **generation speed**, target decode step execution, HBM bandwidth utilization, KV cache layout, and dynamic batching. To maximize **concurrent user capacity**, optimize KV cache memory allocation, consider weight/KV quantization, and enforce strict admission control to protect tail latency.

> **Key Architectural Takeaway:** In training, the optimization objective is maximizing *training tokens processed per second* under batch parallelism. In online serving, aggregate throughput is meaningless if decoupled from user-centric SLAs: when the first token arrives (TTFT) and how smoothly tokens stream (TPOT).

---

### 4.9. Sizing the KV Cache: Why the Same Model Runs Smoothly with 10 Users but Chokes on 100

Imagine operating a support center speaking with dozens of clients simultaneously. For each client, you must retain the full context of their conversation. One client's history is two sentences; another's spans hundreds of pages. As concurrency scales, the primary burden is not just processing new incoming sentences, but **storing the stateful representation of every past exchange** in active memory.

In a Transformer, the **KV cache (Key–Value cache)** serves this exact function: during autoregressive decoding, it caches the key and value projections of all prior tokens so that attention can be computed without redundant recomputation. It is not a database of raw text; it consists of model- and layer-specific floating-point tensors. Unlike static model weights, **the KV cache scales dynamically with request concurrency and context length**.

We can formulate the KV cache footprint per token across all Transformer layers as follows:

$$
M_{\mathrm{KV/token}}=2\times L\times H_{\mathrm{KV}}\times d_{\mathrm{head}}\times b
$$

- $2$: Distinct tensors for Key and Value.
- $L$: Number of Transformer layers.
- $H_{\mathrm{KV}}$: Number of KV heads (in GQA/MQA, this is strictly less than query heads).
- $d_{\mathrm{head}}$: Dimension per head ($d_{\mathrm{model}} / H_{\mathrm{Q}}$).
- $b$: Bytes per stored scalar (e.g., 2 bytes for BF16/FP16).

**Concrete Example:** For a model analogous to Llama 3.1 70B, with 80 layers, 8 KV heads, a head dimension of 128, and a BF16 KV cache ($b=2$):

$$
M_{\mathrm{KV/token}}=2\times80\times8\times128\times2
=327\,680\;\mathrm{byte}\approx320\;\mathrm{KiB}
$$

For a single 8,192-token conversation, the KV cache alone demands $320\;\mathrm{KiB}\times8192\approx2.5\;\mathrm{GiB}$. At a 32,768-token context, this expands to roughly **10 GiB**. Note that these values represent **logical KV cache volume**; tensor parallelism (TP), context parallelism (CP), block allocation granularity, quantization, padding, and framework runtime overhead determine the actual physical footprint per GPU. [K21]

Consider the operational consequence: with 16 concurrent long-context sessions, the KV cache alone requires tens to hundreds of gigabytes. Concurrently, static model weights, runtime scratchpads, and activation buffers must reside on the same GPUs. Consequently, even when `nvidia-smi` reports low SM compute utilization, the server can experience out-of-memory (OOM) faults or aggressive request thrashing due to memory starvation.

<svg viewBox="0 0 720 244" role="img" aria-label="Stacked bars for a 70B model with 8,192-token contexts. KV per token is 2 times 80 times 8 times 128 times 2 bytes, 320 KiB, so each request needs 2.5 GiB or 2.68 GB. With 10 users: 140 GB of weights plus 26.8 GB of KV is about 167 GB, which fits under an illustrative 4 × H100 pool of 320 GB, before workspace. With 100 users: 140 plus 268.4 GB is about 408 GB, which does not fit." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="kg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Same 70B model, 10 vs. 100 users with 8,192-token contexts</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">KV per token = 2 × 80 × 8 × 128 × 2 B = 320 KiB → 2.5 GiB (2.68 GB) per 8K request · illustrative pool: 4 × H100 = 320 GB</text>
<text x="16" y="94" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">10 users</text>
<rect x="130" y="76" width="161.778" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="210.88888888888889" y="95.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">weights 140</text>
<rect x="291.778" y="76" width="30.9689" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="330.74666666666667" y="88" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">167 GB</text>
<text x="330.74666666666667" y="104" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">fits (+ workspace?)</text>
<text x="324.74666666666667" y="122" text-anchor="start" style="fill:var(--c-warn);font-size:11px">KV 26.8</text>
<text x="16" y="156" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">100 users</text>
<rect x="130" y="138" width="161.778" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="210.88888888888889" y="157.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">weights 140</text>
<rect x="291.778" y="138" width="310.151" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="446.85333333333335" y="157.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">KV 268.4</text>
<text x="609.9288888888889" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">408 GB</text>
<text x="609.9288888888889" y="166" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">does not fit</text>
<line x1="499.778" y1="60" x2="499.778" y2="202" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="499.77777777777777" y="216" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">4 × 80 GB = 320 GB</text>
<text x="130.0" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="245.55555555555554" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100</text>
<text x="361.1111111111111" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">200</text>
<text x="476.66666666666663" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">300</text>
<text x="592.2222222222222" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">400</text>
<text x="690" y="232" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">GB</text>
</svg>

**Why does GQA (Grouped-Query Attention) revolutionize this equation?** In standard Multi-Head Attention (MHA), every query head has its own dedicated key and value head ($H_{\mathrm{KV}} = H_{\mathrm{Q}}$). GQA groups multiple query heads to share a single KV head. Intuitively, multiple analysts share the same filing cabinet. This slashes KV cache memory footprint proportionally, though architectural capacity and quality trade-offs must be resolved during pre-training. [K21]

> **Architectural Decision:** Never size serving clusters using naive `model size / GPU memory` math. Production capacity planning requires calculating **Weights + KV Cache + Execution Buffers + Headroom Margin**, parameterized against empirical prompt/completion sequence length distributions.

### 4.10. PagedAttention and Prefix Caching: Managing KV Memory Intelligently

Imagine allocating a massive, contiguous bookshelf in a library to every patron based on the maximum number of books they might hypothetically borrow. Most readers borrow only a few books, yet the reserved shelf space remains locked and unavailable to others. This mirrors the **memory fragmentation** and over-allocation waste endemic to legacy KV cache systems.

**PagedAttention** solves this by dividing the KV cache into fixed-size virtual blocks stored in non-contiguous physical GPU memory, mimicking OS virtual memory paging. While the model logically sees a contiguous sequence of tokens, the memory manager maps logical blocks to arbitrary physical memory slots via a block table. This eliminates external fragmentation and dynamically allocates blocks on demand, unlocking dramatic throughput gains under variable-length workloads. [K22]

The second major optimization is **prefix caching** (prompt caching): when multiple requests share identical prompt prefixes, precomputed KV blocks for that shared prefix are retained in cache and reused. For instance, if every request begins with an extensive system prompt or a shared reference document, the engine avoids redundant prefill compute, serving subsequent requests almost instantaneously.

Here are both ideas in one picture:

<svg viewBox="0 0 720 258" role="img" aria-label="Two requests share a common prefix. Request A has logical KV blocks P0, P1, A2 and A3; request B has P0, P1 and B2. A block table maps each logical block to a physical slot in HBM: P0 to row 0 column 3, P1 to row 2 column 2, A2 to row 0 column 1, A3 to row 1 column 4, B2 to row 1 column 0; the other slots are free. Because both requests point to the same physical P0 and P1 blocks, the shared prefix is prefilled once. Blocks need not be contiguous, which reduces fragmentation. Sharing across tenants must be fenced, for example with vLLM cache_salt, and decode work and per-token K and V do not disappear." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="pg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="pg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">PagedAttention + prefix caching: logical sequence ≠ physical layout</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">logical KV blocks (per request)</text>
<text x="430" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">physical GPU memory (HBM)</text>
<text x="16" y="86" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">request A</text>
<rect x="100" y="70" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="156" y="70" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="182.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="212" y="70" width="52" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="238.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A2</text>
<rect x="268" y="70" width="52" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A3</text>
<text x="16" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">request B</text>
<rect x="100" y="134" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="156" y="134" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="182.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="212" y="134" width="52" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="238.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B2</text>
<rect x="430" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="456.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="486" y="62" width="52" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="512.0" y="81.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A2</text>
<rect x="542" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="568.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="598" y="62" width="52" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="624.0" y="81.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="654" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="680.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="430" y="98" width="52" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="456.0" y="117.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B2</text>
<rect x="486" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="512.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="542" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="568.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="598" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="624.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="654" y="98" width="52" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="680.0" y="117.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A3</text>
<rect x="430" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="456.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="486" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="512.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="542" y="134" width="52" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="568.0" y="153.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="598" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="624.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="654" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="680.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">free</text>
<rect x="330" y="64" width="84" height="100" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="372" y="82" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">block table</text>
<text x="372" y="98" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">P0 → (0,3)</text>
<text x="372" y="111" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">P1 → (2,2)</text>
<text x="372" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">A2 → (0,1)</text>
<text x="372" y="137" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">A3 → (1,4)</text>
<text x="372" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">B2 → (1,0)</text>
<line x1="316" y1="114" x2="328" y2="114" marker-end="url(#pg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="414" y1="114" x2="428" y2="114" marker-end="url(#pg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="182" width="688" height="64" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="202" text-anchor="start" style="fill:var(--c-text);font-size:12px">Shared prefix (P0, P1): both requests point to the same physical blocks → its prefill runs once.</text>
<text x="28" y="220" text-anchor="start" style="fill:var(--c-text);font-size:12px">Blocks need not be contiguous → no giant up-front reservation, less fragmentation.</text>
<text x="28" y="238" text-anchor="start" style="fill:var(--c-danger);font-size:12px">Sharing across tenants must be fenced (e.g. vLLM cache_salt); decode work and per-token K/V do not disappear.</text>
</svg>

However, two critical operational boundaries must be respected:

1. Prefix caching primarily eliminates **prefill compute for the shared prompt prefix**; it does not eliminate the sequential decode cost of generating new output tokens.
2. In multi-tenant enterprise architectures, cache sharing is not merely a performance optimization—it is a security boundary requiring rigorous **tenant isolation**. Sensitive contexts belonging to distinct tenants must never collide or cross-contaminate. Cache keys and namespace boundaries must enforce access control. For example, vLLM supports a `cache_salt` configuration to prevent unauthorized tenants from reusing cached prefix blocks. This separation guards not only against direct data disclosure, but also against timing-based side-channel attacks where cache hit latency differentials leak information about another user's prompt content. [K22]

**Continuous batching** (iteration-level scheduling) dynamically coalesces ready token generation steps from active requests into unified execution batches at each iteration. Requests enter the batch immediately after prefill and exit as soon as they emit an end-of-sequence token, eliminating head-of-line blocking. However, excessively aggressive batch sizes increase decode iteration time, inflating per-user latency. An optimal serving scheduler **jointly optimizes aggregate throughput and queuing/decode latency**.

### 4.11. Disaggregating Prefill and Decode: When Does Split Serving Make Sense?

Earlier, we observed that prefill and decode exhibit fundamentally contrasting execution profiles. Consider a high-volume restaurant: the kitchen team prepares courses in bulk batches, while the waitstaff delivers plates to tables continuously. When a single team handles both roles simultaneously, an influx of new orders clogs the kitchen, starving seated guests who are waiting for their next dish.

**Disaggregated serving** partitions cluster infrastructure into dedicated prefill and decode worker pools. An incoming request is first scheduled onto high-throughput prefill nodes; once intermediate KV states are computed, they are transmitted across high-speed interconnects (e.g., InfiniBand/RoCE or PCIe) to dedicated decode nodes that handle token generation. This enables independent autoscaling and hardware specialization tailored to each phase's computational signature. Frameworks such as NVIDIA Dynamo and TensorRT-LLM natively support this disaggregated topology. [K23]



<svg viewBox="0 0 720 324" role="img" aria-label="Two panels. A, aggregated: a router balances load across workers that each run both prefill and decode; there is no KV transfer but the phases interfere. B, disaggregated: a KV-aware gateway sends requests to a compute-heavy prefill pool suited to long input sequences; the resulting KV cache is transferred over RDMA or NVLink to a separate HBM-bound decode pool suited to long outputs; the decode pool feeds back free KV capacity, cached prefixes and queue depth to the router. Splitting pays only if the KV transfer costs less than the interference it removes." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker><marker id="dg-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Aggregated vs. disaggregated serving</text>
<rect x="16" y="38" width="688" height="92" rx="6" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">A · aggregated: every worker does prefill + decode</text>
<rect x="28" y="70" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83.0" y="90.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Router</text>
<text x="83.0" y="105.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">load balance</text>
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
<text x="704" y="58" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">+ no KV transfer  − phases interfere</text>
<rect x="16" y="142" width="688" height="150" rx="6" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">B · disaggregated: separate prefill and decode pools</text>
<rect x="28" y="186" width="110" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83.0" y="211.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Gateway / router</text>
<text x="83.0" y="226.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">KV-aware</text>
<rect x="176" y="176" width="170" height="76" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="261" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Prefill pool</text>
<text x="261" y="212" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">compute-heavy · long ISL</text>
<rect x="190" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="227" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="264" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="301" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="496" y="176" width="170" height="76" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="581" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Decode pool</text>
<text x="581" y="212" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">HBM-bound · long OSL</text>
<rect x="510" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="547" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="584" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="621" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="138" y1="214" x2="174" y2="214" marker-end="url(#dg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="346" y1="214" x2="494" y2="214" marker-end="url(#dg-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<text x="420" y="206" text-anchor="middle" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">KV transfer</text>
<text x="420" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">RDMA / NVLink</text>
<path d="M581 252 V272 H83 V244" marker-end="url(#dg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="332" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">feedback: free KV capacity, cached prefixes, queue depth</text>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Wins only if the KV transfer costs less than the interference it removes — measure both with your ISL/OSL mix.</text>
</svg>



This architecture is not universally superior. Moving intermediate states introduces a critical **KV transfer overhead** over the cluster network. If network bandwidth is insufficient or latency is high, the performance gains of workload specialization are entirely swallowed by transfer overhead. Architects must rigorously weigh the trade-offs:

| Topology | Strengths | Trade-offs | Ideal Workload Profile |
|---|---|---|---|
| **Aggregated**: Uniform workers (prefill + decode) | Zero inter-node KV transfer complexity, simpler operational footprint | Prefill and decode compete for SMs and HBM, causing latency jitter | Small-to-medium deployments, uniform request lengths, simpler infrastructure |
| **Disaggregated**: Dedicated prefill & decode pools | Independent horizontal scaling, isolated compute/memory optimization, predictable SLAs | Network KV transfer latency, requires high-bandwidth fabrics, complex orchestration | Large-scale clusters, high concurrency, extreme ISL/OSL divergence |

Here, **ISL (Input Sequence Length)** denotes prompt token count per request, while **OSL (Output Sequence Length)** denotes generated completion token count. A workload summarizing lengthy legal documents exhibits high ISL and low OSL, whereas an interactive creative coding assistant exhibits moderate ISL and high OSL. Evaluating these fundamentally divergent workloads under an identical aggregate benchmark produces meaningless performance conclusions.

**KV-aware routing** schedules incoming queries not merely by finding the least-loaded GPU, but by identifying which worker already holds the relevant prefix blocks in its KV cache while balancing current queue depth. However, naive routing that funnels excessive traffic to a single worker solely to maximize cache hit rates risks inducing severe queuing hotspots. [K23]

### 4.12. Speculative Decoding, Admission Control, and Output Quality: Not All Tokens Cost the Same

Imagine an assistant drafting paragraphs while a senior editor rapidly reviews and validates them in batches. **Speculative decoding** operates on this principle: a smaller, faster draft model (or speculative proposal mechanism) drafts candidate tokens autoregressively; the large target model then verifies all candidate tokens in a single parallel forward pass. When speculation succeeds, multiple tokens are accepted in the time of a single target forward pass, significantly lowering TPOT.

However, executing the draft model incurs its own computational and memory overhead. If the acceptance rate is low or verification overhead dominates, speculative execution can actually increase overall latency. Furthermore, serving architectures must guarantee that speculative verification algorithms (such as speculative sampling) strictly preserve the target model's output probability distribution. Speculative decoding represents another critical juncture where latency gains must be co-validated against output quality.

Production documentation from vLLM highlights that speculative decoding gains vary substantially across traffic patterns, hardware configurations, and model domain alignment. [K19]

**Admission control** prevents a cluster from blindly accepting traffic beyond its physical capacity limit. For example, if a serving worker is processing 50 high-context requests, admitting a 51st request without sufficient buffer can trigger catastrophic KV cache eviction or page thrashing, destroying p99 latency for all active sessions. A resilient architecture queues excess queries, routes them to auxiliary nodes, or applies load-shedding policies based on explicit capacity thresholds.

> **AI Architect Checklist:** Production telemetry must expose **decoupled throughput curves** for prefill and decode, KV cache utilization and eviction rates, prefix cache hit ratios, TTFT/TPOT p95/p99 percentiles, queue wait times, preempted/canceled requests, and per-tenant cost attribution. Monitoring only "GPU utilization: 95%" provides a dangerously incomplete operational picture.

---

---

### Two Requests, Same Model, Fundamentally Divergent Behavior

One user types "Hello!" and requests a 40-token reply. Another user uploads a 50-page legal contract and requests a 1,000-token summary. Both queries target the identical 70B parameter model. Yet from the GPU's execution standpoint, these two requests represent completely different computational workloads.

The first request requires negligible **prefill** compute, followed by dozens of sequential memory-bandwidth-bound decode steps. The second request presents an enormous prompt; prefill is computationally intensive, allocating a massive initial KV cache footprint before entering a prolonged decode phase. When the platform serves both concurrently, the **scheduler** must arbitrate hardware resources between compute-heavy prefill operations and latency-sensitive decode steps.



<svg viewBox="0 0 720 326" role="img" aria-label="Schematic timelines for two requests on the same 70B model. A short Hello prompt queues briefly, needs a tiny prefill, reaches its first token early, then runs 40 decode steps; its KV cache is small and short-lived. A 50-page contract needs a long prefill, so its first token comes later, then runs 1,000 decode steps while holding a large KV cache for the whole time. Colours separate the logical prefill and decode workloads, which may share one GPU or run in separate pools." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tq-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tq-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tq-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Two requests on the same 70B model (schematic, not to scale)</text>
<text x="16" y="70" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">“Hello!”</text>
<text x="16" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">short prompt · 40-token reply</text>
<rect x="196" y="56" width="46" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="219.0" y="74.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">queue</text>
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
<text x="452" y="76" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">40 decode steps</text>
<line x1="278" y1="50" x2="278" y2="96" style="stroke:var(--c-text);stroke-width:1.2"/>
<text x="278" y="110" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TTFT</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">50-page contract</text>
<text x="16" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">long prompt · 1,000-token reply</text>
<rect x="196" y="148" width="46" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="219.0" y="166.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">queue</text>
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
<text x="580" y="168" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">1,000 decode steps</text>
<line x1="418" y1="142" x2="418" y2="188" style="stroke:var(--c-text);stroke-width:1.2"/>
<text x="418" y="202" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TTFT</text>
<text x="16" y="248" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">KV cache held</text>
<rect x="196" y="238" width="40" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="242" y="251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">“Hello!”: small, short-lived</text>
<rect x="196" y="262" width="364" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="566" y="275" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">contract: large, held for</text>
<text x="566" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">all 1,000 steps</text>
<text x="16" y="314" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Colours separate two logical workloads (prefill, decode); they may run on the same GPU or on separate pools.</text>
</svg>



Note that prefill and decode **do not necessarily execute on separate physical GPUs**. The diagram illustrates two logical stages; unified execution on a single GPU cluster is standard in aggregated setups. In disaggregated deployments, however, transferring the KV states between prefill and decode introduces distinct network and latency costs.

### Deriving the KV Cache Sizing Formula Step by Step

During Transformer decode, key and value representations of preceding tokens are not recomputed; they are preserved in a dynamically allocated HBM buffer (*KV cache*):

$$
M_{\mathrm{KV}}=2\times L\times H_{\mathrm{KV}}\times d_{\mathrm{head}}\times S\times b
$$

Each multiplier in this formulation carries direct physical meaning:
- The leading factor of **2** accounts for storing distinct Key and Value state tensors at each layer.
- $H_{\mathrm{KV}}$ (KV head count) leverages Grouped-Query Attention (GQA) to decouple KV heads from query heads ($H_Q$), compressing this memory footprint by 4× to 8×.
- $d_{\mathrm{head}}$ and layer count $L$ dictate vertical model dimensions; $b$ specifies numeric precision (2 bytes for BF16, 1 byte for FP8).
- As sequence length $S$ expands, the memory footprint of each active user session grows linearly.

As demonstrated in Section 4.9 for 70B, a single 8K session consuming $2.5\ \mathrm{GiB}$ appears manageable in isolation. However, while static model weights remain fixed in HBM, the KV cache scales proportionally with concurrent traffic ($100 \times 2.5 \approx 250\ \mathrm{GiB}$), proving that the serving memory wall is governed not by static weights, but by live session concurrency. [K21], [K49]


### Why PagedAttention Borrowed Virtual Memory Principles

Imagine a hotel reserving an unbroken, contiguous block of rooms for every arriving guest based on their maximum possible stay. If guests depart unpredictably, rooms sit empty and fragmented throughout the building; the hotel appears booked to capacity despite being largely vacant. Conventional contiguous KV cache allocation suffers from this exact inefficiency under dynamic, variable-length sequence generation.

PagedAttention resolves this by dividing the KV cache into fixed-size token blocks, decoupling logical sequence continuity from physical memory contiguity. The runtime stitches together non-contiguous physical memory blocks to represent a single logical sequence. This eliminates external fragmentation and facilitates seamless prefix sharing. **Crucially, however, it does not mathematically eliminate the fundamental KV state required per token.** [K49]

### Why High Aggregate Tokens/s Does Not Equal a Responsive User Experience

Suppose a serving cluster outputs 10,000 tokens/second in aggregate. On paper, this appears exceptional. However, if 500 concurrent users are submitting queries simultaneously, many may suffer substantial queuing delays before their generation even begins. While aggregate throughput looks high, **TTFT p99** may be unacceptable. Engineering production LLM systems requires evaluating three coupled metrics:

- **TTFT (Time to First Token):** Total elapsed time from request arrival to first generated token, encompassing network transit, routing, queue wait times, and prefill compute.
- **TPOT (Time per Output Token) / ITL (Inter-Token Latency):** The generation delay between successive tokens, governing the perceptual fluidity of the streaming completion.
- **Goodput:** The volume of useful completion tokens produced strictly within predefined SLA latency and quality thresholds.

The batch configuration maximizing raw throughput rarely delivers the optimal interactive user experience. This trade-off is not an engineering flaw, but a fundamental characteristic of queued queuing systems under concurrent load.

## 5. Why Capacity Is Never Just GPU Count: HBM, KV Cache, Traffic, and Power

GPU procurement decisions should never rely on intuitive guesswork; instead, each limiting resource must be modeled independently. Just as an automobile requires a fuel tank and cooling capacity proportional to its engine power and cargo weight, LLM serving depends fundamentally on **weight memory, KV cache, compute throughput, memory bandwidth, and traffic volatility**.

### 5.1. Model Memory: The Initial Lower Bound

A rough estimate for the weight footprint of a dense model:

$$
M_{\mathrm{weights}}\approx N_{\mathrm{params}}\times\frac{b}{8}
$$

Here, $b$ denotes the number of bits per parameter. For example, a 70-billion-parameter model in BF16 (16 bits) requires approximately **140 GB** of raw weights. That same model occupies roughly 70 GB in INT8 and approximately 35 GB in 4-bit precision. However, quantization scales, zero points, tensor padding, mixed layer precisions, and runtime execution buffers invariably increase this footprint.

Although the model theoretically fits into 2×80 GB of GPU memory, serving inference also requires reserving scratch space for CUDA graphs, activation buffers, and the KV cache. Consequently, the condition $M_{\mathrm{weights}} < M_{\mathrm{GPU}}$ is **necessary, but far from sufficient**.

### 5.2. KV Cache: The Hidden Capacity Consumer

The baseline established in Section 4—roughly $320\ \mathrm{KiB}$ per token, scaling to $2.5\ \mathrm{GiB}$ for an 8K request on a 70B model—represents the most volatile constraint in serving infrastructure:

$$
M_{\mathrm{KV,total}} \approx N_{\mathrm{concurrent}} \times M_{\mathrm{KV/request}}
$$

The most widespread architectural trap is assuming that "once weights fit into memory, remaining HBM is free headroom." In a cluster of four 80 GB H100s (320 GB total HBM) hosting 140 GB of model weights, an apparent 180 GB headroom exists. Yet merely 70 concurrent 8K sessions ($70 \times 2.5 \approx 175\ \mathrm{GB}$) completely exhaust this remaining budget.

In production environments, capacity planning must incorporate three additional non-negotiable overheads:
1. **Memory fragmentation:** Even under PagedAttention, block granularity and tail allocation inefficiencies demand a 10–15% buffer margin.
2. **Transient activation headroom:** Prefill phases generate substantial intermediate activation tensors during large GEMMs that require dedicated workspace.
3. **Allocator reserve:** Runtime frameworks (such as vLLM or TensorRT-LLM) mandate non-evictable memory reserves to prevent catastrophic OOM halts.

Consequently, model weights form a static floor, whereas the KV cache constitutes a dynamic, traffic-dependent ceiling.


### 5.3. Serving Capacity: Translating Tokens/s into Request Throughput

Suppose that for a specific model-GPU configuration, your **empirically measured** successful generation throughput is 6,000 output tokens/s. If an average request generates 300 tokens, a naive first estimate might suggest:

$$
\lambda_{\mathrm{max}}\approx\frac{6000}{300}=20\ \text{req/s}
$$

This figure serves only as a **coarse upper bound**: it omits the prefill cost of input tokens, long prompt variance, batch churn, routing overhead, safety moderation, and p99 SLO constraints. While this back-of-the-envelope calculation identifies a starting baseline for benchmark design, it cannot justify a procurement decision on its own.

**Little's Law** provides a robust sanity check for average concurrency:

$$
L=\lambda W
$$

If 10 requests arrive per second and each request spends an average of 4 seconds in the system, the average in-flight concurrency is roughly **40**. In queuing theory, however, averages offer zero guarantees during p99 traffic spikes. In production, serving systems must be benchmarked against realistic arrival curves and empirical p50/p95 input-output token length distributions.

### 5.4. Capacity = The Tightest Bottleneck

Attainable serving capacity is governed by the minimum of several competing bounds:

$$
C_{\mathrm{serving}} \lesssim \min(C_{\mathrm{compute}},C_{\mathrm{HBM}},C_{\mathrm{KV}},C_{\mathrm{network}},C_{\mathrm{SLO}})
$$

For example, a GPU cluster might generate high aggregate token throughput, yet breach the p95 TTFT threshold, rendering a large fraction of that throughput unusable from a service standpoint. The operational objective is never merely **peak output volume**, but **valid responses fulfilled strictly within SLO budgets**.

### 5.5. Why Headroom Is Not a Luxury

If all ten GPUs in a ten-GPU pool are pinned at peak utilization, a single hardware fault or rolling update will cascade into immediate service degradation. There is no universal percentage for capacity headroom; sizing must stem from N+1 redundancy requirements, fault-domain diversification, burst traffic ratios, pod spin-up latency, and latency SLOs. Crucially, the non-trivial cold-start duration required to stream large model weights into HBM destroys the naive assumption that "we can simply spin up a pod on demand."

### 5.6. Power and Cooling: The Secondary Capacity Ceiling

Assuming 128 eight-GPU servers each draw a **hypothetical** 10 kW, the compute chassis alone require a 1.28 MW power budget—excluding NICs, top-of-rack switches, storage fabrics, and distribution losses. Assuming a PUE of 1.3, the aggregate facility power footprint scales to roughly 1.66 MW. This serves as an **illustrative design baseline, not an empirical audit of any specific facility**.

PUE (Power Usage Effectiveness) is the ratio of total facility power consumption to IT equipment power consumption. While a facility may maintain a low PUE, a model's energy cost per token may still remain steep; tracking application-level efficiency metrics such as **kWh per million successful tokens** is therefore imperative.

---

## 6. Production Inference: The Life of a Request from Queue to Final Token

A successful HTTP 200 return from an API call does not guarantee an acceptable user experience. From the end-user's vantage point, latency splits into two distinct wait states: **When does the first token arrive?** and **at what pace does subsequent text stream?** In technical terms, these correspond to Time to First Token (TTFT) and Time Per Output Token (TPOT) / Inter-Token Latency (ITL).

### 6.1. Deconstructing TTFT

$$
T_{\mathrm{TTFT}}\approx T_{\mathrm{auth}}+T_{\mathrm{safety}}+T_{\mathrm{data}}+T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{first\ decode}}+T_{\mathrm{network}}
$$

Depending on the serving pipeline, some of these phases may overlap. $T_{\mathrm{data}}$ arises exclusively in workflows requiring external retrieval (e.g., RAG vector search or database lookups). Furthermore, un-flushed server-side streaming buffers or reverse proxies buffering initial chunks can inject stealth latency before the first token reaches the client.

Representative scenario: 20 ms authentication, 30 ms safety guardrails, 150 ms optional data retrieval, 500 ms queuing, 650 ms prefill execution, 50 ms first-token decode, and 50 ms network/proxy overhead culminate in a 1.45-second TTFT. These values are purely illustrative; **every segment of this waterfall must be instrumented and monitored via distributed tracing**.

<svg viewBox="0 0 720 298" role="img" aria-label="Illustrative TTFT waterfall totalling 1.45 seconds: authentication 20 ms, safety 30 ms, optional data service 150 ms, queue 500 ms, prefill 650 ms, first decode 50 ms, and network or proxy 50 ms. Queue and prefill account for 79 percent. Each row must be measured with real tracing." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Where 1.45 s of TTFT goes (illustrative waterfall)</text>
<text x="180" y="59" text-anchor="end" style="fill:var(--c-text);font-size:12px">auth</text>
<rect x="190" y="47" width="6.34483" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="202.3448275862069" y="60" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">20 ms</text>
<text x="180" y="87" text-anchor="end" style="fill:var(--c-text);font-size:12px">safety</text>
<rect x="196.345" y="75" width="9.51724" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="211.86206896551724" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">30 ms</text>
<line x1="196.345" y1="65" x2="196.345" y2="75" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="115" text-anchor="end" style="fill:var(--c-text);font-size:12px">data service (optional)</text>
<rect x="205.862" y="103" width="47.5862" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="259.44827586206895" y="116" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">150 ms</text>
<line x1="205.862" y1="93" x2="205.862" y2="103" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="143" text-anchor="end" style="fill:var(--c-text);font-size:12px">queue</text>
<rect x="253.448" y="131" width="158.621" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.6"/>
<text x="418.0689655172414" y="144" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">500 ms</text>
<line x1="253.448" y1="121" x2="253.448" y2="131" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="171" text-anchor="end" style="fill:var(--c-text);font-size:12px">prefill</text>
<rect x="412.069" y="159" width="206.207" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="624.2758620689656" y="172" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">650 ms</text>
<line x1="412.069" y1="149" x2="412.069" y2="159" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="199" text-anchor="end" style="fill:var(--c-text);font-size:12px">first decode</text>
<rect x="618.276" y="187" width="15.8621" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="640.1379310344828" y="200" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">50 ms</text>
<line x1="618.276" y1="177" x2="618.276" y2="187" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="227" text-anchor="end" style="fill:var(--c-text);font-size:12px">network / proxy</text>
<rect x="634.138" y="215" width="15.8621" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="656.0" y="228" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">50 ms</text>
<line x1="634.138" y1="205" x2="634.138" y2="215" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<line x1="190" y1="246" x2="650" y2="246" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="190" y1="246" x2="190" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="190.0" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0 s</text>
<line x1="348.621" y1="246" x2="348.621" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="348.62068965517244" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0.5 s</text>
<line x1="507.241" y1="246" x2="507.241" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="507.2413793103448" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1.0 s</text>
<line x1="650" y1="246" x2="650" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="650.0" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1.45 s</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Queue + prefill = 79% here. Every row must come from real tracing; terms can overlap in practice.</text>
</svg>

### 6.2. Why Queuing Latency Blows Up Non-linearly

As serving systems approach saturation, queuing latency escalates non-linearly. A restaurant kitchen running ten order stations can comfortably manage an arrival rate of six orders per minute; at nine orders, even microscopic variance triggers massive backlogs. In LLM inference, there is an inherent structural tension between **maximizing GPU utilization and guaranteeing low p99 latency**.

Consequently, scaling policies driven solely by average GPU utilization (e.g., standard Kubernetes metrics) are prone to severe failure. Robust autoscalers must synthesize high-fidelity signals: request queue duration, pending token backlog, active decoding sequences, remaining unallocated KV cache blocks, and TTFT SLO error-budget burn rates.

### 6.3. Goodput and Admission Control

**Throughput** measures gross output volume. **Goodput**, by contrast, measures only the productive volume that satisfies defined service contracts (e.g., successful completions within strict latency budgets without truncation or error). Unchecked admission may temporarily inflate aggregate throughput figures while fatally degrading tail latency across all concurrent tenants.

**Admission control** dictates whether an incoming request is accepted, deferred, or rejected based on priority tiers, tenant quotas, and real-time system headroom. For example, routing high-context offline batch processing away from critical real-time conversational traffic into dedicated pools prevents batch prefill bursts from stalling interactive users.

### 6.4. Why Four Distinct Traffic Profiles Yield Divergent Performance

| Profile | Prompt / Output | Benchmark Objective |
|---|---|---|
| Short Chat | 256 / 128 tokens | Small-prompt TTFT, minimum latency |
| Long-Context RAG | 8k / 512 tokens | Heavy prefill, retrieval overhead, context windowing |
| Long Generation | 1k / 4k tokens | Extended decode iterations and prolonged KV cache residency |
| Multi-Turn Repeated | High shared prefix overlap | Prefix caching efficiency and cache isolation |

To these text profiles, multimodal workloads (images and video frames) must be added. Benchmarking requires stepping through discrete arrival rates (e.g., 1, 2, 5, 10, 20... req/s) for each profile to identify the saturation knee, p95/p99 SLO violations, and HTTP 429/503 error thresholds. Runs must be replicated across engine warmups, distinct random seeds, diverse cache hit/miss states, and runtime revisions. The MLPerf Inference benchmark suite and SGLang benchmarking documentation provide industry-standard methodologies for structured evaluation. [K42][K43]

### 6.5. PagedAttention, Prefix Caching, and Disaggregation: Architectural Solutions to Serving Bottlenecks

**PagedAttention** allocates KV cache memory in non-contiguous physical blocks (virtual memory pages), eliminating external fragmentation and sharply reducing memory waste—though it does not eliminate the physical footprint of the KV tensors themselves. **Prefix caching** skips redundant prefill computation across requests sharing identical prefixes (such as system instructions or shared few-shot examples), but is only applicable to truly shared, access-authorized prompt segments.

**Disaggregated serving** decouples the prefill and decode stages onto distinct GPU pools. This architecture is exceptionally beneficial for long-context prompts under heavy concurrency, as compute-heavy prefills no longer monopolize the memory-bandwidth-bound decoding GPUs. However, it introduces the network overhead of transmitting KV cache states across nodes. Under low concurrency, short prompts, or fabric configurations lacking ultra-high-speed inter-node interconnects, unified workers often yield superior latency and cost profiles. NVIDIA Dynamo documentation highlights this fundamental trade-off. [K40]



<svg viewBox="0 0 720 412" role="img" aria-label="Decision tree for serving. Start from which p95 or p99 SLO breaks, after tracing and profiling. If TTFT is late, ask whether the queue grows, then use admission control and warm replicas, or whether prefill dominates, then use chunked prefill and prefix caching; if still slow, A/B test separate prefill and decode pools with your own input and output lengths, and disaggregate only if the gain exceeds the KV transfer cost, otherwise stay aggregated. If TPOT is slow, it points at decode, HBM and KV pressure: use paged KV and continuous batching, or reduce bytes per token with quantization or speculative decoding. Re-check quality and p99 after every change." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="st-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="st-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="st-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker><marker id="st-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Which serving fix? Start from the SLO that breaks</text>
<rect x="250" y="40" width="220" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="360.0" y="62.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Which p95/p99 SLO breaks?</text>
<text x="360.0" y="77.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">trace + profile first</text>
<path d="M300 90 V108 H140 V122" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M420 90 V108 H560 V122" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="124" width="200" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="140.0" y="146.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">TTFT (first token late)</text>
<text x="140.0" y="161.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">queue or long prefill?</text>
<rect x="460" y="124" width="200" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="560.0" y="146.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">TPOT (stream slow)</text>
<text x="560.0" y="161.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">decode, HBM, KV pressure</text>
<path d="M90 174 V192" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M190 174 V192" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="16" y="194" width="140" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">queue grows</text>
<text x="86.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">admission control,</text>
<text x="86.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">warm replicas</text>
<rect x="164" y="194" width="150" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="239.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">prefill dominates</text>
<text x="239.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">chunked prefill,</text>
<text x="239.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">prefix caching</text>
<path d="M239 256 V274" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="150" y="276" width="178" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="239.0" y="298.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">still slow? test split pools</text>
<text x="239.0" y="313.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">A/B with your ISL/OSL</text>
<path d="M190 326 V344 H100 V354" marker-end="url(#st-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<path d="M290 326 V344 H380 V354" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="20" y="356" width="160" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="100.0" y="375.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">gain &gt; KV cost</text>
<text x="100.0" y="390.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">→ disaggregate</text>
<rect x="300" y="356" width="160" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="380.0" y="375.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">gain ≤ KV cost</text>
<text x="380.0" y="390.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">→ stay aggregated</text>
<path d="M510 174 V192" marker-end="url(#st-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<path d="M610 174 V192" marker-end="url(#st-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="470" y="194" width="112" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="526.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">batch / KV</text>
<text x="526.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">paged KV,</text>
<text x="526.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">continuous batch</text>
<rect x="590" y="194" width="114" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="647.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">bytes / token</text>
<text x="647.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">quantization,</text>
<text x="647.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">speculative</text>
<text x="704" y="290" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">re-check quality and</text>
<text x="704" y="305" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">p99 after every change</text>
</svg>



### 6.6. Why GPU Autoscaling Differs Fundamentally from CPU Autoscaling

Provisioning a new GPU serving replica involves node scheduling, container image pulling, reading tens or hundreds of gigabytes of model weights from storage into host RAM and HBM, NCCL/engine runtime initialization, KV memory pre-allocation, and CUDA kernel warmup compilation. If ingress traffic spikes within 3 seconds, spinning up cold pods on demand will inevitably violate p99 SLOs. High-availability inference requires a coordinated posture: **warm standby replicas**, local NVMe model caches, proactive predictive scaling, and intelligent admission control.

### 6.7. Streaming Robustness and Retry Policies

If a client drops connection midway through an HTTP Server-Sent Events (SSE) stream after receiving 200 tokens, a naive client retry will not seamlessly resume generation. Instead, it risks re-triggering the entire computation from scratch, double-billing token quotas, or executing non-idempotent tool calls twice. Production systems must enforce explicit contracts: deterministic **request IDs**, distributed idempotency keys, downstream cancellation propagation (halting generation immediately upon client disconnect), hierarchical timeout budgets, and partial completion reconciliations. In agentic workflows invoking external APIs, this discipline is mission-critical.

---

## 7. VLM, VQA, Vision, and Video: Where Text-Only Capacity Modeling Ends

In a pure LLM cluster, the input is an indexed sequence of discrete tokens. In **Vision-Language Models (VLMs)** and **Visual Question Answering (VQA)** pipelines, generation entails a far more complex preprocessing lifecycle: raw image or video payloads must be decoded, resized, partitioned into patches or multi-resolution crops, processed through a vision encoder, projected into multimodal embedding spaces, and fused with the language model prior to token generation.

Consider querying an intricate medical diagnostic scan or engineering blueprint. The serving bottleneck does not reside solely within the autoregressive decoder. High-resolution image decoding, CPU/GPU preprocessing transforms, patch count explosion, and vision encoder compute intensity can dwarf baseline text prefill latency, dramatically inflating TTFT.

### 7.1. Why an Image Is Never Just a Single Token

VLM architectures partition visual inputs into localized patches; many employ dynamic resolution, multi-crop slicing, or image tiling to preserve detail. For example, dividing a 1024×1024 image into 16×16 patches in a **simplified pedagogical scheme** yields 64×64 = 4,096 patches. However, production architectures frequently integrate patch merging, spatial pooling, adaptive downsampling, and projection layers; hence, patch count does not map 1:1 to LLM sequence tokens.

**Architectural takeaway:** A specification stating "the model supports an 8k context window" does not mean an engineer can pass 8k text tokens alongside unconstrained high-resolution imagery. System designers must account for the model's multimodal preprocessor, max image resolution bounds, visual token allocation quotas, and cross-attention architectures. Hugging Face documentation similarly delineates the distinct role of multimodal `Processor` pipelines compared to standard text tokenizers. [K44]

### 7.2. VLM Request Lifecycle Waterfall



<svg viewBox="0 0 720 246" role="img" aria-label="VLM request waterfall: media input is decoded, sampled into frames, crops or tiles, run through a ViT vision encoder over patches, projected into the language model&#x27;s space, then the language model prefills text plus visual tokens and finally decodes the answer. Everything before decode lands in TTFT. Image example: 1024 by 1024 pixels with 16 by 16 patches gives 64 by 64, or 4,096 patches in a one-plane toy model; merging and pooling change the actual LLM token count. Video example: a 10-second clip is 20 frames at 2 fps but 300 frames at 30 fps." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="vl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="vl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="vl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">VLM request waterfall: work before the first text token</text>
<rect x="16" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="70.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Media input</text>
<text x="70.0" y="76.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">image / video</text>
<text x="70.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">decode, EXIF</text>
<rect x="132" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="186.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Sampling</text>
<text x="186.0" y="76.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">frames, crops,</text>
<text x="186.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tiles</text>
<line x1="124" y1="71" x2="131" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="248" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="302.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Vision encoder</text>
<text x="302.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">ViT over patches</text>
<line x1="240" y1="71" x2="247" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="418.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Projector</text>
<text x="418.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">vision → LM space</text>
<line x1="356" y1="71" x2="363" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="480" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="534.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LM prefill</text>
<text x="534.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">text + visual tokens</text>
<line x1="472" y1="71" x2="479" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="596" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="650.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LM decode</text>
<text x="650.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">answer tokens</text>
<line x1="588" y1="71" x2="595" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M16 112 H592" marker-start="url(#vl-arr)" marker-end="url(#vl-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="306.0" y="128" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">all of this lands in TTFT</text>
<rect x="16" y="142" width="340" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Image: tokens are not “1 per image”</text>
<text x="28" y="182" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">1024 × 1024 px ÷ 16 × 16 patch</text>
<text x="28" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">= 64 × 64 = 4,096 patches (one-plane toy model)</text>
<text x="28" y="220" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">merging/pooling changes the LLM token count</text>
<rect x="364" y="142" width="340" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Video: “10 s clip” is not a load spec</text>
<text x="376" y="189" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">2 fps</text>
<rect x="430" y="179" width="12" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="448.0" y="190" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">20 frames</text>
<text x="376" y="215" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">30 fps</text>
<rect x="430" y="205" width="180" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="616.0" y="216" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">300 frames</text>
</svg>



This execution flow is not identical across all VLM families; fusion mechanisms, cross-attention layers, and prefix token embeddings vary considerably across architectures. Nonetheless, it highlights the essential latency components required for accurate capacity sizing.

### 7.3. Novel Operational Metrics for Multimodal Serving

- **Pixel density**, patch/crop count, and equivalent visual token counts per image.
- For video: **FPS, sampled frame count, total video duration**, and video decode/IO overhead.
- Image preprocessor CPU cycle consumption, hardware-accelerated image decoders (e.g., nvJPEG), and vision encoder inference latency.
- LLM prefill/decode latencies, KV cache allocations, and aggregate memory footprint per request.
- Serving fidelity: **VQA accuracy**, OCR fidelity, visual hallucination benchmarks, and task-specific evaluation metrics.

Specifying a workload as "a 10-second video clip" is insufficient for system capacity planning; extracting 20 frames at 2 FPS versus processing 300 frames at 30 FPS introduces an order-of-magnitude divergence in compute and memory demands. Furthermore, the frame rates and temporal context windows on which the vision encoder was pretrained dictate both processing efficiency and model fidelity. [K44]

### 7.4. When Multimodal Quantization Delivers Value

Post-training quantization (PTQ) from BF16 to INT8/INT4 can compress LLM decoder weight footprints. However, the vision encoder, multimodal projection adapters, normalization layers, attention kernels, and KV caches often operate across disparate data types. Labeling a deployment "4-bit" does not mean the entire multimodal pipeline executes in uniform 4-bit precision.

**Key benchmarks to evaluate:** Weight allocation per GPU; peak HBM footprint per image request; end-to-end TTFT; output TPOT; vision encoder execution time; VQA task degradation; and numerical stability across diverse input resolutions. During quantization validation, running identical queries under both text-only and image+text conditions isolates speedups to determine exactly which pipeline stage benefits.

### 7.5. Security, Privacy, and Data Governance

Image files frequently harbor sensitive EXIF metadata; ingested enterprise documents and scans routinely contain personally identifiable information (PII). Retention policies, extracted OCR text, intermediate visual embeddings, and diagnostic debug logs must adhere to identical data governance and classification tiers. High-fidelity VLM inference output never excuses a failure to enforce source file licensing, authorization, and data privacy controls.

---




---

**PART III — How Multiple GPUs Cooperate: Interconnects, Topologies, and Collectives**

*Why can thousands of GPUs stall even when every individual chip performs flawlessly? We dissect the physical and algorithmic mechanics of distributed communication.*


## 8. From a Single H100 to an 8-GPU DGX Server

Having analyzed intra-GPU compute execution and memory hierarchy dynamics, we now scale out the model.

A 70-billion-parameter model in BF16 precision consumes approximately 140 GB of storage for its weights alone. These weights cannot physically fit onto a single H100 SXM GPU equipped with 80 GB of HBM. Engineering teams face an architectural choice: compress the model into a narrower numerical precision or shard its parameters across multiple GPUs.

The moment parameter sharding is adopted, a fundamental systems challenge emerges: **How frequently, and at what bandwidth, must these GPUs communicate with one another?**

To execute a model exceeding single-GPU HBM capacity, weights must be partitioned across devices. From this point forward, matrix arithmetic becomes inextricably linked to **inter-GPU data movement**.

### 8.1. PCIe, NVLink, and NVSwitch: Three Distinct Architectural Roles

These three interconnect technologies are frequently mentioned together, yet they represent entirely distinct layers of hardware architecture. Conceptualizing a server chassis as a facility, PCIe functions as the general corridor network connecting diverse rooms and peripherals; NVLink serves as the dedicated high-speed expressway between GPUs; and NVSwitch acts as the multi-lane interchange orchestrating high-bandwidth traffic across all connected GPUs simultaneously.

**PCI Express (PCIe)** is an industry-standard peripheral interconnect; host CPUs, accelerators, NICs, and NVMe drives all interface via the server's PCIe topology. A PCIe Gen5 ×16 link delivers a theoretical unidirectional bandwidth of roughly 63 GB/s; real-world throughput is somewhat lower due to line encoding (128b/130b) and protocol packet overhead.

**NVLink** is NVIDIA's proprietary high-bandwidth, direct GPU-to-GPU interconnect technology. An H100 SXM GPU provides an aggregate NVLink 4 bandwidth of **450 GB/s unidirectional** and **900 GB/s bidirectional**. These two figures are not mutually exclusive alternatives; they represent the **identical physical link bundle expressed under different directional definitions**. [K2]

**NVSwitch** is the switching fabric ASIC that routes NVLink traffic across multiple GPUs at full line rate. Inside an 8-GPU DGX H100 server, all eight accelerators communicate concurrently across an on-board NVSwitch fabric.

The wiring detail matters. Each H100 spreads its 18 NVLinks over the four NVSwitches as 5 + 4 + 4 + 5, so any pair of GPUs talks at the same rate, and the cut separating four GPUs from the other four carries $4\times450\ \mathrm{GB/s}\times2=3.6\ \mathrm{TB/s}$ (the bisection bandwidth). The NVLink fabric does not come up on its own: on NVSwitch systems the GPU-to-GPU routes are only established once the **Fabric Manager** service is running [K53]. With Blackwell the domain leaves the chassis: a GB200/GB300 NVL72 rack joins 72 GPUs through 18 NVSwitch chips into a single NVLink domain [K51].



<svg viewBox="0 0 720 350" role="img" aria-label="Logical topology of a DGX H100. Four NVSwitch chips sit on top, and each of the eight GPUs, 80 GB each, connects to all four with 18 NVLinks of 25 GB/s, 450 GB/s per direction per GPU. Below, a PCIe Gen5 tree links each GPU to its own ConnectX-7 NIC at 400 Gb/s, or 50 GB/s per direction, into the compute fabric. CPU 0 serves GPUs and NICs 0 to 3 as NUMA domain 0, CPU 1 serves 4 to 7 as NUMA domain 1, and the CPUs are linked by UPI. The node also has two dual-port ConnectX-7 for the storage fabric, eight 3.84 TB NVMe drives, 2 TB of RAM and draws up to 10.2 kW." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dx-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dx-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dx-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">DGX H100: logical topology</text>
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
<text x="704" y="24" text-anchor="end" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">each GPU: 18 NVLink × 25 GB/s = 450 GB/s/dir</text>
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
<text x="51" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">56 cores</text>
<rect x="634" y="154" width="70" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="669" y="175" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="669" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">56 cores</text>
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
<text x="358.0" y="278" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">compute fabric: 8 × ConnectX-7 · 400 Gb/s (50 GB/s/dir) each — one NIC per GPU</text>
<rect x="92" y="148" width="268" height="100" rx="8" style="fill:none;stroke:var(--c-warn);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="356" y="148" width="268" height="100" rx="8" style="fill:none;stroke:var(--c-warn);stroke-width:1;stroke-dasharray:4 3"/>
<text x="16" y="222" text-anchor="start" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">PCIe Gen5</text>
<text x="16" y="238" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:600">NUMA 0</text>
<text x="16" y="254" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">UPI ⇄ CPU 1</text>
<text x="704" y="222" text-anchor="end" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">PCIe Gen5</text>
<text x="704" y="238" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:600">NUMA 1</text>
<rect x="16" y="292" width="688" height="46" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="311" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Also on the node: 2 × dual-port ConnectX-7 → storage fabric · 8 × 3.84 TB NVMe · 2 TB RAM · up to 10.2 kW</text>
<text x="28" y="328" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">A GPU talks to GPUs over NVSwitch, to other servers through its own NIC (GPUDirect RDMA), and to the host over PCIe.</text>
</svg>



While this diagram is not an exhaustive schematic of every physical PCIe switch and clock domain, it clearly illustrates why each distinct interconnect path exists.

### 8.2. What Are NICs and GPUDirect RDMA?

A **NIC (Network Interface Card)** connects the server chassis to the external network fabric. **RDMA (Remote Direct Memory Access)** is a hardware-offloaded networking paradigm enabling direct memory transfers between networked hosts without traversing the host CPU operating system stack or copying data through intermediate system buffers. **GPUDirect RDMA** establishes direct DMA pipelines between compatible network adapters and GPU HBM over the PCIe bus.

**Do not conflate GPUDirect RDMA with GPUDirect Storage (GDS):** GPUDirect RDMA governs the NIC $\leftrightarrow$ GPU HBM communication path, whereas GPUDirect Storage optimizes the NVMe storage $\leftrightarrow$ GPU HBM I/O path. While modern distributed storage architectures frequently leverage RDMA transports, the two technologies address fundamentally distinct subsystems. [K6]

### 8.3. NUMA and the Puzzle of "Why Is It Slow Within the Same Server?"

Hardware components co-located within the same server chassis do not share uniform interconnect distances. In dual-socket enterprise servers, specific memory pools and PCIe expansion complexes are physically attached to a particular CPU socket. Assuming that data transfers will be instantaneous simply because devices reside in the same chassis overlooks the physical interconnect path.

**NUMA (Non-Uniform Memory Access)** dictates that memory access latencies and bandwidths vary depending on which CPU socket an execution thread runs on and which local or remote memory controller manages the physical RAM address. A high-speed NIC or GPU PCIe root complex connects directly to one host socket. Misaligned CPU core affinities, unpinned host memory allocations, and improper NUMA socket mapping force cross-socket interconnect traversal (e.g., UPI/QPI), inducing severe latency penalties.

In a DGX platform, dual host CPUs do not merely serve as administrative orchestrators: data ingestion pipelines, page-locked memory allocations, NIC driver interrupts, distributed storage streaming, and orchestration runtimes are all directly constrained by the underlying CPU–PCIe–NUMA topology.

### 8.4. Why Numbers Dictate Architecture

<svg viewBox="0 0 720 286" role="img" aria-label="Log-scale bandwidth bars per direction. HBM3 peaks at about 3,350 GB/s for a GPU reading its own weights and activations. NVLink gives 450 GB/s per direction per GPU inside the NVLink domain, 7.4 times less than HBM. PCIe 5.0 x16 gives about 63 GB/s per direction in theory, for the host and PCIe devices, 53 times less. An NDR 400 Gb/s port gives 50 GB/s per direction line rate for server-to-server traffic, 67 times less than HBM." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="bl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">The bandwidth cliff: each hop away from HBM is several times narrower</text>
<text x="16" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">HBM3</text>
<text x="16" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">peak</text>
<rect x="150" y="48" width="383.579" height="24" rx="3" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="541.5790576486218" y="65" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">3,350 GB/s</text>
<text x="150" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU reads its own weights and activations</text>
<text x="16" y="116" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NVLink</text>
<text x="16" y="132" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">per dir / GPU</text>
<rect x="150" y="102" width="251.139" height="24" rx="3" style="fill:var(--c-success);fill-opacity:.6"/>
<text x="409.1391862669635" y="119" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">450 GB/s</text>
<text x="150" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU ↔ GPU inside the NVLink domain</text>
<text x="704" y="119" text-anchor="end" style="fill:var(--c-success);font-size:12px;font-weight:700">÷7.4 vs HBM</text>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">PCIe 5.0 ×16</text>
<text x="16" y="186" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">per dir, theoretical</text>
<rect x="150" y="156" width="121.428" height="24" rx="3" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="279.42766490529937" y="173" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">63 GB/s</text>
<text x="150" y="196" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">host and PCIe devices</text>
<text x="704" y="173" text-anchor="end" style="fill:var(--c-warn);font-size:12px;font-weight:700">÷53 vs HBM</text>
<text x="16" y="224" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NDR 400 Gb/s</text>
<text x="16" y="240" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">per dir, line rate</text>
<rect x="150" y="210" width="106.18" height="24" rx="3" style="fill:var(--c-danger);fill-opacity:.6"/>
<text x="264.18039523127993" y="227" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">50 GB/s</text>
<text x="150" y="250" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">server ↔ server, GPU–NIC traffic</text>
<text x="704" y="227" text-anchor="end" style="fill:var(--c-danger);font-size:12px;font-weight:700">÷67 vs HBM</text>
<line x1="150" y1="258" x2="560" y2="258" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="150" y1="258" x2="150" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="150.0" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10</text>
<line x1="301.91" y1="258" x2="301.91" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="301.9098023843601" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100</text>
<line x1="453.82" y1="258" x2="453.82" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="453.8196047687201" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1,000</text>
<text x="560" y="276" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">GB/s (log)</text>
</svg>

This figure illustrates a **hierarchy of physical interconnect bandwidths**. Raw figures for HBM, NVLink, PCIe, and network NICs cannot be translated directly into end-to-end execution times without modeling transfer volumes and overheads. For example, if a server chassis houses eight independent NICs, the chassis-level aggregate egress bandwidth must not be conflated with the network bandwidth available to an individual GPU port.

**Intuitive analogy:** Fetching ingredients from your workstation counter, walking to an adjoining prep kitchen, and ordering supplies from a distribution center in another city incur radically different time and energy costs. When partitioning deep learning models across GPUs, the frequency of tensor synchronization must align strictly with the bandwidth and latency of the underlying physical interconnect.

Let us turn the ladder into time. Suppose one GPU moves the 140 GB of BF16 weights of a 70B model once over each path, $t=140\ \mathrm{GB}/W$: about **42 ms** from HBM, **0.31 s** over NVLink, **2.2 s** over PCIe, **2.8 s** over a single NDR port and **5.6 s** from a 25 GB/s shared storage system. The same arithmetic bounds decode speed. At batch 1, decode reads every weight from HBM for every token. With TP = 8 the weights are split over eight GPUs, so each token needs at least $140/(8\times3.35)\approx5.2$ ms: a ceiling of **~190 tokens per second** before the KV cache or any other read is even counted.

---

## 9. From DGX to SuperPOD: Why Network Topology Is Far More Than "Cable Counts"

We have unified eight GPUs within a single chassis into a coherent high-speed NVLink domain. Now envision a cluster of 128 such servers. While the GPUs inside each node crunch local tensors, they must periodically exchange intermediate activations, gradients, and attention states with GPUs residing in remote chassis across the data center.

At this scale, networking ceases to be a matter of plugging in cables. Which nodes communicate with which peers, what traffic flows contend for the same shared links, and what jobs stall when an intermediate switch buffer saturates? The **network fabric**—the high-performance computing interconnect—provides the physical answer to these architectural questions.

### 9.1. Disentangling Scale-Up vs. Scale-Out

**Scale-up** refers to expanding capacity within a unified, tightly coupled high-speed shared memory domain (e.g., adding GPUs across a shared NVLink crossbar). **Scale-out** refers to expanding capacity horizontally by interconnecting autonomous server nodes over an external networking fabric.

For standard H100 systems, the common scale-up boundary is the 8-GPU NVSwitch domain within a single DGX chassis. Exceeding this boundary requires inter-node communication across the cluster's **compute fabric**. However, the dogmatic claim that "Tensor Parallelism (TP) can never cross the 8-GPU boundary" is **factually incorrect**: TP degrees greater than 8 are entirely feasible; they simply incur the higher latency and reduced bandwidth of inter-node networking. NVIDIA Megatron documentation natively supports TP configurations spanning multiple nodes in conjunction with other parallelism strategies. [K3]

### 9.2. Leaf and Spine: Mapping Network Hops

Consider a metropolitan road network. Navigating between two adjacent streets in your immediate neighborhood does not require entering an interstate highway. However, traveling across town demands merging onto major arterial routes. A leaf–spine topology functions on this exact principle: host servers connect directly to an access tier of leaf switches, while inter-leaf communications traverse a non-blocking core tier of spine switches.

A **leaf switch** (top-of-rack or access switch) terminates physical links from compute nodes and NICs. A **spine switch** forms the core backbone tier, cross-connecting the entire array of leaf switches.



<svg viewBox="0 0 720 340" role="img" aria-label="Compute fabric of a DGX H100 SuperPOD. Sixteen spine switches sit on top. Below are four scalable units, each with eight leaf switches numbered 0 to 7, one per rail, and 32 DGX H100 servers. Leaf r in an SU connects NIC r of all 32 servers, so same-rail traffic inside an SU takes one leaf hop. Every leaf links to every spine with two 400 Gb/s cables, 32 ports up and 32 down, so the fabric is 1 to 1 by design. A highlighted path shows traffic between different SUs going leaf, spine, leaf. Four SUs give 128 slots; one hosts UFM, leaving 127 DGX or 1,016 GPUs. Storage runs on a separate network." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ft-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ft-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="ft-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">DGX H100 SuperPOD compute fabric: 32 leaf + 16 spine, rail-optimized</text>
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
<text x="704" y="104" text-anchor="end" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">different SU → leaf → spine → leaf</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Leaf r in an SU connects NIC r of all 32 DGX (rail r) → same rail inside an SU = one leaf hop.</text>
<text x="16" y="310" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Each leaf: 32 ports down + 32 up (2 × 400 Gb/s to each of 16 spines) → 1:1, non-blocking by design.</text>
<text x="16" y="328" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">4 SU × 32 slots = 128; one slot hosts UFM → 127 DGX = 1,016 GPUs. Storage fabric is a separate network.</text>
</svg>



Two servers homed to the identical leaf switch communicate with minimal switch latency. Communication between nodes under disparate leaf switches follows a **leaf $\rightarrow$ spine $\rightarrow$ leaf** traversal.

A **hop** denotes each traversal through an intermediate network switch ASIC. **Fat-tree / Clos topologies** represent a class of multi-stage switching architectures engineered to deliver non-oversubscribed, parallel cross-sectional bandwidth across large numbers of endpoints.

**Bisection bandwidth** is the aggregate throughput capacity between two equal halves of a bisected network under worst-case communication partitions. A **non-blocking** or **full-bisection bandwidth** fabric guarantees that arbitrary all-to-all permutation traffic is never bottlenecked by uplinks under balanced routing. In real-world deployments, however, packet hashing collisions, queue buffer overflow, and transient microburst congestion can still introduce tail latency.

### 9.3. How Rail Optimization Works

To make the network topology concrete, consider an enterprise server containing eight GPUs paired with eight discrete network adapters. If these adapters are connected arbitrarily to switches, collective traffic from identical GPU ranks will collide unpredictably across uplinks. Rail optimization organizes network links into structured parallel paths, ensuring that predictable collective communication patterns traverse the fabric with minimal contention.

Assume every node features eight NICs mapped to eight GPUs. By routing all NICs of index 0 across all nodes to a common dedicated leaf switch, we establish a logical "rail."

<svg viewBox="0 0 720 342" role="img" aria-label="Rail-optimized wiring. Three servers, A, B and C, each have eight NICs numbered 0 to 7, one per GPU. NIC i of every server is cabled to leaf switch i, so leaf 0 is rail 0, leaf 1 is rail 1, and so on; a spine layer above connects the leaves. Highlighted: GPU 0 of server A reaches GPU 0 of server B through NIC 0 and leaf 0 in a single leaf hop within one SU. Reaching GPU 1 of server C from GPU 0 of A crosses rails: either up to the spine, or first over NVLink to GPU 1 of A and then over rail 1, which NCCL calls PXN." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="rl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Rail optimization: NIC i of every server meets on leaf i</text>
<rect x="96" y="40" width="600" height="24" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="396.0" y="56.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">spine layer (cross-rail / cross-SU traffic)</text>
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
<text x="24" y="172" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">server A</text>
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
<text x="24" y="218" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">server B</text>
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
<text x="24" y="264" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">server C</text>
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
<text x="16" y="312" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">A·GPU0 → NIC0 → leaf 0 → NIC0 → B·GPU0: same rail, one leaf hop (inside one SU).</text>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">A·GPU0 → C·GPU1: different rail → up to the spine, or hop over NVLink to A·GPU1 first and use rail 1 (NCCL PXN).</text>
</svg>

Under this layout, intra-rail collective operations within the same Scalable Unit (SU) can execute across a single leaf switch stage. However, traffic flowing between different Scalable Units or across disparate rails must escalate to the spine switching tier. The assertion that **all rail-optimized traffic is single-hop** is inaccurate. NVIDIA DGX H100 reference designs explicitly distinguish between intra-SU rail traffic and inter-SU spine traversals. [K4]

### 9.4. How Many Independent Networks Does an AI Supercomputer Require?

An enterprise-scale AI supercomputer does not rely on a monolithic network. In the NVIDIA DGX H100 SuperPOD architecture, distinct communication planes are physically segregated into dedicated fabrics:

<svg viewBox="0 0 720 304" role="img" aria-label="One DGX H100 attaches to four separate networks. The compute fabric, eight ConnectX-7 at 400 Gb/s InfiniBand, carries GPU collectives and distributed training, kept apart so AllReduce and AllToAll do not fight storage. The storage fabric, two dual-port ConnectX-7, carries datasets and checkpoints, so bursty high-volume I/O is planned separately. In-band management on 100 GbE carries the operating system, services and scheduler traffic. Out-of-band management on 1 GbE reaches the BMC for power cycling, console and fault response, and works even if the operating system is dead." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="fn-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">One DGX, four separate networks (NVIDIA DGX H100 SuperPOD reference)</text>
<rect x="16" y="44" width="120" height="228" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="76" y="150" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">DGX H100</text>
<text x="76" y="168" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPU</text>
<line x1="136" y1="69" x2="168" y2="69" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="44" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="182" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Compute fabric</text>
<text x="182" y="82" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × CX-7 · 400 Gb/s IB</text>
<text x="380" y="64" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU collectives, distributed training</text>
<text x="380" y="82" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">AllReduce/AllToAll should not fight storage</text>
<line x1="136" y1="127" x2="168" y2="127" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="102" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="182" y="122" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Storage fabric</text>
<text x="182" y="140" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">2 × dual-port CX-7</text>
<text x="380" y="122" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">datasets, checkpoints</text>
<text x="380" y="140" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">bursty high-volume I/O is planned apart</text>
<line x1="136" y1="185" x2="168" y2="185" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="160" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="182" y="180" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">In-band management</text>
<text x="182" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100 GbE</text>
<text x="380" y="180" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">OS, services, scheduler</text>
<text x="380" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">day-to-day ops traffic kept apart</text>
<line x1="136" y1="243" x2="168" y2="243" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="218" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="182" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Out-of-band (BMC)</text>
<text x="182" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1 GbE</text>
<text x="380" y="238" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">power cycle, console, fault response</text>
<text x="380" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">works even if the OS is dead</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BMC = baseboard management controller; UFM manages and monitors the InfiniBand fabric; a scheduler such as Slurm places jobs.</text>
</svg>

In the reference design each of these networks has its own switches: 48 Quantum-2 QM9700 on the compute fabric (32 leaf + 16 spine), 16 QM9700 on the storage fabric, 8 SN4600C for in-band management and 8 SN2201 for out-of-band management [K4]. A single QM9700 offers 64 ports of 400 Gb/s [K54].

A **Baseboard Management Controller (BMC)** is an independent micro-controller providing hardware-level out-of-band management. **Unified Fabric Manager (UFM)** is NVIDIA's platform software for provisioning, monitoring, and dynamically optimizing InfiniBand fabrics. A **workload scheduler**, such as Slurm, orchestrates multi-node job queues and binds tasks to physical topology.

Beyond networking, automated container orchestration, GPU driver runtimes, NCCL configurations, telemetry pipelines, power distribution, and thermal cooling are integral to the system. In reference architectures where an individual node draws on the order of 10 kW, electrical and cooling provisioning are far from mundane facilities details: thermal throttling or voltage sags directly induce GPU frequency downclocking, straggler ranks, and cluster-wide training stalls. [K4]

### 9.5. SuperPOD by the Numbers: A Reference Architecture

For a baseline NVIDIA DGX H100 SuperPOD reference deployment:

- **1 Scalable Unit (SU)**: A modular building block comprising 32 DGX nodes.
- **4 SUs**: 128 server rack positions in the physical infrastructure design.
- **Full deployment**: Due to dedicated UFM appliance integration, the cluster consists of **127 compute DGX nodes (1,016 H100 GPUs)**.
- **Compute fabric**: **32 leaf switches + 16 spine switches = 48 InfiniBand switches**.
- **Storage fabric**: Segregated entirely onto an independent InfiniBand or high-speed Ethernet switching tier.
- **Port count per leaf**: 32 ports down (NIC $i$ of each DGX in the SU) + 32 ports up (2 cables to each of 16 spines) = 64 ports. The spine tier carries $16\times64\times50\ \mathrm{GB/s}=51.2\ \mathrm{TB/s}$, more than the $1016\times50=50.8\ \mathrm{TB/s}$ all GPUs can inject at once, so the design is non-blocking.
- **Beyond two tiers**: leaf–spine stretches to 8 SUs (64 leaf × 32 = 2,048 GPUs); from 16 SUs a third **core** tier is added. A 2,048-DGX build (16,384 GPUs) uses 512 leaf, 512 spine and 256 core switches, and the longest path grows from 3 to 5 switches [K55].

These metrics do not represent an immutable universal AI standard; they represent the precise engineering parameters of the documented reference architecture. [K4]

### 9.6. Ethernet vs. InfiniBand

A perennial industry question arises: "Must an enterprise mandate InfiniBand to train frontier-scale foundational models?" Selecting a fabric technology is never simply a choice between vendor brand names or physical cabling. Architecture teams must evaluate collective communication efficiency, telemetry transparency, congestion management, failover dynamics, and mature software runtime integration.

**InfiniBand** is a purpose-built, credit-based flow control networking architecture developed specifically for high-performance computing (HPC). **RoCE (RDMA over Converged Ethernet)** provides RDMA capabilities over standard enterprise Ethernet infrastructures using Priority-based Flow Control (PFC) and Explicit Congestion Notification (ECN).

The dogma that "frontier LLMs can only be trained over InfiniBand" is demonstrably false. Meta's production training cluster for Llama 3 405B was famously deployed over a customized RoCEv2 Ethernet fabric. What matters is not the high-level marketing label of the physical wire, but the protocol's RDMA fidelity, congestion control dynamics, collective offload efficiency, fabric bisection topology, error recovery semantics, and empirically validated end-to-end goodput. [K5]

---

### 9.7. The Fourth Physical Wall: Power, Thermal Cooling, and Facility Headroom

In previous sections, we modeled interconnect bandwidth and latency. But what happens if the data center lacks the **electrical power and thermal heat rejection capacity** required to energize 1,024 H100 GPUs simultaneously? The theoretical TFLOP/s figures printed on hardware spec sheets remain pure fiction if thermal dissipation limits prevent hardware from running at full clock frequencies.

The documented **maximum system power draw of a DGX H100/H200 node is approximately 10.2 kW**. In our **pedagogical reference scenario** housing 1,024 GPUs across 128 eight-GPU DGX chassis:

$$
P_{\mathrm{DGX,max}}=128\times10.2\ \mathrm{kW}
=1305.6\ \mathrm{kW}\approx1.31\ \mathrm{MW}
$$

This figure represents solely the **sum of the nameplate maximum power ratings** for the 128 compute chassis; it is not a direct measurement of steady-state workload consumption. Furthermore, it completely excludes spine switches, storage arrays, optical transceivers, transformer losses, and mechanical cooling chillers. Nor should one assume that all nodes operate continuously at peak nameplate draw. [K26]

**PUE (Power Usage Effectiveness)** quantifies the ratio of total facility power consumption to the power delivered to productive IT hardware:

$$
\mathrm{PUE}=\frac{E_{\mathrm{facility}}}{E_{\mathrm{IT}}}
$$

For instance, with an IT load of 1 MW and a PUE of 1.3, the aggregate electrical utility demand across the facility is approximately 1.3 MW. This is an illustrative PUE baseline, not a universal measurement of any specific data center.

Another critical architectural parameter is **rack density**: the concentration of electrical power and heat dissipation within a single server rack footprint (kW/rack). As GPU clusters concentrate 40 kW to 100+ kW into individual racks, conventional forced-air cooling reaches physical airflow velocity limits, necessitating direct-to-chip liquid cooling or rear-door heat exchangers (RDHx). The Uptime Institute emphasizes that extreme-density AI infrastructures demand cooling and power delivery paradigms radically distinct from traditional enterprise IT. [K27]

**Power capping** enforces deterministic electrical consumption ceilings via server firmware or operating system drivers (e.g., `nvidia-smi -pl`). While power capping protects facility substation thresholds and stabilizes thermal excursions during peak workloads, active throttling can depress clock speeds and distort Model Flops Utilization (MFU) benchmarks.

**Facility checklist:** GPU and host power envelopes, PDU/UPS redundancy topologies ($N+1$ or $2N$), catastrophic outage failover scenarios, rack-level power density limits, chilled-water vs. air cooling infrastructure, optical transceiver thermal dissipation, and physical expansion floor space must be co-designed. Electrical and civil engineering reviews must precede—never follow—GPU procurement approvals.

---

## 10. Collective Communication: Why GPUs Wait for One Another

Imagine four analysts collaborating on a unified quarterly financial report. Each analyst computes their individual section at breakneck speed. At the end of the report, however, a critical aggregate balance figure must be known identically by all four analysts. No team member can proceed to draft the subsequent section until that combined total is verified and broadcast across the table.

Distributed AI training features dozens of identical synchronization barriers in every training step. Even if individual GPUs crunch matrix math in microseconds, they must routinely halt execution to synchronize tensors. **Collective communication** is the formal discipline governing these multi-process data exchange operations.

In deep learning systems, GPUs must synchronize local weight gradients and activation slices across ranks. These synchronization events are orchestrated as **collective communication** primitives.

### 10.1. Four Fundamental Collective Primitives

Consider a minimal 4-GPU cluster, where each device holds a scalar value:

<svg viewBox="0 0 720 428" role="img" aria-label="Four panels on four GPUs. AllReduce with sum: GPUs start with 10, 20, 30 and 40 and all end with 100. AllGather: each GPU starts with one value and all end with all four pieces. ReduceScatter with sum: each GPU starts with four slices, and GPU i ends with the sum of slice i across all GPUs. AllToAll: each GPU starts with four chunks, and chunk j of GPU i ends up on GPU j, like transposing a matrix. AllReduce equals ReduceScatter followed by AllGather." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="cl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Four collectives on four GPUs (before → after)</text>
<rect x="16" y="40" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllReduce(sum)</text>
<text x="110" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="178" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="246" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="314" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="28" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">before</text>
<text x="28" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">after</text>
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
<text x="28" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">everyone ends with the total</text>
<rect x="364" y="40" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllGather</text>
<text x="458" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="526" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="594" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="662" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="376" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">before</text>
<text x="376" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">after</text>
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
<text x="376" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">everyone ends with every piece</text>
<rect x="16" y="224" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="244" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">ReduceScatter(sum)</text>
<text x="110" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="178" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="246" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="314" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="28" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">before</text>
<text x="28" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">after</text>
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
<text x="28" y="384" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU i keeps the sum of slice i</text>
<rect x="364" y="224" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="244" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllToAll</text>
<text x="458" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="526" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="594" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="662" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="376" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">before</text>
<text x="376" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">after</text>
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
<text x="376" y="384" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">chunk j of GPU i goes to GPU j</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Colour = source GPU · digit = slice index · Σi = sum of slice i over all GPUs. AllReduce = ReduceScatter + AllGather.</text>
</svg>

| Primitive | Output | Operational Intuition |
|---|---|---|
| **AllReduce (sum)** | Each GPU: 100 | Every rank contributes a value; every rank receives the globally reduced sum |
| **AllGather** | Each GPU: [10, 20, 30, 40] | Every rank contributes a local slice; all ranks assemble the concatenated global array |
| **ReduceScatter (sum)** | Slices of the globally reduced sum are distributed across GPUs | Reduce values globally, then shard the reduced result across ranks |
| **AllToAll** | Each GPU sends distinct personalized chunks to all other GPUs | Full crossbar shuffle: each department mails specialized data directly to every other department |

### 10.2. Ring AllReduce: Step by Step

To synchronize parameters across all GPUs, the naive solution would gather all data onto a single coordinator GPU, perform reduction, and broadcast the result back. At scale across hundreds or thousands of GPUs, this centralized approach instantly collapses under severe bandwidth bottlenecks. Ring AllReduce distributes this workload across a logical ring topology, transforming every participating rank into both a sender and an aggregator.

The elegance of the algorithm lies in subdividing local tensors into equal chunks, enabling every GPU to transmit and receive data simultaneously. The trade-off is that each data chunk traverses multiple sequential hops.

The ring algorithm pipelines data around the ring in two distinct phases: first, a *reduce-scatter* phase, followed by an *all-gather* phase.

<svg viewBox="0 0 720 342" role="img" aria-label="Ring AllReduce with four GPUs connected in a ring, GPU 0 to 1 to 2 to 3 and back to 0. The data is cut into four chunks of size S over N. In the first three steps, reduce-scatter, each GPU sends one chunk to its neighbour, which adds its own part, so partial sums travel around the ring; afterwards each GPU holds one finished sum. In the next three steps, all-gather, the finished sums are passed around until every GPU has all of them. Each GPU sends 2 times N minus 1 over N times S bytes; for N equal 8 and S equal 8 GB that is 14 GB per GPU. Time is roughly 2 times N minus 1 latency steps plus that volume divided by bandwidth." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="rg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Ring AllReduce on N = 4: 2(N−1) = 6 steps, one chunk of S/N per step</text>
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
<text x="120" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">ring</text>
<text x="297.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">step 1</text>
<text x="367.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">step 2</text>
<text x="437.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">step 3</text>
<text x="507.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">step 4</text>
<text x="577.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">step 5</text>
<text x="647.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">step 6</text>
<text x="254" y="93" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU0</text>
<rect x="266" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +0</text>
<rect x="336" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +3</text>
<rect x="406" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +2</text>
<rect x="476" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ1</text>
<rect x="546" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ0</text>
<rect x="616" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ3</text>
<text x="254" y="133" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU1</text>
<rect x="266" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +1</text>
<rect x="336" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +0</text>
<rect x="406" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +3</text>
<rect x="476" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ2</text>
<rect x="546" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ1</text>
<rect x="616" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ0</text>
<text x="254" y="173" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU2</text>
<rect x="266" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +2</text>
<rect x="336" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +1</text>
<rect x="406" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +0</text>
<rect x="476" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ3</text>
<rect x="546" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ2</text>
<rect x="616" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ1</text>
<text x="254" y="213" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU3</text>
<rect x="266" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +3</text>
<rect x="336" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +2</text>
<rect x="406" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send +1</text>
<rect x="476" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ0</text>
<rect x="546" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ3</text>
<rect x="616" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">send Σ2</text>
<line x1="266" y1="238" x2="468" y2="238" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="476" y1="238" x2="678" y2="238" style="stroke:var(--c-accent-2);stroke-width:2"/>
<text x="367.0" y="254" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:600">N−1 steps: partial sums travel</text>
<text x="577.0" y="254" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">N−1 steps: finished sums spread</text>
<rect x="16" y="278" width="688" height="52" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="298" text-anchor="start" style="fill:var(--c-text);font-size:12px">Each GPU sends 2(N−1)/N · S bytes. N = 8, S = 8 GB → 2 · 7/8 · 8 = 14 GB per GPU (not total traffic).</text>
<text x="28" y="318" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Time ≈ 2(N−1)·L + 2(N−1)/N · S/W: latency grows with N, the bandwidth term barely does.</text>
</svg>

Given an initial local tensor size of $S$ bytes per GPU and a cluster of $N$ GPUs, the **data volume transmitted per GPU** in an idealized ring implementation is:

$$
V_{\mathrm{ring}}=2\frac{N-1}{N}S
$$

For example, with $N = 8$ and $S = 8\text{ GB}$:

$$
V_{\mathrm{ring}}=2\cdot\frac78\cdot8=14\ \mathrm{GB/GPU}
$$

This equation calculates the bytes transmitted by an individual GPU rank; it must not be confused with the cluster-wide aggregate fabric traffic ($N \times V_{\mathrm{ring}}$). Translating this data volume into execution wall-clock time requires accounting for physical link speeds, algorithmic bus bandwidth efficiency, payload sizes, and connection **latency**.

Communication latency can be modeled using a classical $\alpha$–$\beta$ network model:

$$
T_{\mathrm{ring}}\approx2(N-1)L+2\frac{N-1}{N}\frac{S}{W}
$$

Where $L$ denotes the fixed per-step latency (the $\alpha$ component), and $W$ represents the effective transmission bandwidth (the reciprocal of the per-byte cost $\beta$). While this provides an analytical baseline, production NCCL implementations leverage multi-ring topologies, tree-based reductions, and NVLS (NVLink SHARP) hardware primitives tailored to the underlying network fabric.

In the same model AllGather is a single phase: in $N-1$ steps each GPU sends one $S/N$ chunk, so $T_{\mathrm{AllGather}}\approx(N-1)L+\frac{N-1}{N}\frac{S}{W}$. That is exactly why AllReduce takes about twice as long: it is a reduce-scatter followed by an all-gather. For small messages the $(N-1)L$ latency term dominates and for large ones the $S/W$ bandwidth term does, which is why measured NCCL bandwidth only saturates once messages are large enough. [K11]

### 10.3. SHARP: Moving Arithmetic into the Network Switch

Now consider a more radical architectural optimization: If GPUs are destined to transmit tensors across network fabrics purely to sum them together, why not execute that reduction directly inside the network switches while packets are in flight?

Modern high-performance network fabrics do precisely this. Rather than acting as passive packet forwarders, intermediate switches participate actively in floating-point math operations. This capability is termed **in-network computing**.

**SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** executes supported reduction operations (such as floating-point summations) directly within the switch ASICs of compatible NVIDIA InfiniBand networks.

<svg viewBox="0 0 720 334" role="img" aria-label="Two panels. Traditional: GPUs A, B and C exchange data through a switch that only forwards packets, and the additions are done on the GPUs, so data crosses the switch many times over the ring steps. SHARP: the switch performs supported reductions in hardware; each GPU sends its input up once and receives the result down once. The benefit depends on message size, reduction type and traffic, not a fixed 50 percent saving." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sh-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="sh-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="sh-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Where does the addition happen?</text>
<rect x="16" y="40" width="336" height="262" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Traditional: reduce on GPUs</text>
<rect x="134" y="84" width="100" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="184" y="102" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">switch</text>
<text x="184" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">forwards only</text>
<rect x="46" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU A</text>
<text x="82" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">adds</text>
<path d="M72 196 L164 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M174 126 L92 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="186" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU B</text>
<text x="186" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">adds</text>
<path d="M176 196 L184 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M194 126 L196 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="254" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="290" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU C</text>
<text x="290" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">adds</text>
<path d="M280 196 L204 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M214 126 L300 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="28" y="262" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">data crosses the switch many times (ring steps)</text>
<text x="28" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GPU cores spend time on arithmetic</text>
<rect x="368" y="40" width="336" height="262" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="380" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">SHARP: reduce in the switch</text>
<rect x="486" y="84" width="100" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="536" y="102" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">switch</text>
<text x="536" y="117" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">Σ in hardware</text>
<rect x="398" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="434" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU A</text>
<text x="434" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">sends once</text>
<path d="M424 196 L516 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M526 126 L444 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="502" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="538" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU B</text>
<text x="538" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">sends once</text>
<path d="M528 196 L536 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M546 126 L548 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="606" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="642" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU C</text>
<text x="642" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">sends once</text>
<path d="M632 196 L556 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M566 126 L652 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="380" y="262" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">up once: inputs</text>
<text x="490.46500000000003" y="262" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">· down once: result</text>
<text x="380" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">supported reductions handled in fabric</text>
<text x="16" y="322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Benefit depends on message size, reduction type and traffic — not a fixed 50% saving.</text>
</svg>

By aggregating tensors in-flight, SHARP reduces cross-sectional data movement and slashes collective latency. However, claims that "SHARP cuts all network traffic by exactly 50%" or "halves total training duration" are fundamentally inaccurate. The real-world speedup depends heavily on collective communication patterns, reduction algorithms, message payload sizes, and fabric contention dynamics. [K7]

A worked example makes the size of the effect concrete. Train Llama 3 70B with TP = 8 inside each DGX and data parallelism across 127 DGX. Each GPU holds the BF16 gradients of one eighth of the 70 billion parameters: $70\times10^9\times2/8=17.5$ GB. A 127-participant ring AllReduce between same-index GPUs (one rail) puts $2\cdot\frac{126}{127}\cdot17.5\approx34.7$ GB per GPU on the wire, about **0.69 s** on a 50 GB/s NIC. With SHARP each GPU sends its gradients up once and receives the result once: about 17.5 GB, ideally **0.35 s**. The gap approaches 2× for large, bandwidth-bound messages and shrinks for small ones or under real traffic.

### 10.4. Compute–Communication Overlap

In distributed algorithms, execution threads often stall waiting for communication buffers to clear. However, if the current layer contains compute operations that are mathematically decoupled from the in-flight tensor transfer, execution can proceed in parallel. Just as part of a team can draft a new chapter while another section is being transmitted for review, concurrent pipelining reduces aggregate idle time.

This principle recurs across virtually every distributed training optimization: **interleaving communication and computation to overlap execution timelines**.

While the GPU's **compute stream** crunches matrix multiplications for a given layer, an asynchronous communication stream dispatches tensors across the network fabric, masking network latency behind active compute cycles.

<svg viewBox="0 0 720 274" role="img" aria-label="Two timelines with a compute stream and a network stream. Poor schedule: compute steps 1 and 2 run, then the network transfer runs while compute sits idle, then compute steps 3 and 4. Overlapped: compute steps 1 to 4 run back to back while the independent part of the network transfer runs beneath them, so the step finishes earlier by the length of the transfer. This only works when the transfer does not depend on the math running beside it." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="os-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="os-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Compute and communication on two CUDA streams</text>
<text x="16" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">poor schedule</text>
<text x="140" y="82" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">compute</text>
<text x="140" y="116" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">network</text>
<line x1="150" y1="96" x2="704" y2="96" style="stroke:var(--c-border);stroke-width:0.6;stroke-dasharray:4 3"/>
<rect x="150" y="66" width="140" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="220.0" y="82.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute 1–2</text>
<rect x="290" y="100" width="140" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">network</text>
<rect x="290" y="66" width="140" height="26" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="360.0" y="82.96" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">idle</text>
<rect x="430" y="66" width="140" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="500.0" y="82.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute 3–4</text>
<line x1="570" y1="56" x2="570" y2="132" style="stroke:var(--c-text);stroke-width:1.4"/>
<text x="576" y="138" text-anchor="start" style="fill:var(--c-text);font-size:11px">finish</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">overlapped</text>
<text x="140" y="186" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">compute</text>
<text x="140" y="220" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">network</text>
<line x1="150" y1="200" x2="704" y2="200" style="stroke:var(--c-border);stroke-width:0.6;stroke-dasharray:4 3"/>
<rect x="150" y="170" width="280" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="290.0" y="186.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute 1–4</text>
<rect x="220" y="204" width="170" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="305.0" y="220.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">network (independent part)</text>
<line x1="430" y1="160" x2="430" y2="236" style="stroke:var(--c-text);stroke-width:1.4"/>
<text x="436" y="242" text-anchor="start" style="fill:var(--c-text);font-size:11px">finish</text>
<path d="M434 183 H566" marker-start="url(#os-b)" marker-end="url(#os-b)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5"/>
<text x="500" y="176" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:600">time saved</text>
<text x="16" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Only works when the transfer is independent of the math running beside it; a true dependency stays on the critical path.</text>
</svg>

However, communication cannot be masked unconditionally. If subsequent layer computations maintain strict data dependencies on tensors residing on remote GPUs, execution must stall at a synchronization barrier until the payload arrives.

---




---

### Unraveling Fabric Dynamics: Asking "Where Does the Data Actually Go?"

Discussions around distributed GPU communication frequently fixate on headline link bandwidth. In practice, the governing constraint is the exact physical traversal path: Does a packet remain within the local intra-node NVLink mesh, cross an internal PCIe switch domain, or traverse optical transceivers into a multi-tier leaf-spine InfiniBand fabric?

Two distinct operational regimes govern this data journey:
- **Latency-bound regime:** For small payloads (such as fine-grained tensor-parallel exchanges or barrier synchronizations), fabric hop counts and switch buffer queuing latencies dominate; link bandwidth is largely irrelevant if packets spend their lifecycle waiting in switch buffers.
- **Bandwidth-bound regime:** For massive tensor transfers (such as data-parallel gradient reductions or FSDP parameter gathering), bisection bandwidth, rail alignment, and congestion-free routing become paramount.

Sound architectural design starts by diagnosing which of these two physical regimes governs your specific collective workload.


### Why Expanding NVLink Domains Forces a Rethink of Tensor Parallelism

In conventional H100 DGX/HGX systems, eight GPUs are interconnected via high-speed NVLink across an on-board NVSwitch fabric, establishing $TP=8$ as the intuitive default for intra-node parallelism. However, **$TP=8$ is not an immutable law of physics**. Tensor Parallelism spanning beyond a single node is architecturally viable; it simply incurs the latency and bandwidth penalties of cross-node network fabrics.

Next-generation rack-scale architectures push this paradigm further. In the GB200 NVL72 platform, up to 72 GPUs share a unified, rack-level NVLink domain. This deployment shatters the misconception that "Tensor Parallelism beyond eight GPUs is unfeasible." Yet, having a 72-GPU *scale-up domain* does not imply that every production workload should automatically configure $TP=72$: attention head divisibility, hidden dimension sizes, memory sharding efficiencies, and communication overhead still dictate the optimal parallelism split. [K51]

### Can Communication Latency Be Fully Hidden?

Consider the overlap concept in a commercial kitchen: If a line cook prepares the sauce for an upcoming order while a runner restocks bread, prep time seamlessly overlaps with delivery. However, if the sauce cannot be prepared without the bread present, pipelined overlap is impossible.

Distributed training operates under an identical **directed acyclic dependency graph**. If the forward computation of layer $k$ is mathematically independent of the parameter AllGather transfer for layer $k+1$, the two operations can overlap. If a strict data dependency exists, execution must stall on the network fabric. Furthermore, if the communication stream and compute kernels contend for shared memory buses, PCIe lanes, or streaming multiprocessor resources, concurrent execution can mutually degrade throughput.

Consequently, engineering claims like "95% of communication overhead is hidden" are meaningless in isolation; they are valid only when contextualized by **specific model architectures, message sizes, compute intensity, and hardware interconnect specifications**.

## 11. Understanding Network Performance: Why Traffic Patterns Matter Just as Much as Topology

Low raw cable bandwidth is far from the only reason GPUs stall waiting for each other. Misconfigured GPU–NIC affinity, asymmetrical traffic bottlenecks, excessive network hops, packet drops, or software binding to the wrong network interface produce identical symptoms.

### 11.1. First, Ask: 'Which Traffic Pattern?'

| Traffic Pattern | Typical Characteristics | Dominant Risk |
|---|---|---|
| TP collective | Frequent and latency-sensitive | Intra-layer stall |
| DP AllReduce/ReduceScatter | Large tensor transfers | Network bandwidth saturation and congestion |
| MoE AllToAll | Multi-target, irregular routing | Hotspots and straggler/load imbalance |
| PP activation transfer | Point-to-point between adjacent stages | Pipeline bubbles |
| Checkpointing | High-volume bursts to storage | Storage fabric saturation and I/O throttling |
| Disaggregated KV transfer | Prefill → decode handoff | KV transfer latency wiping out throughput gains |

The key takeaway from this table: **When benchmarking a cluster network, a single generic iperf run tells you almost nothing about real workload performance.** If the NCCL collective message sizes and communication topologies differ, real-world behavior diverges dramatically.

### 11.2. A Simple Network Transfer Model

A first-order analytical model for transmitting a message:

$$
T_{\mathrm{transfer}}\approx T_{\mathrm{startup}} + \frac{B}{W_{\mathrm{effective}}}
$$

- $T_{\mathrm{startup}}$: Startup overhead, protocol handshake, and fixed latency cost.
- $B$: Total bytes transferred.
- $W_{\mathrm{effective}}$: Effective sustained bandwidth achieved in practice.

A link marketed as 400 Gb/s has a theoretical peak throughput of **50 GB/s**; in practice, protocol headers, PCIe/NIC bottlenecks, and contending traffic will yield substantially less. Moreover, the total execution time of a collective operation cannot simply be inferred from raw single-link speeds: the underlying collective algorithm (ring vs. tree), total bytes exchanged, and network diameter play decisive roles.

### 11.3. Bisection Bandwidth and Oversubscription

**Bisection bandwidth** defines the total aggregate throughput achievable across the narrowest boundary splitting a network into two equal halves. In a modern datacenter, equipping every host with a 400 Gb/s NIC does not imply that every host-pair can concurrently communicate at 400 Gb/s. If leaf–spine uplink capacity is undersized, the fabric becomes **oversubscribed**.

Example: Eight hosts attached to a leaf switch at 400 Gb/s each generate 3.2 Tb/s of downlink host capacity. If total uplink capacity from that leaf is only 1.6 Tb/s, the oversubscription ratio is **2:1**. If traffic is entirely inter-rack, it slams directly into this uplink bottleneck; if communications remain confined within the leaf, this subscription ratio poses no penalty.

### 11.4. RoCE vs. InfiniBand: A More Rigorous Answer to 'Which is Faster?'

An architect must first evaluate: RDMA capabilities across the selected switch and NIC vendors; workload-specific collective traffic patterns; operational maturity in managing lossy vs. lossless Ethernet congestion; end-to-end telemetry; and troubleshooting tooling. While RoCE runs RDMA semantics over Ethernet, achieving stable performance hinges strictly on **rigorous network validation**. Blindly enabling congestion control primitives like PFC (Priority Flow Control) and ECN (Explicit Congestion Notification) without fine-tuning buffer thresholds frequently backfires, causing head-of-line blocking, pause frames storms, or deadlocks.

Similarly, comparing InfiniBand against RoCE cannot be reduced to hardware price tags alone: switches, NICs, optical cabling, fabric management software, operational engineering overhead, and vendor ecosystem lock-in must all be evaluated holistically.

### 11.5. The Measurement Pyramid: From Physical Cable to Model



<svg viewBox="0 0 720 306" role="img" aria-label="Six-layer measurement pyramid, verified from the bottom up: 1, physical links, with link state, optics and error counters; 2, the host data path, with PCIe Gen5, NUMA and GPU-to-NIC affinity, checked with nvidia-smi topo -m and lspci; 3, RDMA micro-benchmarks for point-to-point bandwidth and latency; 4, NCCL collectives, such as all_reduce_perf bus bandwidth, AllGather and AllToAll; 5, model step time and compute–communication overlap; 6, workload SLO and cost, including time-to-quality, MFU and p99. Each layer&#x27;s baseline is meaningful only if the layer below is clean." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mp2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Network measurement pyramid: verify bottom-up</text>
<path d="M150 40 L190 40 L210 76 L130 76 Z" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="214" y1="58" x2="322" y2="58" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="55" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">6 · Workload SLO &amp; cost</text>
<text x="330" y="71" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time-to-quality, MFU, p99</text>
<path d="M130 80 L210 80 L230 116 L110 116 Z" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="234" y1="98" x2="322" y2="98" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">5 · Model step time</text>
<text x="330" y="111" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">compute/communication overlap</text>
<path d="M110 120 L230 120 L250 156 L90 156 Z" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="254" y1="138" x2="322" y2="138" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="135" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">4 · NCCL collectives</text>
<text x="330" y="151" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">all_reduce_perf busbw, AllGather, AllToAll</text>
<path d="M90 160 L250 160 L270 196 L70 196 Z" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<line x1="274" y1="178" x2="322" y2="178" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="175" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">3 · RDMA micro-benchmark</text>
<text x="330" y="191" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">point-to-point bandwidth and latency</text>
<path d="M70 200 L270 200 L290 236 L50 236 Z" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<line x1="294" y1="218" x2="322" y2="218" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="215" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">2 · Host data path</text>
<text x="330" y="231" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe Gen5, NUMA, GPU↔NIC: nvidia-smi topo -m, lspci</text>
<path d="M50 240 L290 240 L310 276 L30 276 Z" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.2"/>
<line x1="314" y1="258" x2="322" y2="258" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="255" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">1 · Physical links</text>
<text x="330" y="271" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">link state, optics, error counters</text>
<text x="16" y="294" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Each layer&#x27;s baseline is meaningful only if the layer below it is clean; sweep message sizes, repeat before and after upgrades.</text>
</svg>



**An instructive benchmarking progression:** Run `all_reduce_perf` intra-node first, then between two nodes, across a single rack, and finally across multiple racks. Sweep message sizes systematically from small to large at every stage, logging `busbw`, effective application bandwidth, p50/p99 latencies, and network error counters. Compare this baseline before and after any software, driver, or firmware upgrades. Usage patterns and examples for NCCL Tests are documented in [K39].

```bash
# Illustrative example: if NCCL Tests is compiled and all 8 node GPUs are idle:
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

# Inspect GPU↔NIC affinity and NUMA topology (given appropriate driver permissions)
nvidia-smi topo -m

# Verify PCI devices, link speeds, and network interfaces
lspci | grep -Ei 'nvidia|network|infiniband'
```

None of these three commands will magically diagnose a root cause on their own, but they establish an empirical baseline for systematic debugging. Rather than cargo-culting aggressive NCCL environment variables in production, engineers must first validate both physical wiring and OS-level topology mappings. [K38][K39]

---


## 12. GPU Cluster Validation and Acceptance Testing: Why Hardware Visibility Alone Falls Short

Procuring an AI cluster does not mean accepting a system that is production-ready. Seeing 1,024 GPUs listed in an `nvidia-smi` query is good news, but it provides zero proof that those 1,024 GPUs can execute **cohesively, stably, and at target performance** under full distributed load.

### 12.1. The Golden Image and Version Matrix

The host driver, CUDA toolkit, NCCL, container runtime, framework, Linux kernel, NIC firmware, OFED/network stack, and GPU fabric manager are inextricably linked. A **golden image** is a rigorously validated, version-controlled operating system and software base. Replacing ad-hoc node updates with automated, immutable image provisioning eliminates silent configuration drift and unrepeatable failures.

Representative version tracking matrix:

| Component | Metadata to Track | Rationale |
|---|---|---|
| GPU | Model, SKU, BIOS/firmware | Capacity and hardware feature consistency |
| Driver / CUDA | Exact version | ABI compatibility and software support matrix |
| NCCL / Framework | Exact version + image digest | Reproducibility of collective communication behavior |
| NIC | Model, firmware, port speeds | RDMA error tracing and link throughput profiling |
| OS / Kernel | Version, BIOS NUMA/ACS/IOMMU settings | Impacts P2P interconnects and DMA mapping |
| Storage Client | Filesystem version, mount policies | Dataset loading and checkpointing IOPS/bandwidth |
| Scheduler | Queue / topology / admission policies | Disentangling bad node placement from code regressions |

### 12.2. What Do Different Validation Tiers Reveal?

**Tier 1 — Node Health:** GPU discovery, expected VRAM capacity, ECC counts, Xid error logs, NVLink link status, thermals, power draw, cooling loop/fan telemetry, local storage SMART metrics, and NIC link states.

**Tier 2 — GPU Compute & Memory:** Sustained GEMM throughput, HBM memory bandwidth, and intra-node GPU↔GPU peer memory copies. Catching marginal hardware that passes superficial sanity checks but faults under sustained thermal load requires extended stress and soak testing. Naturally, such destructive stress testing must only run on drained, out-of-service nodes.

**Tier 3 — Network:** Multi-rail RDMA microbenchmarks and cross-rack NCCL collective sweeps to detect degraded links and asymmetric latency.

**Tier 4 — Storage:** Concurrent multi-client read bursts, large checkpoint dump simulations, and storage failover tests. Always evaluate tail latencies (p99) and metadata pressure rather than simple average throughput.

**Tier 5 — Application Workloads:** Reference end-to-end training steps, sustained soak runs, and verification of deterministic loss convergence on known configurations.

**Tier 6 — Fault Injection:** Controlled simulations of worker evictions, link drops, degraded storage I/O, and automated node cordoning/health tagging. Chaos experiments must never be executed on production clusters without strict isolation and approvals.

### 12.3. What Class of Problems Does DCGM Detect?

**DCGM (Data Center GPU Manager)** is NVIDIA's suite of datacenter-grade GPU telemetry, policy management, and health diagnostics. It captures critical events such as PCIe replay counts, uncorrectable double-bit ECC errors, dynamic page retirement, row remap exhaustion, Xid errors, and thermal/power throttling signals. However, not every metric is exposed on every GPU architecture; teams must audit their active exporter collector configs against the official hardware support matrix. Moreover, certain hardware profiling counters are time-multiplexed; an instantaneous zero reading does not necessarily imply an idle GPU. [K36][K37]

**A critical operational principle:** When a GPU logs uncorrectable memory errors or catastrophic Xid fault codes, your job is not to "tune software to improve MFU." The immediate priority is to safely cordon and drain the node, contain blast radius, and trigger hardware remediation workflows.

### 12.4. Sample Cluster Acceptance Checklist

```text
[ ] Hardware inventory and software version matrix logged
[ ] DCGM diagnostic and firmware health checks passed cleanly
[ ] Intra-node GPU↔GPU and GPU↔NIC topologies validated
[ ] Intra-node / two-node / intra-rack / cross-rack NCCL baselines established
[ ] Storage throughput and checkpoint save/restore verified
[ ] Model convergence and numeric regression tests passed
[ ] Real traffic benchmarks meet defined SLO targets
[ ] Rollback mechanisms, secrets isolation, and network segmentation audited
[ ] Single-node failure recovery and rolling upgrade scenarios rehearsed
[ ] Telemetry dashboards, alerting thresholds, and on-call runbooks in place
```

The true value of this checklist for an infrastructure architect: It forces hardware, platform, network, and ML engineering teams to agree on a shared, unambiguous definition of "production-ready."

---




---

**PART IV — Distributing the Model and Completing the Training Step**

*Where do weights, activations, gradients, and optimizer states actually go?*


## 13. The Art of Model Partitioning: TP, PP, DP, FSDP, CP, EP

Suppose we have thousands of GPUs at our disposal. How do we partition a 405-billion-parameter model across them?

There is no single universal answer. Sometimes we want identical model replicas processing different batches of data. Sometimes the parameter weight matrix of a single layer cannot fit into the memory of a single GPU. Sometimes network depth dictates slicing sequential layers across distinct machines. And in other cases, massive context lengths or sparse MoE routing introduce entirely different partitioning dimensions.

These strategies should not be viewed as mutually exclusive rivals, but as complementary tools designed to solve distinct physical bottlenecks.

As models scale into the hundreds of billions of parameters, relying on a single parallelization axis is rarely sufficient. Each strategy targets a specific constraint.

### 13.1. Data Parallelism (DP): Processing Different Data on Identical Models

Let us start with the most intuitive approach. Rather than having four students read the exact same book simultaneously, we place four identical copies of the model across four separate GPUs and feed each a unique batch of training data. Each replica independently executes forward and backward passes to compute local gradients; these gradients are subsequently aggregated across all replicas. This enables us to process substantially larger global batches in the same wall-clock time.

In classic DP, every worker maintains a full replica of model states, processes disjoint data mini-batches, and synchronizes gradients via an AllReduce collective before taking an optimizer step.

<svg viewBox="0 0 720 296" role="img" aria-label="Data parallelism. The global batch is split into shards A, B and C. Each goes to a GPU group holding a full model replica, which runs forward and backward and produces gradients A, B and C. An AllReduce averages the gradients, and every replica applies the identical optimizer step. It is simple to scale with data, but every replica stores the full model state and gradients must be synchronized every step." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dp-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Data parallelism: same model, different data, one shared update</text>
<rect x="16" y="104" width="96" height="60" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="64.0" y="131.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Global batch</text>
<text x="64.0" y="146.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">split 3 ways</text>
<path d="M112 134 C 130 134, 130 70, 150 70" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="56" width="58" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="179.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">shard A</text>
<line x1="208" y1="70" x2="224" y2="70" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="46" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU group A</text>
<text x="319" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">full model replica · fwd + bwd</text>
<line x1="412" y1="70" x2="428" y2="70" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="56" width="70" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad A</text>
<path d="M500 70 C 520 70, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M112 134 C 130 134, 130 132, 150 132" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="118" width="58" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="179.0" y="135.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">shard B</text>
<line x1="208" y1="132" x2="224" y2="132" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="108" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="128" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU group B</text>
<text x="319" y="144" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">full model replica · fwd + bwd</text>
<line x1="412" y1="132" x2="428" y2="132" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="118" width="70" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="465.0" y="135.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad B</text>
<path d="M500 132 C 520 132, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M112 134 C 130 134, 130 194, 150 194" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="180" width="58" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="179.0" y="197.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">shard C</text>
<line x1="208" y1="194" x2="224" y2="194" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="170" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="190" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU group C</text>
<text x="319" y="206" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">full model replica · fwd + bwd</text>
<line x1="412" y1="194" x2="428" y2="194" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="180" width="70" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="465.0" y="197.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad C</text>
<path d="M500 194 C 520 194, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="540" y="106" width="164" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="622.0" y="132.0" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">AllReduce</text>
<text x="622.0" y="147.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">average gradients</text>
<path d="M622 162 V240 H319 V220" marker-end="url(#dp-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="470" y="256" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">identical optimizer step on every replica</text>
<text x="16" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">+ simple to scale with data   − every replica stores the full model state; gradient sync on every step</text>
</svg>

**Advantage:** Conceptually simple and scales smoothly with data volume. **Trade-off:** High memory footprint due to redundant model state replicas on every GPU, plus network overhead for gradient synchronization.

When does DP lose to its own communication? Compute grows with the batch, while the gradient AllReduce is fixed by model size. The Scaling Book's first-order condition is [K11]:

$$
\frac{B}{X}\gtrsim\frac{C_{\mathrm{peak}}}{W_{\mathrm{collective}}}
$$

Read it as: $B/X$ is the token batch per GPU, $C_{\mathrm{peak}}$ the GPU's peak FLOP/s and $W_{\mathrm{collective}}$ the bandwidth the collective runs on. Inside a node (NVLink, 450 GB/s) this is $990\times10^{12}/450\times10^{9}=2{,}200$ tokens per GPU; across nodes (400 GB/s node egress) it is $990\times10^{12}/400\times10^{9}=2{,}475$. Below that per-GPU batch, gradient sync outlasts the math, which is why overlap and FSDP's timing matter.

### 13.2. Tensor Parallelism (TP): Sharding Intra-Layer Matrices

But what happens when the weight tensors of even a single transformer layer exceed the memory capacity of a GPU? Here, we must shard the actual **parameter matrices inside each layer**. Rather than different workers reading different chapters of a book, imagine multiple workers computing distinct columns of a massive algebraic matrix concurrently.

For a standard linear projection $Y=XW$, column-parallel partitioning splits $W$ along its columns:

$$
W=[W_1\ W_2\ W_3\ W_4],\qquad
Y=[XW_1\ XW_2\ XW_3\ XW_4]
$$

Each GPU processes its local slice $W_i$. Depending on the layer architecture, partial activations are either gathered or reduced in the subsequent operation. In Megatron-LM-style transformer layers, column-parallel and row-parallel linear layers are paired consecutively; intermediate activations can flow without synchronization, requiring an AllReduce or ReduceScatter only at layer boundaries. The common claim that **every TP layer requires an identical fixed number of blocking AllReduce collectives** is technically inaccurate.

<svg viewBox="0 0 720 352" role="img" aria-label="Tensor parallelism with four GPUs in a Megatron-style MLP. Every GPU receives the same input X. GPU i holds column block i of W1 and computes the activation of X times W1 i without communicating; it then multiplies by row block i of W2 to get a partial result Z i. An AllReduce sums the four partial results into Z. This AllReduce sits on the critical path of every layer, so TP is usually kept inside the NVLink domain." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tp-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Tensor parallelism in a Megatron-style MLP (TP = 4)</text>
<rect x="16" y="140" width="70" height="60" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="51.0" y="175.76" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-family:var(--font-mono)">X</text>
<text x="51" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">same X on</text>
<text x="51" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">every GPU</text>
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
<text x="196" y="302" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">column-parallel (no communication)</text>
<text x="392" y="302" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">row-parallel (partial sums)</text>
<rect x="562" y="140" width="142" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="633.0" y="167.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">AllReduce</text>
<text x="633.0" y="183.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">Z = Σ Zᵢ</text>
<text x="633" y="222" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">on the critical path,</text>
<text x="633" y="236" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">every layer → NVLink</text>
<text x="16" y="324" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Each GPU holds ¼ of W1 and W2. One AllReduce per MLP forward: small, frequent, latency-sensitive —</text>
<text x="16" y="340" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">so keep TP inside the NVLink domain when you can.</text>
</svg>

It is now clear why TP is almost exclusively mapped across ultra-high-speed intra-node NVLink domains: fine-grained synchronization is required at every single transformer layer. Nonetheless, **TP > 8 is technically viable across clusters**; deciding whether to cross node boundaries depends strictly on model dimensions, fabric bisection bandwidth, and empirical latency profiling. [K3]

We can turn this intuition into an inequality. For a Megatron MLP split over $Y$ GPUs, compute time and AllReduce time are roughly [K11]:

$$
T_{\mathrm{math}}=\frac{4BDF}{Y\,C_{\mathrm{peak}}},\qquad T_{\mathrm{comm}}=\frac{4BD}{W_{\mathrm{collective}}}
$$

Read it as: $B$ tokens, hidden size $D$, MLP width $F$, TP degree $Y$. For the math to outlast the communication, $T_{\mathrm{math}}\ge T_{\mathrm{comm}}$:

$$
Y\le\frac{F\,W_{\mathrm{collective}}}{C_{\mathrm{peak}}}
$$

For Llama 3 70B, $F=28{,}672$. On NVLink $C/W=990\times10^{12}/450\times10^{9}=2{,}200$, so $Y\le28{,}672/2{,}200\approx13$; the natural power-of-two choice is **8**. Try the same TP across nodes over a 50 GB/s NIC per GPU and $C/W=19{,}800$: the bound collapses to $Y\le1.45$, so TP barely scales at all. A second, practical reason for TP = 8 is divisibility: Llama 3 70B has 64 attention heads and 8 KV heads [K21]. At TP = 8 each GPU owns exactly one KV head and eight query heads; at TP = 16 the KV heads would have to be replicated.

### 13.3. Pipeline Parallelism (PP): Partitioning Layers Across Sequential GPUs

Picture an industrial assembly line: the first station shapes the chassis, the second mounts the engine, and the third conducts quality inspection. An individual vehicle must travel sequentially through all three stations, but once the pipeline is primed, multiple vehicles are processed concurrently across different stations.

This is the intuition behind Pipeline Parallelism. Early transformer layers reside on one set of GPUs, intermediate layers on the next, and final layers on the last. However, when the pipeline is ramping up (fill phase) or winding down (drain phase), upstream or downstream GPUs sit idle; this dead time is known as the **pipeline bubble**.

<svg viewBox="0 0 720 302" role="img" aria-label="Pipeline schedule with 4 stages and 6 microbatches, forward pass only, over 9 time slots. Stage 0 runs microbatches 1 to 6 in slots 1 to 6; each later stage starts one slot later, so stage 3 runs them in slots 4 to 9. The empty slots at the start of later stages and at the end of earlier stages are the pipeline bubble: 3 of 9 slots per stage, about 33 percent, matching p minus 1 over m plus p minus 1. With 16 stages the ideal bubble is 48.4 percent for 16 microbatches, 31.9 percent for 32 and 10.5 percent for 128." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Pipeline bubble: p = 4 stages, m = 6 microbatches (forward pass)</text>
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
<text x="110" y="222" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">dashed red = bubble (idle stage)</text>
<rect x="16" y="234" width="688" height="56" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="254" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Bubble ≈ (p − 1) / (m + p − 1) = 3 / 9 ≈ 33% here.</text>
<text x="28" y="274" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">p = 16: m = 16 → 48.4% · m = 32 → 31.9% · m = 128 → 10.5% (idealized, equal stages)</text>
</svg>

Each stage holds a sequential slice of transformer layers. Across stage boundaries, GPUs exchange boundary **activation tensors during forward passes and activation gradient tensors during backward passes**, rather than massive weight matrices.

How big is that boundary tensor? Llama 3 405B has a hidden size of 16,384; the BF16 activation of an 8,192-token microbatch is $8192\times16384\times2\approx268$ MB. Because TP = 8 splits it over eight GPUs, each GPU sends about **34 MB** per microbatch to the neighbouring stage [K5]. Next to TP's two AllReduces per layer that volume is small: the real price of PP is not bandwidth but the **bubble** below.

But until the first microbatch reaches the final stage, downstream stages remain idle. This latency overhead is the **pipeline bubble**.

For an idealized schedule with balanced stage computation times (e.g., standard 1F1B scheduling), the theoretical pipeline bubble fraction is approximately:

$$
Bubble\approx\frac{p-1}{m+p-1}
$$

- $p$: number of pipeline stages.
- $m$: number of microbatches per global batch.

For example, when $p = 16$:

| Number of Microbatches ($m$) | Theoretical Bubble Fraction |
|---:|---:|
| 16 | 15/31 ≈ 48.4% |
| 32 | 15/47 ≈ 31.9% |
| 128 | 15/143 ≈ 10.5% |

This exact bubble overhead depends on scheduling schemes (e.g., interleaved 1F1B), asymmetry between forward and backward execution times, and stage compute imbalances. Furthermore, pushing $m$ higher to shrink the bubble increases peak activation memory, requiring careful memory management.

### 13.4. FSDP and ZeRO: Why Replicate Model States Everywhere?

Recall the steep memory cost of Data Parallelism: every replica redundantly stores full copies of parameters, gradients, and optimizer states. In a cluster of thousands of GPUs, storing identical floating-point values thousands of times over becomes an unacceptable waste of expensive HBM.

The ZeRO (Zero Redundancy Optimizer) family and FSDP eliminate this redundancy by sharding model states across the data-parallel domain. However, whatever memory is saved must eventually be communicated when needed during computation. This exposes the fundamental trade-off: **memory footprint reduction in exchange for additional communication volume**.

**FSDP (Fully Sharded Data Parallelism)** shards model parameters, gradients, and optimizer states across data-parallel ranks:

- **ZeRO-Stage 1:** Shards optimizer states across ranks.
- **ZeRO-Stage 2:** Shards optimizer states and gradients across ranks.
- **ZeRO-Stage 3 (FSDP):** Shards optimizer states, gradients, and model parameters across ranks.

This drastically slashes the *persistent* memory footprint on each GPU. The price is runtime collective communication: parameters must be fetched on-the-fly via **AllGather** before each forward/backward pass, and gradients must be synchronized and sharded via **ReduceScatter**.

<svg viewBox="0 0 720 298" role="img" aria-label="Per-GPU model state with Adam in bytes per parameter for 4-way data parallelism. Plain DP keeps everything on every GPU: 2 bytes of parameters, 2 of gradients and 12 of optimizer state, 16 bytes. ZeRO-1 shards the optimizer: 2 plus 2 plus 3, 7 bytes, using ReduceScatter. ZeRO-2 also shards gradients: 2 plus 0.5 plus 3, 5.5 bytes. ZeRO-3 or FSDP also shards parameters: 0.5 plus 0.5 plus 3, 4 bytes, and needs AllGather plus ReduceScatter. Memory saved is paid back in communication." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="zr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Per-GPU model state with Adam (bytes per parameter, DP = 4)</text>
<text x="16" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">plain DP</text>
<text x="16" y="83" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">nothing sharded</text>
<rect x="130" y="54" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="54" width="43" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="196.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="218" y="54" width="263" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="350.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">12</text>
<text x="490" y="72" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">16 B</text>
<text x="704" y="72" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">—</text>
<text x="16" y="118" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-1</text>
<text x="16" y="133" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">optimizer sharded</text>
<rect x="130" y="104" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="104" width="43" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="196.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="218" y="104" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="251.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="292" y="122" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">7 B</text>
<text x="704" y="122" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">ReduceScatter</text>
<text x="16" y="168" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-2</text>
<text x="16" y="183" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">+ gradients sharded</text>
<rect x="130" y="154" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="154" width="10" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="185" y="154" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="218.0" y="172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="259.0" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">5.5 B</text>
<text x="704" y="172" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">ReduceScatter</text>
<text x="16" y="218" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-3 / FSDP</text>
<text x="16" y="233" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">+ parameters sharded</text>
<rect x="130" y="204" width="10" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="204" width="10" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="152" y="204" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="185.0" y="222" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="226.0" y="222" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">4 B</text>
<text x="704" y="222" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">AllGather + ReduceScatter</text>
<rect x="16" y="254" width="12" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="34" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">params</text>
<rect x="120" y="254" width="12" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="138" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">grads</text>
<rect x="222" y="254" width="12" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="240" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">optimizer (FP32 master + 2 moments)</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Memory saved is paid back in communication: sharded state must be gathered or scattered when it is needed.</text>
</svg>

**Intuition:** Instead of eight researchers keeping eight identical 1,000-page textbooks on their desks, they divide the book into eight sections. Each person holds only one section. When anyone needs to read another chapter, they request it over the network. That is the exact memory–communication trade-off of FSDP.

### 13.5. Context Parallelism (CP) and Sequence Parallelism (SP)

**CP (Context Parallelism)** partitions long token sequences along the sequence dimension across GPUs. Because self-attention requires each token to query key/value states across the entire sequence, CP introduces inter-GPU communication (e.g., ring attention or all-to-all K/V exchanges). **SP (Sequence Parallelism)**, in contrast, is an optimization coupled with tensor parallelism that shards activations along the sequence length in non-tensor-parallel regions (such as LayerNorm and Dropout) to eliminate redundant activation storage. The two techniques address distinct bottlenecks and are not synonymous. [K3]

### 13.6. Expert Parallelism (EP): Distributing Mixture-of-Experts

Not every inquiry needs to be routed to the same generalist. In an organization, legal questions go to the legal department and financial queries to accounting. Similarly, in Mixture-of-Experts (MoE) architectures, tokens do not pass through the same monolithic feed-forward network (FFN); instead, a learned router activates only a small subset of specialized experts per token.

While this drastically reduces active FLOPs per token, it introduces a structural communication challenge: if the expert selected for a given token resides on a different GPU, that token's activation must be dispatched across the network. Managing this distributed dispatch is the role of Expert Parallelism.

In **Mixture-of-Experts (MoE)** models, each token is dynamically assigned to a fixed top-$k$ subset of experts. The **gating router** computes routing probabilities. When experts are sharded across separate GPUs, token representations must be dispatched and gathered via cross-GPU **AllToAll** collective operations.

<svg viewBox="0 0 720 298" role="img" aria-label="Expert parallelism with four GPUs, each hosting one expert. Tokens on each GPU are coloured by the expert their router chose. An AllToAll dispatch sends every token to the GPU that holds its expert; expert E2 receives six tokens while the others receive two, so it becomes hot. After the experts run, a second AllToAll combine returns the results to the source GPUs. MoE cuts active FLOPs per token but adds two network shuffles per layer, and an unbalanced router creates a straggler." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="me-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="me-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Expert parallelism: tokens travel to their experts and back (AllToAll × 2)</text>
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
<text x="368" y="187" text-anchor="start" style="fill:var(--c-danger);font-size:10.5px;font-weight:700">hot!</text>
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
<text x="686" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">back to</text>
<text x="686" y="192" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">source</text>
<text x="16" y="270" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Fewer active FLOPs per token, but two network shuffles per MoE layer,</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">and an unbalanced router turns one expert into everyone&#x27;s straggler.</text>
</svg>

**Dense model:** Every token traverses the identical execution graph and weight tensors.

**MoE model:** Each token routes through a dynamically selected subset of experts; active computation (FLOPs) drops, but AllToAll communication volume and expert load imbalance introduce major latency stragglers.

The DeepSeek-V3 technical report details innovative expert allocation schemes and a specialized compute/communication overlapping pipeline known as **DualPipe**. The practical speedups of such approaches must always be analyzed in conjunction with their underlying architectural choices and constrained H800 cluster fabrics. [K9]

MoE also moves DP's critical batch. When each token runs only $k$ of $E$ experts, compute per GPU shrinks by $k/E$ while the weights to synchronize stay the same, so the threshold grows by $E/k$ [K11]:

$$
\left(\frac{B}{X}\right)_{\mathrm{MoE}}\gtrsim\frac{E}{k}\cdot\frac{C_{\mathrm{peak}}}{W_{\mathrm{collective}}}
$$

With $E=128$ and $k=4$ that is $32\times2{,}475=79{,}200$ tokens per GPU across nodes, a number no real run reaches. That is why large MoE training does not replicate expert weights with DP but spreads them with EP, and pays with the two AllToAlls in the figure.

### 13.7. How to Map Parallelism Dimensions to Hardware Topology

| Strategy | Sharded Dimension | Primary Objective | Communication Pattern |
|---|---|---|---|
| DP | Batch | Data throughput scaling | Gradient reduction |
| TP | Intra-layer weights | Sharding oversized layers | High-frequency intra-layer collectives |
| PP | Layer depth | Partitioning deep architectures | Point-to-point activations/gradients to stage peers |
| FSDP | Model states | Eliminating redundant VRAM | Layer-wise AllGather and ReduceScatter |
| CP | Sequence length | Long context window & activation scaling | Cross-rank K/V and attention exchanges |
| EP | Experts | MoE expert sharding | Dynamic token dispatch & combine (AllToAll) |

The two bounds derived above, for TP and for DP, side by side:

<svg viewBox="0 0 720 310" role="img" aria-label="Two communication rooflines for H100 at 990 TFLOP/s. Left, tensor parallelism: the degree Y must stay below F times W over C. For the Llama 3 70B MLP with F equal 28,672, NVLink at 450 GB/s allows Y up to about 13, so TP 8 fits in one DGX; a 50 GB/s NIC allows only about 1.45. Right, data parallelism: the per-GPU token batch must exceed C over W to hide gradient synchronization: 2,200 tokens inside a node, 2,475 across nodes at 400 GB/s per node, and 79,200 for a mixture-of-experts model with 128 experts and 4 active, because the bound grows by E over k." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="cr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Communication rooflines: when does splitting stop paying?</text>
<rect x="16" y="40" width="340" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700;font-family:var(--font-mono)">TP: Y ≤ F · W / C</text>
<text x="28" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP, F = 28,672</text>
<text x="40" y="106" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NVLink 450 GB/s</text>
<rect x="40" y="112" width="236.218" height="22" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="282.21818181818185" y="128" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Y ≤ 13.0</text>
<text x="40" y="168" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NIC 50 GB/s</text>
<rect x="40" y="174" width="26.2465" height="22" rx="2" style="fill:var(--c-danger);fill-opacity:.55"/>
<text x="72.24646464646464" y="190" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Y ≤ 1.45</text>
<line x1="185" y1="92" x2="185" y2="230" style="stroke:var(--c-accent);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="185.0" y="244" text-anchor="middle" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">TP = 8 (one DGX)</text>
<text x="40.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">0</text>
<text x="112.5" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">4</text>
<text x="185.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">8</text>
<text x="257.5" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">12</text>
<text x="330.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">16</text>
<rect x="364" y="40" width="340" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="60" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700;font-family:var(--font-mono)">DP: B/X ≥ C / W</text>
<text x="376" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tokens per GPU needed to hide gradient sync</text>
<text x="388" y="102" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">inside a node (450 GB/s)</text>
<rect x="388" y="108" width="49.9937" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="443.9937114000421" y="123" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2,200</text>
<text x="388" y="150" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">across nodes (400 GB/s/node)</text>
<rect x="388" y="156" width="57.462" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="451.4619796773598" y="171" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2,475</text>
<text x="388" y="198" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">MoE, E = 128, k = 4 (× E/k)</text>
<rect x="388" y="204" width="277.214" height="20" rx="2" style="fill:var(--c-danger);fill-opacity:.55"/>
<text x="659.213876512066" y="219" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">79,200</text>
<text x="388.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">1K</text>
<text x="534.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">10K</text>
<text x="680.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">100K</text>
<text x="16" y="298" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">C = 990 TFLOP/s (H100 BF16). Bounds are first-order (Scaling Book); overlap and topology move them, but not by 10×.</text>
</svg>

**Important Caveat:** For dense models, the global cluster world size is typically expressed as $TP \times PP \times CP \times DP$. Expert Parallelism (EP) can also be layered in; however, in real-world implementations, expert-parallel groups often overlap or interleave with other parallelism dimensions (e.g., sharing ranks with DP or CP). Mechanically multiplying all five dimensions together does not apply universally across all framework implementations.

### 13.8. Real-World Parallelism Case Study: Llama 3 405B

Two configurations from Meta's official Llama 3 technical report: [K5]

| GPUs | TP | CP | PP | DP | Sequence Length | Reported BF16 MFU |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 38% |

Factoring the first row:

$$
8\times1\times16\times128=16{,}384
$$

<svg viewBox="0 0 720 324" role="img" aria-label="How Llama 3 405B maps onto 16,384 H100 GPUs. A tensor-parallel group of 8 GPUs fills one DGX and talks over NVLink. Sixteen such servers in a chain form the 16 pipeline stages of one model replica, 128 GPUs, passing activations server to server. 128 replicas form the data-parallel dimension, synchronizing gradients. With 8K context the configuration TP 8, CP 1, PP 16, DP 128 reached 41 percent BF16 MFU; with 128K context, TP 8, CP 16, PP 16, DP 8 reached 38 percent. The chattiest axis stays on the fastest links; Meta&#x27;s fabric was RoCE." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="l4-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="l4-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="l4-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="l4-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Llama 3 405B on 16,384 H100: 8 × 16 × 128 (TP × PP × DP)</text>
<path d="M620 58 H630 V208 H40" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<path d="M613 51 H623 V201 H33" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<rect x="16" y="44" width="600" height="150" rx="10" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="28" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">one model replica = 128 GPUs</text>
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
<text x="28" y="184" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">PP = 16 stages (one DGX each) → activations hop server to server</text>
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
<text x="668" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVLink,</text>
<text x="668" y="161" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">one DGX</text>
<text x="632" y="226" text-anchor="end" style="fill:var(--c-warn);font-size:12.5px;font-weight:700">× 128 replicas = DP</text>
<text x="16" y="226" text-anchor="start" style="fill:var(--c-warn);font-size:11px">gradients are synchronized across replicas</text>
<rect x="16" y="240" width="688" height="24" rx="4" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">8K context</text>
<text x="160" y="256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">TP 8 · CP 1 · PP 16 · DP 128</text>
<text x="692" y="256" text-anchor="end" style="fill:var(--c-accent);font-size:12px;font-weight:700">BF16 MFU 41%</text>
<rect x="16" y="268" width="688" height="24" rx="4" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="284" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">128K context</text>
<text x="160" y="284" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">TP 8 · CP 16 · PP 16 · DP 8</text>
<text x="692" y="284" text-anchor="end" style="fill:var(--c-accent);font-size:12px;font-weight:700">BF16 MFU 38%</text>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Fast links carry the chattiest axis: TP stays inside NVLink, PP and DP cross the (RoCE) fabric. Meta reports 38–43% MFU.</text>
</svg>

**Crucial nuance:** The "38% MFU" figure cited in this article's introduction was chosen as an illustrative case study. Meta's massive 16,384-GPU H100 pretraining runs were not perpetually pinned at 38%. The official report documents MFU figures ranging between **38% and 43%** depending on sequence length and parallelism topology. Specifically, the 8K context run achieved **41%**, whereas the 128K long-context configuration dipped to **38%** due to communication overhead and memory constraints.

---

### 13.9. Sharding GPUs Is Not Enough: Scheduling Jobs in the Right Place at the Right Time

Having mapped out *how* models are sharded using TP, PP, DP, and EP, we face the next critical question: In an enterprise cluster concurrently running a 405B pretraining run, a 70B fine-tuning job, exploratory Jupyter notebooks, and mission-critical inference endpoints, **which job lands on which GPU, and when?**

This introduces a distinct engineering discipline separate from tensor calculus: **GPU orchestration** and **workload scheduling**.

Imagine an 8-person executive committee checking into a hotel for a summit. If the front desk can only allocate rooms to four members, the meeting cannot convene—yet those four occupied rooms block other incoming guests. In distributed training, spinning up a subset of an interdependent parallel gang is completely useless.

For this reason, **gang scheduling (all-or-nothing admission)** guarantees that tightly coupled distributed workers are scheduled atomically: either all required GPU pods are granted allocation simultaneously, or none are. Within the cloud-native Kubernetes ecosystem, frameworks like Kueue address this challenge via multi-tenant queuing and admission checks. Still, admitting a gang does not guarantee that every distributed worker will initialize without fault at the exact same microsecond. [K24]

| Concept | Concise Definition | Core Infrastructure Question |
|---|---|---|
| **Scheduler** | Component mapping pending tasks to hardware resources | On which physical nodes will this workload launch? |
| **Gang Scheduling** | Atomic all-or-nothing admission for interdependent workers | Are all TP/PP/DP ranks simultaneously available? |
| **Topology-Aware Placement** | Packing jobs with respect to NVLink, PCIe, and rack boundaries | Is the entire TP group localized within an ultra-fast NVLink domain? |
| **Quota Management** | Resource limits enforced per team or tenant | Can a rogue experiment monopolize cluster capacity? |
| **Preemption** | Evicting lower-priority workloads for urgent tasks | If a production service spikes, which training run is safely paused? |
| **Resource Fragmentation** | Scattered idle capacity that cannot satisfy gang shapes | There are 8 idle GPUs cluster-wide, but are they on the same physical host? |

Choosing between Kubernetes, Slurm, or hybrid orchestrators cannot be decided by a feature checklist alone. Massive, long-running, tightly synchronized training runs demand entirely different scheduling primitives than elastic, latency-critical inference fleets. Cloud-native AI organizations frequently standardize on Kubernetes augmented with specialized GPU operators and admission controllers, while traditional HPC and national supercomputing environments heavily rely on Slurm batch schedulers.

#### Multi-Tenant GPU Sharing: MIG, Time-Slicing, and MPS Are Not Interchangeable

**MIG (Multi-Instance GPU)** physically partitions compatible GPUs at the hardware level into fully isolated GPU instances, carving up compute units, memory controllers, and cache slices. This provides predictable QoS and hardware fault isolation for smaller concurrent workloads. However, MIG capabilities vary sharply across GPU architectures; engineers must consult official hardware matrices before designing multi-tenant architectures around it. [K25]

**Time-slicing** rotates access to the GPU chronologically via software context switching, which can introduce latency jitter. **MPS (Multi-Process Service)**, by contrast, allows distinct CUDA applications to multiplex concurrently onto the same hardware execution pipeline, overlapping kernel execution and saturating SMs. Crucially, neither time-slicing nor MPS provides the hardware-enforced memory isolation or fault containment of MIG.

When large-scale distributed training demands uninhibited access to SMs and high-speed fabrics, aggressive multi-tenancy can cosmetically boost cluster utilization metrics while significantly degrading end-to-end job completion time (JCT). An effective AI cluster scheduler cannot merely optimize for **allocated GPUs / total GPUs**; it must optimize for **job completion time, queue waiting times, and SLA/SLO adherence**.

#### The Segregated Pool Strategy

A robust, battle-tested production pattern is partitioning clusters into distinct **GPU pools** optimized for workload characteristics:

- **Latency-Sensitive Inference:** Prioritizing strict P99 latency bounds, dynamic autoscaling, and fault-tolerant replica failover.
- **Synchronous Pretraining:** Prioritizing atomic gang admission, homogeneous hardware topologies, and lossless network fabrics.
- **Fine-Tuning & Exploratory Dev:** Prioritizing rapid queue turnaround, fair-share quotas, and aggressive preemption.

This is not a dogmatic mandate; if workloads share homogeneous profiles, a unified pool may yield superior aggregate bin-packing efficiency. The primary engineering goal is aligning physical resource allocation with **inherent workload dynamics**.

---

## 14. Where Does Data Actually Go During a Single LLM Training Step?

So far, we have examined the individual components in isolation. At this stage, we need to observe them operating together on the same stage.

Let us trace a tokenized slice of data sitting in storage through its entire journey across a single training step. Where is the data read from? Which GPU receives it? At what point do Tensor Cores fire? Where does one GPU stall waiting on another? When do weights actually get updated? And if we snapshot the training state, which exact components are committed to disk?

Once you understand this journey, your approach to the question "Why is the GPU stalling?" will fundamentally change. You will be able to pinpoint bottlenecks to specific stages in the execution pipeline.

Now that we understand the constituent parts, let us trace a **single training step** of a massive dense model as it flows through the system. We will reference the scale of Llama 3 405B and the TP=8, PP=16, DP=128 parallel layout described in Meta's technical report. Keep in mind that the timing budgets and data volumes presented below are **didactic approximations**; we avoid presenting details omitted from Meta's report as empirical facts. [K5]

### Step 1 — Token Data Is Read from Storage

Training corpora are typically processed through a preprocessing pipeline and converted into pre-tokenized samples. **Tokenization** is the process of converting raw text into sequences of numerical token IDs corresponding to the model vocabulary.

<svg viewBox="0 0 720 300" role="img" aria-label="Path of token data in step 1: a distributed file system or object store is read by CPU DataLoader workers, which sample, order, pad or pack; the batch lands in pinned host RAM or uses direct I/O, then an H2D copy over PCIe brings it into GPU HBM, followed by embedding and the Transformer forward pass. Beside it, a size comparison: 16 million tokens times 4 bytes is only 64 MB of token IDs per global step, tiny next to a roughly 7.5 TB checkpoint burst." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dp2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dp2-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Step 1 in detail: how tokens reach HBM</text>
<rect x="16" y="42" width="300" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="166.0" y="68.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Distributed FS / object store</text>
<line x1="166" y1="84" x2="166" y2="98" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="95" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">file read</text>
<rect x="16" y="100" width="300" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="166.0" y="126.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU DataLoader workers</text>
<line x1="166" y1="142" x2="166" y2="156" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">sample, order, pad / pack</text>
<rect x="16" y="158" width="300" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="166.0" y="184.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Pinned host RAM (or direct I/O)</text>
<line x1="166" y1="200" x2="166" y2="214" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="211" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">H2D copy over PCIe</text>
<rect x="16" y="216" width="300" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="166.0" y="242.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM</text>
<text x="326" y="242" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">→ embedding → Transformer forward</text>
<rect x="380" y="42" width="324" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="392" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Per global step (Llama 3 405B scale)</text>
<text x="392" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">16 × 10⁶ token × 4 B</text>
<text x="392" y="106" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">= 64 MB of token IDs</text>
<rect x="392" y="116" width="3" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.8"/>
<text x="400" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tiny next to a checkpoint:</text>
<rect x="392" y="140" width="300" height="16" rx="2" style="fill:var(--c-danger);fill-opacity:.45"/>
<text x="392" y="174" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:600">~7.5 TB checkpoint burst</text>
<text x="392" y="190" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-style:italic">(bar lengths not to scale)</text>
<text x="16" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">For text pretraining the hard part is access pattern and checkpoint bursts, not raw token bytes; prefetch hides the read behind compute.</text>
</svg>

The **DataLoader** is responsible for delivering the correct training samples to compute engines just in time. **Pinned memory (page-locked RAM)** is non-pageable host memory that enables asynchronous, high-throughput DMA transfers between host and GPU. **Prefetching** stages upcoming batches while current batches are executing. **Sequence packing** concatenates shorter sequences to maximize token slot utilization per sequence length window.

An intuitive yet surprising back-of-the-envelope calculation: if a global step processes 16 million tokens and token IDs are stored as 4-byte integers, the raw token ID volume is:

$$
16\times10^6\ \mathrm{token}\times 4\ \mathrm{byte/token}=64\ \mathrm{MB}
$$

This footprint is minuscule compared to a multi-terabyte model checkpoint! Consequently, in text-only pretraining, the storage fabric's primary stress does not stem from streaming raw token bytes; rather, it comes from the concurrent access patterns of thousands of DataLoader workers and the massive **burst writes of checkpoints**. (This dynamic changes drastically in multimodal pipelines handling high-resolution image and video tensors.)

The picture holds per second as well. If each of 16,384 GPUs produces ~400 TFLOP/s of useful compute and a token costs $6\times405\times10^{9}$ FLOPs, the cluster processes $16384\times400\times10^{12}/(6\times405\times10^{9})\approx2.7$ million tokens per second; with 4-byte token IDs that is only **~11 MB/s**. The hard part is not bandwidth but feeding every GPU without a stall. Production frameworks therefore pre-tokenize the corpus into a contiguous `.bin` file of token IDs plus an `.idx` file of document offsets and read it with `mmap` (see §16.3). [K5][K29]

### Step 2 — The Tensor-Parallel Group Processes Its Layer Shard

The data is now resident in GPU memory. First, token IDs are mapped to embedding vectors, after which matrix multiplications across Transformer layers commence. If the model is parallelized with TP=8, each of the eight GPUs operates on its respective shard of that layer's weight matrices.

Here, we return to the Roofline model: fetching weights from HBM, staging and reusing them in shared memory, and keeping Tensor Cores saturated at peak arithmetic intensity represent the culmination of single-GPU performance principles in active execution.

Eight H100 GPUs collectively compute the massive projection matrices within a single layer. Each GPU maintains its designated weight shard. As GEMM and attention kernels run across SMs, weights are streamed from local HBM through the cache hierarchy.

At this stage, we must differentiate between two types of work:

1. **Useful compute:** The actual matrix multiplications required by the mathematical architecture of the layer.
2. **Execution overhead:** Kernel launch overhead, small tensor ops, synchronization barriers, layout transformations, and precision casting.

The latter keeps the GPU busy, but contributes zero useful arithmetic toward theoretical MFU.

### Step 3 — The Necessary Collective Communications Are Executed

For example, partial sums produced by row-parallel projections must be reduced across GPUs via an AllReduce. During this phase, **NCCL (NVIDIA Collective Communications Library)** coordinates data movement using the fastest available hardware topology and collective communication algorithms.

Ideally, communication is overlapped with independent computation already in flight. However, when the next layer's input directly depends on the collective output, execution must stall until that dependency barrier clears.

An estimate of the volume: in Llama 3 405B the activation of an 8,192-token microbatch is ~268 MB, so one TP AllReduce on NVLink takes roughly $2\cdot\frac{7}{8}\cdot268\ \mathrm{MB}/450\ \mathrm{GB/s}\approx1$ ms. A pipeline stage holds about 8 layers (126/16); with 2 AllReduces per layer in the forward pass, 2 in the backward pass and 16 microbatches per DP replica, a step issues ~500 of them, about **0.5 s** of NVLink communication. The more of it overlaps with compute, the less it shows up in MFU.

### Step 4 — The Pipeline Stage Passes Activations to the Next Stage

When Stage 0 finishes computing its partition of layers, it transfers the resulting boundary activations to Stage 1. Unlike tensor parallelism, which requires intra-layer synchronization, pipeline parallelism only transmits the **boundary activation tensor** across stage partitions.

A **microbatch** is a small sub-slice of the global batch scheduled through the pipeline bubbles. Maintaining a sufficiently high ratio of microbatches to pipeline stages helps keep pipeline stages saturated.

### Step 5 — Loss and Backward Pass

At the end of the forward pass, the model emits its predictions. But training is more than inference; we must determine how every parameter should adjust to minimize prediction error. Propagating backward through the loss scalar, we compute gradients via reverse-mode automatic differentiation.

This phase places intense pressure on GPU memory because backward kernels require intermediate activations retained from the forward pass. Retaining every intermediate activation risks exhausting HBM; discarding too many requires re-evaluating forward operations during the backward pass. **Activation recomputation** represents precisely this compute-versus-memory trade-off.

The final pipeline stage computes the loss value. Gradients are then propagated in reverse order using the chain rule. The **backward pass** is memory-intensive precisely because it consumes these cached activations to compute derivatives.

**Activation checkpointing / recomputation** is the technique of discarding selected intermediate activations during the forward pass and recomputing them on-the-fly during backward execution. It must not be confused with a training checkpoint on persistent disk:

| Term | Primary Purpose |
|---|---|
| **Activation checkpointing** | Reclaims GPU HBM capacity by trading redundant compute for activation footprint |
| **Training checkpoint** | Persists global model and optimizer state to disk to survive cluster node failures |

### Step 6 — Gradients Are Synchronized

Gradient reduction and sharding routines execute across DP/FSDP process groups. Gradient synchronization ensures that disparate data-parallel workers reconcile their parameter updates before advancing optimizer state.

A critical determinant of training throughput is the temporal alignment between **when a gradient tensor is produced** and **when the collective communication must complete**. Overlapping gradient AllReduce/ReduceScatter kernels with ongoing backward compute hides interconnect latency and eliminates stalls.

At Llama 3 405B scale: with TP = 8 × PP = 16 each GPU is responsible for a slice of about $405\times10^{9}/128\approx3.2$ billion parameters. FSDP moves that slice's FP32 gradients with a reduce-scatter (~12.7 GB) and its BF16 weights with an all-gather (~6.3 GB): ~19 GB in total, about **0.38 s** on a 50 GB/s NIC per GPU. To the extent it overlaps with backward, it never reaches the step time.

### Step 7 — The Optimizer Updates Weights

The AdamW optimizer updates parameter weights using current gradients alongside first and second momentum tracking vectors. **Weight decay** applies an $L_2$-style penalty directly to parameter magnitudes to regularize training.

In distributed optimizer strategies such as ZeRO-1 / FSDP, optimizer states are sharded across the DP domain. This eliminates the need to replicate redundant optimizer tensors on every single device.

### Step 8 — Checkpoints Are Periodically Written

The step concludes, and the optimizer updates the model weights. But what happens if several cluster nodes fail immediately after? Without a durable persistent record, millions of dollars in compute time could be lost. We would be forced to rewind to where we last saved.

This is why training state is persisted periodically. In frontier clusters, a checkpoint is far more sophisticated than a naive `.pt` file dump; when global state scales to dozens of terabytes, checkpointing is a core component of distributed storage and disaster recovery architecture.

Model weights, optimizer moments, RNG states, learning rate schedules, and step metadata are serialized and committed to persistent storage according to the chosen parallel I/O strategy.

This operation may block all GPUs entirely; alternatively, **asynchronous checkpointing** can take an in-memory snapshot of the state and stream it to persistent storage in the background while GPUs immediately resume training.



<svg viewBox="0 0 720 342" role="img" aria-label="Swim-lane view of one training step across five places: storage, CPU and host RAM, GPU HBM and SMs, NVLink, and the network fabric. 1, the dataloader reads tokens from storage on the CPU side. 2, forward GEMMs run on the GPU. 3, tensor-parallel all-reduce runs over NVLink. 4, pipeline activations cross the fabric. 5, loss and backward run on the GPU. 6, data-parallel or FSDP gradient synchronization crosses the fabric. 7, the optimizer updates weights on the GPU, and the loop returns to forward for the next step. 8, a checkpoint is periodically written to storage. Steps 2 to 6 repeat per microbatch, and 3 and 6 can partly overlap with compute." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ts-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ts-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="ts-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">One training step: where each phase physically happens</text>
<rect x="16" y="44" width="688" height="46" rx="6" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="26" y="71" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Storage</text>
<rect x="16" y="96" width="688" height="46" rx="6" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="26" y="123" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">CPU + host RAM</text>
<rect x="16" y="148" width="688" height="46" rx="6" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="26" y="175" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU: HBM + SM</text>
<rect x="16" y="200" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.05;stroke:var(--c-success);stroke-width:1.2"/>
<text x="26" y="227" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NVLink (TP)</text>
<rect x="16" y="252" width="688" height="46" rx="6" style="fill:var(--c-warn);fill-opacity:.05;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="26" y="279" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Fabric (PP, DP)</text>
<rect x="131" y="100" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="164.0" y="115" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">read</text>
<text x="164.0" y="129" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">1 · dataloader</text>
<rect x="203" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">forward</text>
<text x="236.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">2 · GEMM</text>
<rect x="275" y="204" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="308.0" y="219" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">TP</text>
<text x="308.0" y="233" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">3 · all-reduce</text>
<rect x="347" y="256" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.0" y="271" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">PP</text>
<text x="380.0" y="285" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">4 · activations</text>
<rect x="419" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="452.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">backward</text>
<text x="452.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">5 · + loss</text>
<rect x="491" y="256" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="524.0" y="271" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">DP/FSDP</text>
<text x="524.0" y="285" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">6 · grad sync</text>
<rect x="563" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="596.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">optimizer</text>
<text x="596.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">7 · update</text>
<rect x="635" y="48" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="668.0" y="63" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">checkpoint</text>
<text x="668.0" y="77" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">8 · periodic</text>
<path d="M196.0 119.0 H236.0 V150" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M268.0 171.0 H308.0 V202" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M340.0 223.0 H380.0 V254" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M412.0 275.0 H452.0 V192" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M484.0 171.0 H524.0 V254" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M556.0 275.0 H596.0 V192" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M628.0 164 H668.0 V88" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M122 66 H138 V98" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M596.0 192 V310 H236.0 V192" marker-end="url(#ts-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Steps 2–6 repeat per microbatch; 3 and 6 can partly overlap compute. Step 8 runs every N steps (sync or async).</text>
</svg>



### 14.1. Memory Accounting: The Origin of Numbers for a 405B Model

Consider the baseline training state required for a model of this magnitude:

<svg viewBox="0 0 720 294" role="img" aria-label="Stacked bar of the training state of a 405B model, 16 bytes per parameter: BF16 weights 2 bytes or 0.81 TB, BF16 gradients 2 bytes or 0.81 TB, Adam first and second moments in FP32 at 4 bytes or 1.62 TB each, and an optional FP32 master copy at 4 bytes or 1.62 TB, totalling 6.48 TB. Per GPU, split ideally across 128 GPUs by TP 8 times PP 16, that is 50.6 GB; if the optimizer state is also sharded over DP 128, about 12.96 GB, both under an 80 GB HBM line but before activations, buffers and fragmentation." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="m4-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Training state of a 405B model: 16 bytes per parameter</text>
<rect x="17" y="40" width="84" height="54" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<text x="59.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">BF16 weights</text>
<text x="59.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2 B/param</text>
<text x="59.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.81 TB</text>
<rect x="103" y="40" width="84" height="54" rx="2" style="fill:var(--c-success);fill-opacity:.45"/>
<text x="145.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">BF16 grads</text>
<text x="145.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2 B/param</text>
<text x="145.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.81 TB</text>
<rect x="189" y="40" width="170" height="54" rx="2" style="fill:var(--c-warn);fill-opacity:.38"/>
<text x="274.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Adam m (FP32)</text>
<text x="274.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="274.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.62 TB</text>
<rect x="361" y="40" width="170" height="54" rx="2" style="fill:var(--c-warn);fill-opacity:.38"/>
<text x="446.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Adam v (FP32)</text>
<text x="446.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="446.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.62 TB</text>
<rect x="533" y="40" width="170" height="54" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45"/>
<text x="618.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">FP32 master (optional)</text>
<text x="618.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="618.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.62 TB</text>
<text x="704" y="112" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-weight:700">total 16 B × 405 B params = 6.48 TB</text>
<text x="16" y="140" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Per GPU, before activations and buffers:</text>
<text x="240" y="171" text-anchor="end" style="fill:var(--c-text);font-size:11.5px">÷ 128 (TP 8 × PP 16)</text>
<rect x="250" y="158" width="238.118" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="494.11764705882354" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">50.6 GB</text>
<text x="240" y="207" text-anchor="end" style="fill:var(--c-text);font-size:11.5px">+ optimizer sharded over DP 128</text>
<rect x="250" y="194" width="60.9882" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="316.98823529411766" y="208" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">≈ 12.96 GB</text>
<line x1="626.471" y1="148" x2="626.471" y2="226" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="626.4705882352941" y="240" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">80 GB HBM</text>
<text x="16" y="266" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Idealized split: AllGather buffers, layer imbalance, activations and fragmentation come on top;</text>
<text x="16" y="282" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">FSDP guarantees no exact figure.</text>
</svg>

**Important:** A distinct FP32 copy of master weights is not mandatory in every mixed-precision training recipe. Furthermore, gradients do not need to be serialized into persistent checkpoints. This figure models *in-memory training state requirements*, not static checkpoint file sizes on disk.

If TP=8 and PP=16 could partition model state across $8 \times 16 = 128$ GPUs with ideal balance, the gross baseline model state per GPU would be:

$$
\frac{6.48\ \mathrm{TB}}{128}=50.625\ \mathrm{GB/GPU}
$$

If replicated optimizer states are further sharded across the DP=128 domain (ZeRO-1 / FSDP), this baseline footprint drops significantly. However, transient AllGather buffers, inter-layer imbalances, activation footprints, and memory allocator fragmentation add substantial overhead on top of this figure. We cannot claim **'FSDP strictly guarantees 12.9 GB'**. The following breakdown is merely an idealized illustrative example:

- Weights and gradients replicated across DP post TP×PP: approximately $6.33 + 6.33$ GB.
- Optimizer master + momentum state: approximately $37.97$ GB; sharded across 128 DP replicas, this becomes $0.30$ GB.
- Rough baseline: **approximately 12.96 GB/GPU**, *excluding activations and transient communication buffers*.

Real-world FSDP gradient sharding behavior and transient buffer staging alter the actual resident footprint. The fundamental lesson is that **HBM capacity and network communication are constantly traded off against one another**.

### 14.2. Sanity Check Calculation for Step Time

A standard first-order estimate of total compute required for dense Transformer training is $6PT$ FLOPs, where $P$ is the parameter count and $T$ is the number of tokens processed (attention operations and auxiliary kernels introduce second-order deviations).

$$
F_{\mathrm{step}}\approx6\times405\times10^9\times16\times10^6
=3.888\times10^{19}\ \mathrm{FLOP}
$$

Under the setup documented in Meta's report—16,384 H100 GPUs operating at roughly 400 TFLOP/s sustained per GPU with 41% MFU—the effective aggregate model compute rate is approximately $6.55\ \mathrm{EFLOP/s}$. Plugging this into our simplified model yields:

$$
T_{\mathrm{step}}\approx\frac{38.88\ \mathrm{EFLOP}}{6.55\ \mathrm{EFLOP/s}}
\approx5.9\ \mathrm{seconds}
$$

This is not an empirical claim that 'Meta executed every step in exactly 5.9 seconds.' It is an **approximate engineering sanity check** confirming the consistency between reported cluster throughput and standard $6PT$ scaling formulations. [K5]

---


### 14.3. The Unseen Side of the Data Pipeline: Quality, Provenance, and Reproducibility

When examining the training step, we saw token IDs read from storage. But for an AI Architect, the real question begins one step earlier: **Where did these tokens originate, how were they filtered and sanitized, under what exact version were they ingested, and can we reproduce this exact training run six months from now?**

Imagine a manufacturing facility running at lightning speed, but receiving raw materials that are contaminated or mislabeled. The plant's hourly production volume may look extraordinary on dashboards, but the finished product is defective. The relationship between high MFU and poor-quality training data follows this exact dynamic.

A production-grade **data pipeline** must systematically manage the following stages:



<svg viewBox="0 0 720 182" role="img" aria-label="A production data pipeline in six stages: source records of raw text; access and rights checks for licence and consent; quality filtering and semantic deduplication; the tokenizer with packing; shards with a manifest; and GPU training through the dataloader. Under each stage is the record it must leave: source system, rights decision, filter version, tokenizer and vocabulary, shard hashes, and seed, commit and container image. Together they form the lineage that lets the same run be rebuilt months later." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dpp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">A production data pipeline — and what each stage must leave behind</text>
<rect x="16" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="69.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Source records</text>
<text x="69.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">raw text</text>
<rect x="132" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="185.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Access &amp; rights</text>
<text x="185.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">license, consent</text>
<line x1="122" y1="69" x2="131" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="248" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="301.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Quality &amp; dedup</text>
<text x="301.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">filters, semantic</text>
<line x1="238" y1="69" x2="247" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="417.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Tokenizer</text>
<text x="417.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">packing</text>
<line x1="354" y1="69" x2="363" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="480" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="533.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Shards</text>
<text x="533.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">manifest</text>
<line x1="470" y1="69" x2="479" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="596" y="42" width="106" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="649.0" y="66.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU training</text>
<text x="649.0" y="81.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">dataloader</text>
<line x1="586" y1="69" x2="595" y2="69" marker-end="url(#dpp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="69" y1="96" x2="69" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="20" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="69.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">source system</text>
<line x1="185" y1="96" x2="185" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="136" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="185.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">rights decision</text>
<line x1="301" y1="96" x2="301" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="252" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="301.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">filter version</text>
<line x1="417" y1="96" x2="417" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="368" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="417.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">tokenizer + vocab</text>
<line x1="533" y1="96" x2="533" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="484" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="533.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">shard hashes</text>
<line x1="649" y1="96" x2="649" y2="116" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="600" y="116" width="98" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.1;stroke:var(--c-success);stroke-width:1.2"/>
<text x="649.0" y="131.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">seed, code, image</text>
<path d="M16 152 H704" style="fill:none;stroke:var(--c-success);stroke-width:1.6"/>
<text x="360" y="170" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">lineage: which bytes, which transforms, which code → the same run can be rebuilt months later</text>
</svg>



**Data provenance** is the ability to trace data back to its source system, documenting how it was collected and every processing stage it underwent. **Data lineage** tracks how specific training outputs derive from particular dataset versions and transformational pipelines.

**Deduplication** removes identical or near-duplicate documents across corpora. Why is this critical? If the same text is repeated millions of times, the model overfits to specific syntactic patterns, underlying token distributions warp, and subtle evaluation contamination occurs between training and benchmark splits. Meta's technical disclosures on Llama 3 data preparation—highlighting heuristic filtering, semantic deduplication, and model-based quality classifiers—demonstrate the immense engineering scale required here. [K28]

**Determinism and reproducibility** denote the ability to re-execute a training job and achieve statistically congruent convergence trajectories. While bit-level identical execution across thousands of non-deterministic GPU reduction kernels and dynamic networks is often impractical, systems must persistently record the following provenance artifacts:

| Tracked Artifact | Engineering Purpose |
|---|---|
| Dataset manifests and shard hashes | Verifies exact raw data bytes consumed |
| Tokenizer version and vocabulary | Prevents text-to-token drift across runs |
| Train/validation splits and sampling seeds | Ensures rigorous, uncorrupted evaluation |
| Architecture specs, optimizer configs, precision, global batch size | Enables exact structural and hyperparameter reproduction |
| Git commit SHA, container image digest, CUDA/NCCL versions | Locks the software and driver environment |
| Checkpoint ID and evaluation reports | Provides end-to-end lineage from artifact to loss metric |

NVIDIA Megatron-LM's indexed dataset approach (using `.bin` and `.idx` pairs) combined with versioned training recipes exemplifies this paradigm in practice. However, format selection must always be tuned to target storage topologies, dataset footprints, and versioning architectures. [K29]

> **Architectural Decision:** Optimizing GPU MFU without exercising equivalent rigor over dataset hygiene is like speeding up an assembly line with defective raw steel. **Quality-adjusted throughput** is vastly more meaningful than raw tokens/sec.

### 14.4. Scaling Laws: How Many Parameters and How Many Tokens Should We Train?

Before sizing GPU fleets, architects must resolve a foundational question: What should the parameter scale be, how many tokens must the model consume, and what target capability must it hit?

Frame this as a resource allocation problem. Given a fixed FLOP budget, should you train a larger model on fewer tokens, or a smaller model on substantially more data? The **Chinchilla compute-optimal training** paradigm addressed this directly. Hoffman et al. showed that under standard pretraining assumptions, model size and token volume should be scaled in roughly equal proportions to maximize loss efficiency for a given compute budget. However, this does not dictate a rigid, universal ratio for all models—especially when inference economics favor over-training smaller checkpoints. [K30]

For dense Transformer pretraining, the standard first-order compute formula is:

$$
C_{\mathrm{train}}\approx6ND
$$

- $N$: Parameter count.
- $D$: Total training tokens consumed.
- $C_{\mathrm{train}}$: Total floating-point operations (FLOPs).

For instance, training a 70B parameter model on 1 trillion ($1\times10^{12}$) tokens requires:

$$
C\approx6\times70\times10^9\times10^{12}
=4.2\times10^{23}\ \mathrm{FLOP}
$$

This is an **idealized first-order baseline**. Quadratic attention scaling at extended sequence lengths, embedding projections, activation recomputation, padding waste, and sparse MoE routing dynamics all shift the true arithmetic total. Furthermore, this formula measures compute expenditure, not downstream task capability.

We can link this computational demand directly to hardware provisioning:

$$
T_{\mathrm{idealized}}\approx\frac{C_{\mathrm{train}}}
{N_{\mathrm{GPU}}\times F_{\mathrm{peak}}\times\mathrm{MFU}}
$$

This represents a **purely compute-bound duration estimate**. Wall-clock training duration must account for data ingestion bottlenecks, periodic evaluation passes, checkpoint I/O, unexpected hardware failures, straggler mitigation, and cluster restarts. Furthermore, sustained MFU fluctuates over time rather than remaining static.

**Why does this matter?** Before leadership commits hardware capital or promises delivery milestones based on raw GPU counts, architects must baseline target model capability against token budgets and compute scaling. Otherwise, teams face catastrophic surprises—such as building massive parameter architectures that starve from data deficits.

### 14.5. Fast Training or Correct Training? Mixed Precision and Convergence

Suppose a cluster pushes throughput so aggressively that a model reaches a benchmark loss in 7 days instead of 10. That looks like a massive operational win. But what if that same setup causes frequent numerical overflows, gradient clipping spikes, and ultimately yields inferior downstream zero-shot capability? How should engineering success be measured?

**Convergence** describes the trajectory of a model as its optimization step progresses toward an empirical loss minimum. **Validation loss** and task-specific benchmarks ensure that engineering teams monitor true semantic learning rather than merely raw hardware throughput.

When navigating FP8, BF16, and mixed-precision schemes, architects must rigorously track:

- **Scaling and calibration:** How are high-dynamic-range tensors quantized and scaled into narrow 8-bit representations?
- **Overflow and underflow:** Do activation extremes exceed maximum representable values or vanish beneath the lower precision floor?
- **Gradient norm dynamics and loss spikes:** Does numerical instability trigger loss divergence or catastrophic gradient clipping?
- **Per-layer precision exclusions:** Must sensitive architectural components (such as attention logits, normalization layers, or embedding tables) be preserved in higher precision (e.g., FP32 or BF16)?
- **Time-to-quality:** How many wall-clock hours and GPU-hours were consumed to achieve a predefined validation loss threshold?

Benchmarking must never rely solely on raw `tokens/sec`. The only rigorous metric is **time-to-quality: wall-clock duration to reach a specified capability target over identical datasets, with matched hyperparameters and reproducible seeds**. This is why the MLPerf Training standard mandates convergence thresholds alongside execution runtimes. [K31]

---



---

### Reading a Training Step Like an Accounting Ledger

Engineering discussions often reduce training to "GPUs crunching matrix multiplications." Yet over the course of a single step, hardware does far more: it **ingests data**, **stages and caches activations**, **derives analytical gradients**, **exchanges parameters across interconnects**, **updates optimizer moments**, and **commits checkpoints to persistent storage**.

Maintaining an architectural ledger clarifies the true resource economics across each phase:

| Resource Subsystem | During Forward Pass | During Backward Pass | During Optimizer Step |
|---|---|---|---|
| Tensor Cores | Attention and MLP projections | Derivative GEMMs (data and weight gradients) | Negligible (predominantly element-wise operations) |
| HBM | Weight streaming, activation materialization | Gradient computation, activation playback | Parameter and optimizer state reads/writes |
| Interconnect Network | TP/EP/PP transfers, FSDP parameter AllGather | TP/DP/FSDP ReduceScatter and AllReduce | Distributed state reduction (strategy dependent) |
| Storage Subsystem | Batch streaming; occasional checkpoint restore | Minimal (indirect) | Periodic checkpoint serialization and writes |

This matrix illustrates **why optimizing one dimension frequently exacerbates another**. Selective activation recomputation slashes HBM pressure, but incurs redundant arithmetic passes. FSDP eliminates redundant model replicas across memory, but injects AllGather communication latency before every forward layer. Shrinking microbatch sizes relieves activation memory, but degrades GEMM arithmetic intensity and Tensor Core efficiency. [K48]

### Parameter Size Is Not the Entire Training State

A 70B parameter model stored in BF16 weights occupies approximately **140 GB**. In pretraining, however, this represents only the tip of the memory iceberg. BF16 gradients demand another 140 GB. Two FP32 Adam momentum tensors require $2 \times 4 \times 70\times10^9 = 560\ \text{GB}$. Maintaining an FP32 master weight replica adds another 280 GB. Summing these static components yields **1.12 TB**—and this excludes dynamic activation tensors, CUDA runtime buffers, and collective communication scratchpads.

This aggregate figure does not mean a single GPU must house 1.12 TB. Through TP, PP, DP, and particularly ZeRO/FSDP sharding, these tensors are partitioned across cluster memory. The foundational intuition remains: *Parameter count reflects static model size; per-GPU resident memory reflects how effectively state is partitioned across distributed memory domains.*

### Properly Understanding the "6ND FLOP" Rule of Thumb

The $6ND$ formulation established in Section 14.4 stems from allocating roughly $2N$ FLOPs per token during the forward pass and $4N$ FLOPs for input and parameter gradient derivations in the backward pass. However, this rule represents an idealized lower bound rather than an absolute physical law.

In production pretraining pipelines, empirical compute deviates from this baseline across three primary dimensions:
1. **Long-context quadratic scaling ($O(S^2)$):** In standard self-attention, computational intensity scales quadratically with sequence length; across 32k+ token contexts, attention FLOPs begin to dominate the first-order linear approximation.
2. **Activation recomputation penalty:** When intermediate activations are discarded to conserve HBM and recomputed on-the-fly during backward execution (selective recomputation), arithmetic demand expands by approximately $+2ND$, pushing total workload toward $8ND$.
3. **Sparse MoE routing dynamics:** In Mixture-of-Experts architectures where tokens route dynamically to a restricted top-$k$ subset of experts, arithmetic budgeting must evaluate active parameters per token ($N_{\mathrm{active}}$) rather than total static parameter scale. [K30]


## 15. Anatomy of Distributed Training Costs: Compute, Data, Optimizer, and Quality

Partitioning a model across thousands of GPUs is like staffing a high-rise construction project: without meticulous workflow scheduling, ten specialized teams end up colliding in the same narrow corridor. A distributed training architecture must resolve four fundamental questions simultaneously: Does the model state fit in memory? Does each GPU receive sufficient arithmetic intensity? Does inter-node communication stall execution? And has the distributed scaling strategy altered convergence behavior?

### 15.1. How Is the Global Batch Composed?

Let $D$ denote data-parallel group size, $m$ the microbatch size fed into each GPU pipeline stage, and $a$ the number of gradient accumulation steps executed prior to an optimizer update. In standard data-parallel training:

$$
B_{\mathrm{global}}=D\times m\times a
$$

For instance, if $D=16$, $m=2$, and $a=8$, exactly 256 samples are processed per optimizer step. However, because sequence lengths vary in LLM pretraining, architects must track **global tokens per update** rather than naive sample counts. Unless fixed-length sequence packing is strictly enforced, equal sample counts do not translate to identical token counts.

### 15.2. How to Estimate Transformer Training FLOPs

For dense decoder-only models, the canonical estimate remains:

$$
F_{\mathrm{train}}\approx 6ND
$$

where $N$ is parameter count and $D$ is the total number of training tokens processed. The factor of 6 represents the first-order sum of forward ($2N$) and backward ($4N$) matrix operations per token. Adjustments must be introduced for sequence length growth in attention, vocabulary embeddings, activation recomputation, optimizer overhead, and custom architectures. For Mixture-of-Experts (MoE) architectures, substituting total parameters directly for $N$ is invalid; calculations must be driven by **active parameters per token and actual expert routing compute**.

For a 1B parameter model trained across 10B tokens, the baseline calculation is $6\times10^9\times10^{10}=6\times10^{19}$ FLOPs. This establishes the baseline compute requirement. Translating this into wall-clock duration requires factoring in hardware scale and efficiency:

$$
T_{\mathrm{idealized}}\approx\frac{F_{\mathrm{train}}}{N_{\mathrm{GPU}} C_{\mathrm{peak}} \cdot MFU}
$$

When applying this formula, **$C_{\mathrm{peak}}$ must align with the exact numerical precision and Tensor Core mode assumed in the MFU baseline**. Furthermore, checkpoint I/O pauses, evaluation runs, data loading latency, node failures, and warmup phases must be added to the project timeline.

### 15.3. How to Conduct Memory Budgeting for Training

Resident memory consists of distinct line items: FP16/BF16 weights, gradients, optimizer momentum tensors, optional FP32 master weight copies, intermediate activations, transient operator buffers, collective communication buckets, and framework allocator fragmentation pools. Distributed parallelism strategies shard varying combinations of these categories. Rather than assuming whether an optimizer unconditionally maintains FP32 master weights, verify the specific framework implementation and precision configuration.

**Practical Workflow:** Profile memory consumption on a **single model-parallel group** configured with your maximum sequence length and target microbatch size before scaling out data parallelism. Do not assume that expanding the data-parallel domain will magically shrink per-GPU activation footprints: activation memory is governed by local microbatch sizes and sequence length, not by global DP scale.

### 15.4. How Attention Costs Should Be Evaluated

In standard full self-attention, computational complexity and attention matrix footprints scale as $O(S^2)$ with sequence length $S$. FlashAttention mitigates memory bandwidth bottlenecks by fusing softmax tiling and avoiding full materialization of the $S \times S$ matrix in HBM; however, it does not magically transform the underlying quadratic mathematical complexity into a linear operation.

Context Parallelism (CP) shards ultra-long sequence dimensions across multiple GPUs; however, inter-GPU KV tensor exchange introduces communication overhead. Consequently, there is no simple 1:1 relationship where 'doubling the context window doubles training cost'—the real cost profile depends heavily on sequence length regimes, tiling efficiency, and interconnect topology.

### 15.5. Break Down the Step Time into Phases



<svg viewBox="0 0 720 252" role="img" aria-label="Timeline of step-time phases. On the critical path: forward, backward, an exposed slice of communication, the optimizer, then the next step. Beneath it, background work: gradient AllReduce buckets overlap the later part of backward, data prefetch loads the next batch while compute runs, and an asynchronous checkpoint persists state after a short snapshot pause. Whatever communication does not fit under backward lands on the critical path." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="sp-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Step-time phases: the critical path and what can hide beside it</text>
<text x="16" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">critical path</text>
<rect x="250" y="46" width="135" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.5" y="64.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">forward</text>
<rect x="385" y="46" width="180" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="475.0" y="64.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">backward</text>
<rect x="565" y="46" width="45" height="28" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="587.5" y="63.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">exposed</text>
<rect x="610" y="46" width="45" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="632.5" y="63.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">optim.</text>
<rect x="655" y="46" width="49.5" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="679.75" y="63.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">next</text>
<text x="16" y="103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">grad AllReduce</text>
<text x="16" y="118" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">buckets overlap late backward</text>
<rect x="475" y="92" width="135" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="16" y="147" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">data prefetch</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">next batch while computing</text>
<rect x="272.5" y="136" width="180" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="16" y="191" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">async checkpoint</text>
<text x="16" y="206" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">persist after a short snapshot</text>
<rect x="637" y="180" width="67.5" height="24" rx="4" style="fill:var(--c-danger);fill-opacity:.22;stroke:var(--c-danger);stroke-width:1.2"/>
<line x1="565" y1="40" x2="565" y2="226" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<text x="704" y="240" text-anchor="end" style="fill:var(--c-danger);font-size:11px">communication that did not fit under backward lands on the critical path</text>
</svg>



Map this mental model directly against your profiler traces (e.g., PyTorch Profiler or NVIDIA Nsight Systems). If GPUs exhibit extended idle phases following the backward pass waiting on collective barriers, re-evaluate your network fabric topology or communication overlap strategy. If kernel execution gaps correlate with DataLoader stalls, investigate your host I/O pipeline. However, attempting to overlap every concurrent operation indiscriminately does not guarantee net gains: speculative memory reservation, CUDA stream prioritization conflicts, and driver contention introduce their own performance penalties.

### 15.6. Is the Loss Decreasing? The Scientific Validation of Speedups

Gaining 10% in MFU means nothing commercially if model convergence degrades. Whenever precision regimes, kernel fusion, or optimizer configurations change, teams must monitor **loss curves**, gradient $L_2$ norms, NaN/Inf occurrences, gradient clipping frequencies, learning rate dynamics, validation cross-entropy, and downstream task evaluations.

Migrating from BF16 to FP8 dramatically accelerates token processing throughput, but critical layers may destabilize without selective precision elevation. Following changes to activation checkpointing, optimizer offloading, or fused kernels, engineering reports must prioritize **time-to-quality** over naive token throughput. [K31][K41]

### 15.7. When Do LoRA and PEFT Architectures Require Distinct System Design?

In place of full fine-tuning, **LoRA (Low-Rank Adaptation)** freezes baseline pre-trained weights and injects trainable low-rank decomposition matrices into target layers. This technique slashes trainable parameter counts and optimizer state footprints; however, the forward pass still requires evaluating the full underlying foundation model. In multi-tenant serving, dynamic adapter loading, version pinning, adapter cache management, and multi-tenant security isolation introduce entirely new architectural requirements.

**Decision Criterion:** Does the downstream task genuinely require full fine-tuning, or can parameter-efficient fine-tuning (PEFT) achieve the necessary capability threshold at a fraction of the compute and storage footprint? The answer cannot be deduced from parameter counts alone; it must emerge from **rigorous task-specific benchmarking**.

---


---

**PART V — Data, Checkpointing, and Reliability**

*Why is resilience as crucial as throughput? How a single save operation or a node failure alters cluster economics.*


## 16. The Storage Wall: Your GPU May Not Be Computing, but Waiting on the File System

Focusing solely on GPU arithmetic intensity blinds engineers to an extraordinarily costly bottleneck: **the mechanics of streaming training data to GPUs and reliably checkpointing distributed cluster state**.

Reading training data and writing checkpoints are not symmetric operations. Data ingestion involves continuous, steady streams of small to medium-sized sample batches across thousands of worker threads. Checkpoint writes, by contrast, unleash an explosive, simultaneous tidal wave of multi-terabyte state dumps across the entire cluster.

The difference in storage engineering is identical to the contrast between maintaining a daily delivery fleet across a city versus having every resident in town attempt to move their furniture at the exact same second.

### 16.1. Why Reads and Writes Pose Fundamentally Different Challenges

During pretraining ingestion, relatively small sample slices are requested with extreme frequency. During checkpointing, massive volumes of distributed state are flushed to persistent tiers within a tight time window.

Consequently, the underlying storage fabric must reconcile two contradictory operational profiles:

1. **Steady-state reads:** Predictable, low-latency, uninterrupted streaming of token batches to feed GPU HBM.
2. **Burst writes:** The ability to ingest dozens of terabytes of concurrent writes across thousands of storage targets without saturating metadata servers or stalling the network fabric.

### 16.2. Why the Linux Page Cache Is Sometimes a Blessing, Sometimes a Bottleneck

When applications read files under Linux, the operating system caches read blocks in system RAM to accelerate subsequent accesses. For general applications, this is an outstanding optimization. But when continuously streaming massive datasets to GPUs, copying data into host CPU memory only to re-copy it over PCIe/NVLink into GPU HBM introduces redundant memory hops and CPU cache thrashing.

Does this mean the Page Cache is inherently flawed? Not at all. The utility of an optimization depends strictly on the access pattern. The problem is not the cache itself, but the superfluous intermediate memory copies incurred when forcing GPU-bound workloads through standard OS file paths.

The **Linux Page Cache** buffers file pages in host DRAM. For workloads that repeatedly access the same files, it drastically reduces physical disk I/O. In high-bandwidth GPU data ingestion, however, staging file contents into CPU RAM before copying them across PCIe to GPU HBM consumes system memory bandwidth and introduces latency jitter.

Therefore, the Page Cache is **not inherently an enemy of performance**; its efficiency is governed entirely by working set size and data movement topology.

On the classic path a byte usually makes four stops on its way to the GPU: (1) from NVMe over PCIe to the host, (2) into the kernel's page cache, (3) into the application's user-space bounce buffer, and (4) over PCIe again with an H2D copy into HBM. The data crosses PCIe twice and exists twice in CPU memory. For a few large reads that is harmless; for terabytes of checkpoint it ties up the CPU's memory bus and cores. GDS in the next section aims precisely at skipping stops 2 and 3. [K6]

### 16.3. Situating mmap, the Python GIL, and NUMA in Proper Context

**`mmap`** maps files directly into a process's virtual address space, allowing files to be accessed via pointer arithmetic and page-faulted on demand.

The **Python GIL (Global Interpreter Lock)** restricts concurrent execution of Python bytecode within a single CPython process. `mmap` alone does not circumvent GIL contention; it merely simplifies zero-copy address manipulation. Mitigating GIL bottlenecks requires multi-process DataLoaders, C/C++ worker extensions, asynchronous I/O architectures, and coarse-grained batching.

**NUMA-aware placement** pins DataLoader worker threads, memory allocations, PCIe lanes, and NIC endpoints to the specific CPU socket directly wired to the target GPU, eliminating cross-socket QPI/UPI interconnect penalties.

Production dataloaders combine all three. Megatron-style frameworks write the dataset once as `uint16`/`int32` token IDs in a single contiguous `.bin` file, with document boundaries in a separate `.idx` file; workers open it with `mmap`, so no text parsing or Python object creation happens during training [K29]. On a DGX H100, CPU 0 feeds GPUs 0–3 and CPU 1 feeds GPUs 4–7; pinning the workers that feed GPU $i$ to cores on the same socket and using pinned memory keeps the data from crossing UPI to the other socket and back.

### 16.4. What Path Does GPUDirect Storage (GDS) Actually Alter?

Imagine routing raw inventory straight to the assembly floor via automated conveyors, instead of hauling it to an intermediate corporate warehouse, unpacking it, and reloading it onto secondary trucks. You eliminate handling overhead and turnaround delay.

This captures the core intuition behind GPUDirect Storage (GDS). On supported hardware fabrics, GDS streams data directly between NVMe storage and GPU HBM via DMA, bypassing intermediate bounce buffers in CPU host memory. But claiming "the CPU is entirely bypassed" is inaccurate: control plane orchestration, NVMe drivers, and filesystem metadata management still reside on the host CPU.



<svg viewBox="0 0 720 308" role="img" aria-label="Two data paths from storage to GPU HBM. Classic: data is copied from storage into a CPU RAM bounce buffer, then copied again over PCIe into HBM. GPUDirect Storage: data moves by DMA over PCIe or NVMe-oF straight into HBM through cuFile, while the CPU only handles control. GDS removes a copy, not the storage limit: a 25 GB/s target stays 25 GB/s, and compatibility mode can fall back to the CPU path." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="gd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="gd-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Classic bounce-buffer path vs. GPUDirect Storage</text>
<text x="16" y="56" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">1 · classic: copy through CPU RAM</text>
<rect x="16" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Storage</text>
<text x="86.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVMe / parallel FS</text>
<rect x="564" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="634.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM3</text>
<text x="634.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="290" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU RAM</text>
<text x="360.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bounce buffer</text>
<line x1="156" y1="92" x2="288" y2="92" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="430" y1="92" x2="562" y2="92" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="222" y="86" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">copy 1</text>
<text x="496" y="86" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">copy 2 (PCIe)</text>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">2 · GDS: direct DMA into HBM</text>
<rect x="16" y="182" width="140" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="204.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Storage</text>
<text x="86.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVMe / parallel FS</text>
<rect x="564" y="182" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="634.0" y="204.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM3</text>
<text x="634.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<line x1="156" y1="206" x2="562" y2="206" marker-end="url(#gd-o)" style="stroke:var(--c-success);stroke-width:2"/>
<text x="360" y="200" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">DMA over PCIe / NVMe-oF (cuFile)</text>
<rect x="290" y="240" width="140" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="360.0" y="261.75" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">CPU: control only</text>
<line x1="360" y1="240" x2="360" y2="212" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="16" y="296" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GDS removes a copy, not the storage limit: a 25 GB/s target stays 25 GB/s. Compatibility mode can fall back to the CPU path.</text>
</svg>



**GPUDirect Storage (GDS)** establishes a direct DMA path between local NVMe drives or remote storage fabrics (NVMe-oF) and GPU memory using the `cuFile` API, provided compliant hardware, filesystem drivers, and kernel configurations are present. This eliminates CPU bounce buffer copies. However, not every I/O call executes over the direct path; unsupported block alignments or filesystems silently trigger **compatibility mode**, falling back to standard CPU bounce buffering. [K6]

Architects must therefore distinguish between three claims:

- **Fact:** GDS eliminates intermediate CPU memory copies in supported workloads.
- **Conditional:** It can lower I/O latency and improve aggregate system throughput when CPU-to-PCIe memory bandwidth is the primary bottleneck.
- **Myth:** GDS accelerates every checkpoint by 10×, drops CPU utilization to zero, and allows 7.5 TB of state to hit disk in one second.

GDS is a **data-path optimization**. If your distributed storage cluster caps sustained ingestion throughput at 25 GB/s, establishing a direct DMA path from GPU HBM will not magically boost storage write speeds to 250 GB/s.

### 16.5. A 7.5 TB Snapshot Example

Consider a **hypothetical** cluster checkpoint with an aggregate state footprint of 7.5 TB. In decimal units:

$$
T_{\mathrm{write}}\approx\frac{S_{\mathrm{checkpoint}}}{W_{\mathrm{storage}}}
$$

| Aggregate Cluster Storage Write Bandwidth | Baseline Serialization Duration |
|---:|---:|
| 25 GB/s | 300 seconds (5.0 minutes) |
| 80 GB/s | 93.75 seconds |
| 250 GB/s | 30.0 seconds |
| 2 TB/s | 3.75 seconds (*ideal theoretical division*) |

Even with **2 TB/s aggregate filesystem write capacity**, an actual checkpoint will rarely complete in exactly 3.75 seconds. Filesystem metadata locks, atomic commit barriers, `fsync` flush operations, network hash partition imbalances, and tail stragglers introduce measurable real-world drag.

**The Tectonic Benchmark:** In the Llama 3 technical report, Meta describes their Tectonic distributed filesystem as provisioning 240 PB of raw storage across 7,500 SSD storage nodes, delivering **2 TB/s sustained** and **7 TB/s peak aggregate throughput**. These figures represent cluster storage tier capacity and bandwidth limits, not the instantaneous end-to-end duration of a single checkpoint. [K5]

### 16.6. Real Gains from Asynchronous Checkpointing

Imagine an entire software engineering team stopping all work and staring at blank screens for five minutes every time someone saves their code. What if they could snapshot their state in two seconds and immediately resume coding, letting a background process persist the files to disk? Even if physical disk writes still took five minutes, the productive stall time would shrink to two seconds.

This is the primary motivation behind asynchronous checkpointing. But it introduces an imperative operational question: Is an in-flight snapshot that has not yet completed persistence viable for disaster recovery?

A **synchronous checkpoint** halts all training execution until every byte has been flushed and acknowledged by persistent storage.

An **asynchronous checkpoint** decouples execution by capturing a consistent in-memory or host DRAM snapshot, allowing GPUs to immediately resume while storage I/O executes concurrently in the background. Systems engineers must track two independent metrics:

- **Snapshot / pause time:** The actual blocking stall duration imposed on training execution.
- **Persistence / durability time:** The wall-clock interval required for all shards to be verified, flushed, and committed to durable storage.

If a cluster crash occurs before the background flush finishes, that checkpoint is incomplete and invalid. In massive clusters, consistency barriers, atomic metadata manifests, shard integrity validation, and automated recovery drills are mission-critical.

---


## 17. Why Failures Are Inevitable in a 16,384-GPU Cluster

It is entirely normal for a personal laptop to operate for months without a hardware crash. But in a hyperscale datacenter housing tens of thousands of complex compute servers, the probability of at least one server failing on any given day approaches certainty.

This introduces a fundamental reliability paradox: individual servers may exhibit outstanding component reliability, yet the statistical probability that **the entire cluster remains 100% operational simultaneously** drops exponentially. In frontier LLM pretraining, reliability is therefore not about "eliminating all failures"; it is about "detecting anomalies instantaneously, isolating faulty nodes, and resuming execution with minimal lost compute."

### 17.1. MTBF: Single-Node Reliability Does Not Equal Cluster Reliability

**MTBF (Mean Time Between Failures)** denotes the expected average operating duration between successive system failures in a repairable system. Assume an individual server node has an MTBF of 26,000 hours (~3 years). If a cluster comprises 2,048 identical nodes and failures are assumed to be *independent and identically distributed*:

$$
M_{\mathrm{cluster}}\approx\frac{26000}{2048}=12.7\ \mathrm{hours}
$$

This is **not an empirically measured MTBF for a production cluster**; it is a didactic baseline derived from an idealized Poisson failure model. In real datacenter environments, correlated optical transceiver failures, power distribution unit (PDU) trips, software regressions, thermal throttling, and scheduled maintenance windows break independent failure assumptions.

### 17.2. Real-World Case Study: 54 Days of Llama 3 405B Training

We can confirm that this is not merely a theoretical exercise by examining frontier pretraining reports. The interruption telemetry documented in Meta's Llama 3 405B report illustrates why automated fault remediation is a foundational prerequisite for large-scale AI operations.

Across a 54-day observation window during Llama 3 405B pretraining, Meta recorded:

- **466 total job interruptions**.
- **47 planned interruptions** (maintenance windows and operational configuration changes).
- **419 unexpected interruptions**.
- Approximately **78% of unexpected interruptions** were attributed to confirmed or suspected hardware failures.
- Only **3 incidents required significant manual engineer intervention**; the remaining 416 events were resolved automatically by orchestration systems.

Under these operational conditions, the mean interval between unexpected interruptions was:

$$
\frac{54\times24}{419}\approx3.09\ \mathrm{hours}
$$

This must not be misconstrued as 'a physical H100 GPU died every 3 hours.' It represents the empirical rate of **unexpected distributed job interruptions**, whose root causes encompass silent data corruption, host memory errors, optical link degradation, network timeouts, and software runtime anomalies. [K5]

<svg viewBox="0 0 720 196" role="img" aria-label="Stacked bar of the 466 job interruptions during 54 days of Llama 3 405B pretraining: 47 planned and 419 unplanned, of which about 78 percent, roughly 327, were attributed to confirmed or suspected hardware issues and about 92 to other causes. That is one unplanned interruption every 3.09 hours on average. Only 3 needed significant manual intervention; automation handled the rest." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="li-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Llama 3 405B pretraining: 466 job interruptions in 54 days</text>
<rect x="17" y="44" width="67.3906" height="36" rx="2" style="fill:var(--c-text-mute);fill-opacity:.25"/>
<text x="50.69527896995708" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">47 planned</text>
<rect x="86.3906" y="44" width="480.781" height="36" rx="2" style="fill:var(--c-danger);fill-opacity:.45"/>
<text x="326.78111587982835" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">~327 hardware (≈78% of unplanned)</text>
<rect x="569.172" y="44" width="133.828" height="36" rx="2" style="fill:var(--c-warn);fill-opacity:.4"/>
<text x="636.0858369098713" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">~92 other</text>
<path d="M86.39055793991416 88 V94 H703 V88" style="fill:none;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="394.6952789699571" y="110" text-anchor="middle" style="fill:var(--c-danger);font-size:12px;font-weight:700">419 unplanned</text>
<rect x="16" y="124" width="688" height="60" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="146" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">54 days × 24 h ÷ 419 ≈ 3.09 h between unplanned interruptions</text>
<text x="28" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Only 3 needed significant manual intervention; automation handled the rest. Not “an H100 broke every 3 h”.</text>
</svg>

### 17.3. Young–Daly Formula: Why Checkpointing Must Be Neither Too Frequent Nor Too Infrequent

Suppose a cluster runs without saving for an hour, and a hardware crash occurs at minute 59. Nearly an hour of multi-thousand-GPU compute is permanently lost. Conversely, if the system checkpoints every 60 seconds, GPUs spend a massive fraction of their operational life stalled on disk I/O.

An optimal checkpoint cadence must strike a rigorous mathematical equilibrium between these two penalty modes.

When recovering from a node failure, a distributed job rolls back to the last verified checkpoint, incurring two distinct overhead penalties:

- Frequent checkpoints: Excessive I/O time lost to snapshot serialization.
- Infrequent checkpoints: Excessive compute lost to re-executing steps from the last checkpoint upon failure.

We can formulate this trade-off via an approximate loss ratio function:

$$
Loss(\tau)\approx\frac{\delta}{\tau}+\frac{\tau}{2M}
$$

where:

- $\tau$: Interval between consecutive checkpoints.
- $\delta$: Blocking duration of a checkpoint pause.
- $M$: Mean Time Between Failures (MTBF) of the distributed job.

The first term represents checkpoint I/O overhead; the second term represents expected recomputation waste following an unexpected crash. Setting the first derivative with respect to $\tau$ to zero:

$$
\frac{dLoss}{d\tau}=-\frac{\delta}{\tau^2}+\frac{1}{2M}=0
$$

$$
\boxed{\tau_{\mathrm{opt}}\approx\sqrt{2\delta M}}
$$

This is the classic **Young–Daly first-order approximation**. Non-Poisson failure distributions, state restore latency, asynchronous checkpoint overlapping, and cluster restart delays require higher-order extensions. [K10]

### 17.4. Comparison Under a 3-Hour MTBF Assumption

Let $M = 3\ \text{hours} = 10,800\ \text{seconds}$, with a checkpoint payload size of 7.5 TB:

| Metric | 25 GB/s Storage Fabric | 250 GB/s Storage Fabric |
|---|---:|---:|
| Blocking checkpoint pause $\delta$ | 300 seconds | 30 seconds |
| Theoretical optimal interval $\tau_{\mathrm{opt}}$ | 42.4 minutes | 13.4 minutes |
| Checkpoint overhead ratio $\delta/\tau$ | 11.8% | 3.7% |
| Expected recomputation waste $\tau/(2M)$ | 11.8% | 3.7% |
| **Total Approximate Overhead** | **23.6%** | **7.5%** |

<svg viewBox="0 0 720 326" role="img" aria-label="Young–Daly curves of total loss against checkpoint interval for a 7.5 TB checkpoint and a 3-hour mean time between interruptions. With 25 GB/s storage the blocking write takes 300 seconds; the curve bottoms out at an interval of 42.4 minutes with 23.6 percent loss. With 250 GB/s storage the write takes 30 seconds; the optimum is 13.4 minutes with 7.5 percent loss. Very short intervals are dominated by writing, very long ones by recomputation after failures." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="yd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Young–Daly: total loss W(τ) = δ/τ + τ/2M, with M = 3 h and a 7.5 TB checkpoint</text>
<line x1="70" y1="280" x2="690" y2="280" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="284.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0%</text>
<line x1="70" y1="234" x2="690" y2="234" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="238.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10%</text>
<line x1="70" y1="188" x2="690" y2="188" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="192.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">20%</text>
<line x1="70" y1="142" x2="690" y2="142" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="146.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">30%</text>
<line x1="70" y1="96" x2="690" y2="96" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="100.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">40%</text>
<line x1="70" y1="50" x2="690" y2="50" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="54.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">50%</text>
<text x="70.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="173.33333333333331" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">15</text>
<text x="276.66666666666663" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">30</text>
<text x="380.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">45</text>
<text x="483.3333333333333" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">60</text>
<text x="586.6666666666667" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">75</text>
<text x="690.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">90</text>
<line x1="70" y1="280" x2="690" y2="280" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="70" y1="50" x2="70" y2="280" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="380.0" y="314" text-anchor="middle" style="fill:var(--c-text);font-size:12px">checkpoint interval τ (minutes)</text>
<path d="M144.4 53.2 L145.9 57.3 L147.5 61.2 L149.1 64.9 L150.6 68.5 L152.2 71.9 L153.7 75.2 L155.2 78.3 L156.8 81.4 L158.3 84.3 L159.9 87.1 L161.4 89.8 L163.0 92.4 L164.6 94.9 L166.1 97.3 L167.7 99.6 L169.2 101.9 L170.8 104.0 L172.3 106.1 L173.8 108.2 L175.4 110.1 L176.9 112.0 L178.5 113.8 L180.1 115.6 L181.6 117.3 L183.1 119.0 L184.7 120.6 L186.2 122.1 L187.8 123.6 L189.4 125.1 L190.9 126.5 L192.4 127.9 L194.0 129.2 L195.6 130.5 L197.1 131.8 L198.7 133.0 L200.2 134.2 L201.8 135.3 L203.3 136.4 L204.8 137.5 L206.4 138.5 L207.9 139.6 L209.5 140.5 L211.1 141.5 L212.6 142.4 L214.2 143.3 L215.7 144.2 L217.2 145.1 L218.8 145.9 L220.3 146.7 L221.9 147.5 L223.4 148.3 L225.0 149.0 L226.6 149.8 L228.1 150.5 L229.7 151.1 L231.2 151.8 L232.8 152.5 L234.3 153.1 L235.9 153.7 L237.4 154.3 L238.9 154.9 L240.5 155.4 L242.1 156.0 L243.6 156.5 L245.2 157.1 L246.7 157.6 L248.2 158.0 L249.8 158.5 L251.3 159.0 L252.9 159.4 L254.4 159.9 L256.0 160.3 L257.5 160.7 L259.1 161.1 L260.6 161.5 L262.2 161.9 L263.8 162.3 L265.3 162.6 L266.9 163.0 L268.4 163.3 L270.0 163.7 L271.5 164.0 L273.1 164.3 L274.6 164.6 L276.1 164.9 L277.7 165.2 L279.2 165.5 L280.8 165.7 L282.4 166.0 L283.9 166.3 L285.4 166.5 L287.0 166.7 L288.6 167.0 L290.1 167.2 L291.6 167.4 L293.2 167.6 L294.8 167.8 L296.3 168.0 L297.9 168.2 L299.4 168.4 L300.9 168.6 L302.5 168.7 L304.1 168.9 L305.6 169.0 L307.1 169.2 L308.7 169.3 L310.2 169.5 L311.8 169.6 L313.4 169.8 L314.9 169.9 L316.4 170.0 L318.0 170.1 L319.6 170.2 L321.1 170.3 L322.6 170.4 L324.2 170.5 L325.8 170.6 L327.3 170.7 L328.9 170.8 L330.4 170.9 L331.9 170.9 L333.5 171.0 L335.1 171.1 L336.6 171.1 L338.1 171.2 L339.7 171.2 L341.2 171.3 L342.8 171.3 L344.4 171.4 L345.9 171.4 L347.4 171.4 L349.0 171.5 L350.6 171.5 L352.1 171.5 L353.6 171.5 L355.2 171.5 L356.8 171.6 L358.3 171.6 L359.9 171.6 L361.4 171.6 L362.9 171.6 L364.5 171.6 L366.1 171.6 L367.6 171.6 L369.1 171.5 L370.7 171.5 L372.2 171.5 L373.8 171.5 L375.4 171.5 L376.9 171.4 L378.4 171.4 L380.0 171.4 L381.6 171.4 L383.1 171.3 L384.6 171.3 L386.2 171.2 L387.8 171.2 L389.3 171.2 L390.9 171.1 L392.4 171.1 L393.9 171.0 L395.5 170.9 L397.0 170.9 L398.6 170.8 L400.1 170.8 L401.7 170.7 L403.2 170.6 L404.8 170.6 L406.3 170.5 L407.9 170.4 L409.4 170.4 L411.0 170.3 L412.6 170.2 L414.1 170.1 L415.6 170.0 L417.2 170.0 L418.8 169.9 L420.3 169.8 L421.9 169.7 L423.4 169.6 L424.9 169.5 L426.5 169.4 L428.1 169.3 L429.6 169.2 L431.2 169.1 L432.7 169.0 L434.2 168.9 L435.8 168.8 L437.4 168.7 L438.9 168.6 L440.5 168.5 L442.0 168.4 L443.6 168.3 L445.1 168.2 L446.6 168.1 L448.2 168.0 L449.8 167.8 L451.3 167.7 L452.9 167.6 L454.4 167.5 L455.9 167.4 L457.5 167.2 L459.1 167.1 L460.6 167.0 L462.1 166.9 L463.7 166.7 L465.2 166.6 L466.8 166.5 L468.4 166.3 L469.9 166.2 L471.4 166.1 L473.0 165.9 L474.5 165.8 L476.1 165.7 L477.6 165.5 L479.2 165.4 L480.8 165.2 L482.3 165.1 L483.8 165.0 L485.4 164.8 L486.9 164.7 L488.5 164.5 L490.1 164.4 L491.6 164.2 L493.1 164.1 L494.7 163.9 L496.2 163.8 L497.8 163.6 L499.4 163.5 L500.9 163.3 L502.4 163.1 L504.0 163.0 L505.6 162.8 L507.1 162.7 L508.7 162.5 L510.2 162.4 L511.8 162.2 L513.3 162.0 L514.9 161.9 L516.4 161.7 L518.0 161.5 L519.5 161.4 L521.0 161.2 L522.6 161.0 L524.1 160.9 L525.7 160.7 L527.2 160.5 L528.8 160.4 L530.4 160.2 L531.9 160.0 L533.5 159.8 L535.0 159.7 L536.5 159.5 L538.1 159.3 L539.6 159.2 L541.2 159.0 L542.8 158.8 L544.3 158.6 L545.9 158.4 L547.4 158.3 L549.0 158.1 L550.5 157.9 L552.0 157.7 L553.6 157.5 L555.1 157.4 L556.7 157.2 L558.2 157.0 L559.8 156.8 L561.3 156.6 L562.9 156.4 L564.5 156.2 L566.0 156.1 L567.5 155.9 L569.1 155.7 L570.6 155.5 L572.2 155.3 L573.8 155.1 L575.3 154.9 L576.9 154.7 L578.4 154.5 L580.0 154.3 L581.5 154.1 L583.0 154.0 L584.6 153.8 L586.1 153.6 L587.7 153.4 L589.2 153.2 L590.8 153.0 L592.4 152.8 L593.9 152.6 L595.5 152.4 L597.0 152.2 L598.5 152.0 L600.1 151.8 L601.6 151.6 L603.2 151.4 L604.8 151.2 L606.3 151.0 L607.9 150.8 L609.4 150.6 L611.0 150.4 L612.5 150.2 L614.0 150.0 L615.6 149.8 L617.1 149.6 L618.7 149.3 L620.2 149.1 L621.8 148.9 L623.4 148.7 L624.9 148.5 L626.5 148.3 L628.0 148.1 L629.5 147.9 L631.1 147.7 L632.6 147.5 L634.2 147.3 L635.8 147.1 L637.3 146.8 L638.9 146.6 L640.4 146.4 L642.0 146.2 L643.5 146.0 L645.0 145.8 L646.6 145.6 L648.1 145.4 L649.7 145.1 L651.2 144.9 L652.8 144.7 L654.4 144.5 L655.9 144.3 L657.5 144.1 L659.0 143.8 L660.5 143.6 L662.1 143.4 L663.6 143.2 L665.2 143.0 L666.8 142.8 L668.3 142.5 L669.9 142.3 L671.4 142.1 L673.0 141.9 L674.5 141.7 L676.0 141.4 L677.6 141.2 L679.1 141.0 L680.7 140.8 L682.2 140.6 L683.8 140.3 L685.4 140.1 L686.9 139.9 L688.5 139.7" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<circle cx="362.271" cy="171.577" r="5" style="fill:var(--c-danger);fill-opacity:.9"/>
<line x1="362.271" y1="171.577" x2="362.271" y2="280" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<text x="370.27080289043965" y="193.5769602180627" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">τ* = 42.4 min → 23.6%</text>
<line x1="428.222" y1="110.4" x2="450.222" y2="110.4" style="stroke:var(--c-danger);stroke-width:2.2"/>
<text x="456.2222222222222" y="114.4" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">25 GB/s → δ = 300 s</text>
<path d="M77.8 74.1 L79.3 107.9 L80.8 132.0 L82.4 149.9 L84.0 163.8 L85.5 174.9 L87.0 183.9 L88.6 191.4 L90.2 197.6 L91.7 203.0 L93.2 207.5 L94.8 211.5 L96.3 215.0 L97.9 218.0 L99.5 220.7 L101.0 223.1 L102.5 225.3 L104.1 227.2 L105.7 228.9 L107.2 230.5 L108.8 231.9 L110.3 233.2 L111.8 234.4 L113.4 235.4 L115.0 236.4 L116.5 237.3 L118.0 238.1 L119.6 238.9 L121.2 239.5 L122.7 240.2 L124.2 240.7 L125.8 241.3 L127.3 241.7 L128.9 242.2 L130.4 242.6 L132.0 242.9 L133.6 243.3 L135.1 243.6 L136.7 243.9 L138.2 244.1 L139.8 244.3 L141.3 244.6 L142.8 244.7 L144.4 244.9 L145.9 245.1 L147.5 245.2 L149.1 245.3 L150.6 245.4 L152.2 245.5 L153.7 245.5 L155.2 245.6 L156.8 245.6 L158.3 245.7 L159.9 245.7 L161.4 245.7 L163.0 245.7 L164.6 245.7 L166.1 245.7 L167.7 245.7 L169.2 245.6 L170.8 245.6 L172.3 245.5 L173.8 245.5 L175.4 245.4 L176.9 245.3 L178.5 245.3 L180.1 245.2 L181.6 245.1 L183.1 245.0 L184.7 244.9 L186.2 244.8 L187.8 244.7 L189.4 244.6 L190.9 244.5 L192.4 244.3 L194.0 244.2 L195.6 244.1 L197.1 244.0 L198.7 243.8 L200.2 243.7 L201.8 243.5 L203.3 243.4 L204.8 243.2 L206.4 243.1 L207.9 242.9 L209.5 242.8 L211.1 242.6 L212.6 242.4 L214.2 242.3 L215.7 242.1 L217.2 241.9 L218.8 241.8 L220.3 241.6 L221.9 241.4 L223.4 241.2 L225.0 241.0 L226.6 240.8 L228.1 240.7 L229.7 240.5 L231.2 240.3 L232.8 240.1 L234.3 239.9 L235.9 239.7 L237.4 239.5 L238.9 239.3 L240.5 239.1 L242.1 238.9 L243.6 238.7 L245.2 238.5 L246.7 238.3 L248.2 238.0 L249.8 237.8 L251.3 237.6 L252.9 237.4 L254.4 237.2 L256.0 237.0 L257.5 236.8 L259.1 236.5 L260.6 236.3 L262.2 236.1 L263.8 235.9 L265.3 235.7 L266.9 235.4 L268.4 235.2 L270.0 235.0 L271.5 234.8 L273.1 234.5 L274.6 234.3 L276.1 234.1 L277.7 233.8 L279.2 233.6 L280.8 233.4 L282.4 233.2 L283.9 232.9 L285.4 232.7 L287.0 232.4 L288.6 232.2 L290.1 232.0 L291.6 231.7 L293.2 231.5 L294.8 231.3 L296.3 231.0 L297.9 230.8 L299.4 230.5 L300.9 230.3 L302.5 230.1 L304.1 229.8 L305.6 229.6 L307.1 229.3 L308.7 229.1 L310.2 228.8 L311.8 228.6 L313.4 228.4 L314.9 228.1 L316.4 227.9 L318.0 227.6 L319.6 227.4 L321.1 227.1 L322.6 226.9 L324.2 226.6 L325.8 226.4 L327.3 226.1 L328.9 225.9 L330.4 225.6 L331.9 225.4 L333.5 225.1 L335.1 224.9 L336.6 224.6 L338.1 224.4 L339.7 224.1 L341.2 223.8 L342.8 223.6 L344.4 223.3 L345.9 223.1 L347.4 222.8 L349.0 222.6 L350.6 222.3 L352.1 222.1 L353.6 221.8 L355.2 221.5 L356.8 221.3 L358.3 221.0 L359.9 220.8 L361.4 220.5 L362.9 220.3 L364.5 220.0 L366.1 219.7 L367.6 219.5 L369.1 219.2 L370.7 219.0 L372.2 218.7 L373.8 218.4 L375.4 218.2 L376.9 217.9 L378.4 217.7 L380.0 217.4 L381.6 217.1 L383.1 216.9 L384.6 216.6 L386.2 216.3 L387.8 216.1 L389.3 215.8 L390.9 215.5 L392.4 215.3 L393.9 215.0 L395.5 214.8 L397.0 214.5 L398.6 214.2 L400.1 214.0 L401.7 213.7 L403.2 213.4 L404.8 213.2 L406.3 212.9 L407.9 212.6 L409.4 212.4 L411.0 212.1 L412.6 211.8 L414.1 211.6 L415.6 211.3 L417.2 211.0 L418.8 210.8 L420.3 210.5 L421.9 210.2 L423.4 210.0 L424.9 209.7 L426.5 209.4 L428.1 209.2 L429.6 208.9 L431.2 208.6 L432.7 208.4 L434.2 208.1 L435.8 207.8 L437.4 207.5 L438.9 207.3 L440.5 207.0 L442.0 206.7 L443.6 206.5 L445.1 206.2 L446.6 205.9 L448.2 205.7 L449.8 205.4 L451.3 205.1 L452.9 204.8 L454.4 204.6 L455.9 204.3 L457.5 204.0 L459.1 203.8 L460.6 203.5 L462.1 203.2 L463.7 203.0 L465.2 202.7 L466.8 202.4 L468.4 202.1 L469.9 201.9 L471.4 201.6 L473.0 201.3 L474.5 201.0 L476.1 200.8 L477.6 200.5 L479.2 200.2 L480.8 200.0 L482.3 199.7 L483.8 199.4 L485.4 199.1 L486.9 198.9 L488.5 198.6 L490.1 198.3 L491.6 198.0 L493.1 197.8 L494.7 197.5 L496.2 197.2 L497.8 196.9 L499.4 196.7 L500.9 196.4 L502.4 196.1 L504.0 195.8 L505.6 195.6 L507.1 195.3 L508.7 195.0 L510.2 194.8 L511.8 194.5 L513.3 194.2 L514.9 193.9 L516.4 193.7 L518.0 193.4 L519.5 193.1 L521.0 192.8 L522.6 192.5 L524.1 192.3 L525.7 192.0 L527.2 191.7 L528.8 191.4 L530.4 191.2 L531.9 190.9 L533.5 190.6 L535.0 190.3 L536.5 190.1 L538.1 189.8 L539.6 189.5 L541.2 189.2 L542.8 189.0 L544.3 188.7 L545.9 188.4 L547.4 188.1 L549.0 187.9 L550.5 187.6 L552.0 187.3 L553.6 187.0 L555.1 186.7 L556.7 186.5 L558.2 186.2 L559.8 185.9 L561.3 185.6 L562.9 185.4 L564.5 185.1 L566.0 184.8 L567.5 184.5 L569.1 184.3 L570.6 184.0 L572.2 183.7 L573.8 183.4 L575.3 183.1 L576.9 182.9 L578.4 182.6 L580.0 182.3 L581.5 182.0 L583.0 181.7 L584.6 181.5 L586.1 181.2 L587.7 180.9 L589.2 180.6 L590.8 180.4 L592.4 180.1 L593.9 179.8 L595.5 179.5 L597.0 179.2 L598.5 179.0 L600.1 178.7 L601.6 178.4 L603.2 178.1 L604.8 177.8 L606.3 177.6 L607.9 177.3 L609.4 177.0 L611.0 176.7 L612.5 176.5 L614.0 176.2 L615.6 175.9 L617.1 175.6 L618.7 175.3 L620.2 175.1 L621.8 174.8 L623.4 174.5 L624.9 174.2 L626.5 173.9 L628.0 173.7 L629.5 173.4 L631.1 173.1 L632.6 172.8 L634.2 172.5 L635.8 172.3 L637.3 172.0 L638.9 171.7 L640.4 171.4 L642.0 171.1 L643.5 170.9 L645.0 170.6 L646.6 170.3 L648.1 170.0 L649.7 169.7 L651.2 169.5 L652.8 169.2 L654.4 168.9 L655.9 168.6 L657.5 168.3 L659.0 168.1 L660.5 167.8 L662.1 167.5 L663.6 167.2 L665.2 166.9 L666.8 166.7 L668.3 166.4 L669.9 166.1 L671.4 165.8 L673.0 165.5 L674.5 165.3 L676.0 165.0 L677.6 164.7 L679.1 164.4 L680.7 164.1 L682.2 163.8 L683.8 163.6 L685.4 163.3 L686.9 163.0 L688.5 162.7" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<circle cx="162.424" cy="245.714" r="5" style="fill:var(--c-success);fill-opacity:.9"/>
<line x1="162.424" y1="245.714" x2="162.424" y2="280" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<text x="170.42414306999132" y="267.7136243450032" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">τ* = 13.4 min → 7.5%</text>
<line x1="428.222" y1="124.2" x2="450.222" y2="124.2" style="stroke:var(--c-success);stroke-width:2.2"/>
<text x="456.2222222222222" y="128.2" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">250 GB/s → δ = 30 s</text>
<text x="194.0" y="63.80000000000001" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">short τ: writing dominates</text>
<text x="428.2222222222222" y="91.4" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">long τ: recomputation after failures dominates</text>
</svg>

Two critical engineering takeaways emerge:

1. Slashing storage commit latency by a factor of 10 compresses the optimal checkpoint interval by a factor of $\sqrt{10}\approx3.16$.
2. At the theoretical optimum, the overhead of writing checkpoints exactly balances the expected cost of recomputing lost work.

**Caveat:** When modeling asynchronous checkpoint systems, do not substitute the entire end-to-end background storage write duration for $\delta$; substitute only the synchronous GPU blocking stall time, while verifying persistence guarantees.

---

### 17.5. Is Checkpointing Enough on Its Own? Health Checks and Quarantining Faulty Nodes

We have examined checkpoint cadence and recovery math. Yet in hyperscale clusters, the critical architectural challenge is frequently not "How quickly can we write a checkpoint?", but rather **"When a node degrades, how quickly can orchestration detect, diagnose, and cordon that node from the active training ring?"**

Imagine a high-speed train with a defective axle. Decoupling and restarting the engine while leaving the defective car in the formation does not fix the train; it guarantees a derailment seconds later. In an AI cluster, blind restart loops that relaunch from checkpoints while retaining a degraded GPU or flapping NIC merely result in immediate crash loops.

**RAS (Reliability, Availability, and Serviceability)** encompasses architectural engineering principles designed to detect faults, maintain uninterrupted service, and streamline maintenance. Diagnostic telemetry in libraries like NCCL helps identify unresponsive ranks and communication timeouts; however, NCCL itself does not provide automated self-healing. [K32]

A production GPU cluster demands automated health checks across at least three operational tiers:

1. **Node Tier:** GPU uncorrectable ECC errors, NVIDIA driver Xid events, PCIe bus errors, thermal throttling, power capping, and BMC telemetry.
2. **Fabric Tier:** InfiniBand/RoCE link flaps, symbol errors, packet retransmissions, PFC pause storms, RDMA queue-pair drops, and collective timing anomalies.
3. **Workload Tier:** Step duration outliers, loss/grad-norm NaN checks, straggler rank divergence, checkpoint commit validation, and crash restart counters.

NVIDIA's SuperPOD architecture operational guide recommends immediately cordoning degraded nodes from the scheduling pool and running rigorous single- and multi-node diagnostic suites (including NCCL tests) before returning nodes to production. [K33]



<svg viewBox="0 0 720 342" role="img" aria-label="Node quarantine loop. A node in the healthy pool is flagged by telemetry such as Xid, ECC or link errors, or by straggler step times. It is drained and isolated from the batch pool, then validated in escalating stages: DCGM diagnostics, a single-node NCCL test and a multi-node NCCL test. If it passes, it returns to the pool in stages; if it fails, it goes to repair or RMA and is re-tested afterwards. Restarting from a checkpoint with the faulty node still in the job reproduces the failure." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="qr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="qr-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="qr-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Node quarantine loop: detect → isolate → prove healthy → return</text>
<rect x="16" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="91.0" y="72.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Healthy pool</text>
<text x="91.0" y="86.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">jobs scheduled here</text>
<rect x="206" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="281.0" y="64.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Detect</text>
<text x="281.0" y="79.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Xid · ECC · link errors</text>
<text x="281.0" y="93.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">straggler step time</text>
<rect x="396" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="471.0" y="72.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Drain + isolate</text>
<text x="471.0" y="86.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">out of the batch pool</text>
<line x1="166" y1="74" x2="204" y2="74" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="356" y1="74" x2="394" y2="74" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="380" y="128" width="324" height="132" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="392" y="148" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Validate (escalating)</text>
<rect x="392" y="158" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="175.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">1 · DCGM diagnostics</text>
<rect x="392" y="190" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="207.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">2 · single-node NCCL test</text>
<rect x="392" y="222" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="239.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">3 · multi-node NCCL test</text>
<line x1="471" y1="104" x2="471" y2="126" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M380 200 H91 V106" marker-end="url(#qr-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<text x="236" y="192" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">pass → staged return</text>
<path d="M542 260 V286 H360" marker-end="url(#qr-b)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="206" y="270" width="152" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="282.0" y="295.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Repair / RMA</text>
<text x="452" y="300" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">fail</text>
<path d="M282 270 V244 H378" marker-end="url(#qr-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="300" y="238" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">re-test</text>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Restarting from a checkpoint with the faulty node still in the job just reproduces the failure.</text>
</svg>



**Silent Data Corruption (SDC)** occurs when microarchitectural hardware faults alter computational results without raising operating system traps or ECC hardware exceptions. Assuming training is executing correctly simply because "no rank crashed" is hazardous. Deterministic spot checks, loss-spike watchdogs, weight checksums, and arithmetic canary kernels must be integrated into cluster validation pipelines.

Distinguish clearly between two operational metrics: **RPO (Recovery Point Objective)** defines the maximum acceptable window of lost training progress; **RTO (Recovery Time Objective)** defines the maximum duration allowed to restore the cluster to active computation following a crash. RPO is bounded by checkpoint frequency and durability; RTO is bounded by failure isolation, node replacement latency, and checkpoint deserialization speed.

Formulate explicit **recovery objectives**: When an outage strikes, what volume of lost computation is acceptable? In training, this dictates checkpoint intervals and storage throughput targets. In production inference, by contrast, the objective is to protect user experience by routing traffic to healthy replicas and maintaining low-latency request completion. Applying identical recovery strategies to training and inference is an architectural anti-pattern.

---



---

### A Checkpoint Is Not Just a File; It Is a Ticket on a Time Machine

When a cluster loses a worker after weeks of pretraining, the engineering challenge is never simply "Where was the checkpoint file written?" The real question is: **From what exact mathematical state, and with what consistency guarantees, can we resume optimization?** Beyond raw model weights, valid resumption requires optimizer momentum tensors, learning rate scheduler state, dataset shard offsets, and distributed RNG states. Serializing model weights alone is never sufficient to resume training along the identical statistical trajectory.

Two distinct temporal metrics govern this process:

- **Checkpoint I/O time:** The aggregate time required to transfer and persist serialized state onto durable storage.
- **Training pause time:** The actual synchronous interval during which GPU compute engines are halted.

When training architectures decouple in-memory snapshots from background storage flushes, these two durations diverge. Massive aggregate storage bandwidth is essential to clear snapshots before the next interval triggers, but bandwidth metrics alone do not indicate *how many seconds GPUs stalled*.

**Intuitive RPO/RTO framing for pretraining:** RPO measures the volume of lost gradient steps when rolling back to the prior checkpoint; RTO measures the duration required to isolate failed nodes, reschedule the job, reload state across thousands of GPUs, and resume execution. In production inference environments, RPO/RTO encompasses dropped in-flight requests, service availability SLOs, and user experience degradation.

### Why Does the Young–Daly Approach Involve a Square Root?

The square root in the Young–Daly formulation ($\tau_{\mathrm{opt}}\approx\sqrt{2\delta M}$) emerges directly from balancing two opposing geometric curves. As checkpoint interval $\tau$ shrinks, storage serialization overhead expands hyperbolically ($1/\tau$). Conversely, as the interval stretches, expected recomputation loss following an unexpected node failure grows linearly ($\tau/2M$).

The composite overhead curve achieves its minimum precisely where the marginal rates of these two penalty modes cancel each other out. The primary engineering insight is clear: compressing storage pause duration ($\delta$) by 10× permits tightening the optimal checkpoint cadence by a factor of $\sqrt{10}\approx 3.16$. In asynchronous snapshot architectures, $\delta$ must strictly represent the synchronous GPU pause time rather than background disk drain duration.


## 18. Post-Failure Continuity: Blast Radius, RPO, RTO, and Rollbacks

In a datacenter orchestrating thousands of GPUs, assuming that "failures will never occur" is both economically ruinous and architecturally fragile. The mandate of resilient infrastructure is to **contain the blast radius, detect failures instantly, recover automatically, and feed operational telemetry back into preventive maintenance**.

### 18.1. Blast Radius: How Broad Is the Impact of a Single Failure?

The **blast radius** measures the scope of degradation triggered by a single component failure or erroneous configuration rollout. A crashed inference worker simply drops one replica; a corrupted global routing policy drops all incoming production traffic. A single top-of-rack network switch failure disables one rack; a flawed cluster-wide firmware push can take down the entire datacenter.

Resilient engineering demands explicit **failure domain isolation**: structuring systems around hosts, racks, power delivery feeds, leaf switches, network rails, availability zones, and storage failure domains. Provisioning redundant replicas on the same physical server or shared rack delivers the illusion of high availability on paper, while sharing the identical single point of failure in reality.

### 18.2. Training and Serving Outages Require Distinct Recovery Strategies

| Failure Mode | Large-Scale Distributed Training | Production Online Inference |
|---|---|---|
| Single GPU Failure | Halts distributed job; triggers automated node cordon and restore from checkpoint | Request fails over to alternate replica; active streaming token response may abort |
| Network Rail Degradation | Triggers collective communication hangs and watchdog timeouts | Degrades cross-node KV cache transfers and inter-node routing |
| Storage Outage | Stalls DataLoader workers and blocks checkpoint writes | Blocks dynamic model loading, telemetry logging, and prompt cache tiers |
| Defective Model Rollout | Diverges loss trajectory and corrupts weight checkpoints | Degrades generation quality and spikes user-facing latency |
| Authorization Policy Bug | Risks training dataset leakage | Exposes unauthorized user queries, privileged tool executions, and data retrieval |

### 18.3. RPO, RTO, and Point-in-Time Recovery

**RPO (Recovery Point Objective)** dictates the maximum tolerable loss of data or compute progress; **RTO (Recovery Time Objective)** defines the maximum acceptable downtime required to restore normal service. In large-scale training, an RPO might be established as "at most 20 minutes of discarded compute"; in online serving, the objective is to re-route requests within sub-seconds to alternate replicas.

The mere existence of a checkpoint file is **no guarantee of recoverability**. Teams must continually validate shard completeness, schema compatibility, weight corruption checksums, access permissions, and cross-failure-domain restore pipelines through automated verification drills.

### 18.4. Silent Data Corruption (SDC) and Model Quality

Certain microarchitectural and bus defects produce corrupt numerical values without triggering operating system crashes or hardware interrupts. This is known as **Silent Data Corruption (SDC)**. While hardware protections such as ECC memory detect and correct single-bit memory flips, they provide no blanket defense against ALU arithmetic errors, silent register corruption, or bus transceivers dropping bits. In training runs, anomalous loss spikes, rank-isolated NaN/Inf occurrences, checkpoint divergence, or abrupt validation drops should trigger operational alarms. However, not every loss spike indicates hardware corruption; optimizer learning rate schedules, corrupted training samples, and precision underflows must be ruled out systematically.

### 18.5. Safe Rollout Pipelines: Canary → Shadow → Production

Rather than deploying an updated inference engine or model checkpoint directly across your entire GPU fleet:

1. **Staging and Replay:** Replay captured production requests using sanitized, anonymized, or synthetic traces in accordance with data governance policies.
2. **Shadow Traffic:** Where compliant, mirror live production queries to the candidate system asynchronously without impacting user responses. Verify data processing authorizations for secondary shadow tiers.
3. **Canary Deployment:** Route a small percentage of live user traffic to candidate instances, monitoring latency SLOs, error rates, and response quality.
4. **Phased Rollout:** Progressively expand routing weights across fleet tiers, gating progression at each step against automated regression criteria.
5. **Automated Rollback:** Rigorously validate the ability to instantly roll back model checkpoints, tokenizers, runtime engines, routers, and policy rules to their known-good state.

**Warning:** Rolling back a model weight checkpoint does not automatically roll back external schema mutations, vector index versions, or upstream API dependencies. Release packages and dependency pins must be versioned and deployed as atomic units.

### 18.6. Telemetry That Demystifies an Incident

If your sole operational alert reads "GPU utilization dropped," mean-time-to-resolution (MTTR) will be agonizingly slow. An actionable incident card must consolidate: request ID or distributed job ID, model checkpoint version, physical GPU/node/failure-domain topology, recent deployment timestamps, exact anomaly onset time, DCGM/Xid events, NCCL collective state, network fabric link counters, queue depth metrics, storage I/O throughput, and initial safe containment instructions.

---




---

**PART VI — Measurement: MFU, Latency, Quality, Cost**

*Learning not to confuse a percentage on a dashboard with real hardware efficiency, and understanding the interplay between metrics.*

## 19. 38% MFU: Where Is the GPU Truly Spending Its Time?

We have arrived at the overarching question posed at the very beginning: “We procured 1,024 GPUs, yet our MFU is only 38%. Where did the rest of the performance go?”

By now, we have several plausible answers: the GEMM shapes might not be large enough, weights may be moving too slowly out of HBM, collective communications could be taking too long, pipeline stages might be sitting idle in bubbles, or the system could be stalling during checkpoints.

Yet, without rigorous measurement, none of these possibilities can be definitively declared the root cause. MFU is an **outcome metric**; it is not a diagnostic report in itself.

### 19.1. GPU Utilization, MFU, and HFU Are Not the Same Metric

First, let us dispel a common pitfall. Seeing 95% GPU utilization on a telemetry dashboard does not mean that "we converted 95% of theoretical Tensor Core capacity into useful model computation."

Think of these metrics analogously to a factory asking: "How long were the factory lights on?", "How many units did the machines assemble?", and "How many finished goods actually delivered value to customers?" They may sound similar, but they measure fundamentally different realities.

Conflating these three concepts is treacherous:

**GPU Utilization** is hardware telemetry reflecting whether the GPU is executing any active kernel within a given sampling interval. A high utilization figure does not indicate that matrix compute engines are operating anywhere near peak efficiency.

**Model FLOPs Utilization (MFU)** is the ratio of the model's useful FLOP/s rate to the theoretical peak hardware FLOP/s ceiling. The model FLOP count is typically an estimate based on model architecture and token throughput; the exact operations accounted for must always be explicitly stated.

**Hardware FLOPs Utilization (HFU)** compares the actual executed hardware FLOP/s against the peak ceiling. For instance, recalculations such as activation recomputation inflate HFU, but because they do not contribute to useful model FLOPs, they do not increase MFU.

| Metric | Question It Asks | Typical Pitfall |
|---|---|---|
| GPU utilization | Is the GPU active? | 'Active' ≠ 'efficient model compute' |
| SM active / achieved occupancy | How heavily are execution resources allocated? | High occupancy alone does not imply high TFLOP/s |
| Tensor Core throughput | How much work are MMA pipes executing? | Measures only GEMMs, not the full step |
| MFU | What fraction of theoretical peak was converted to useful model FLOPs? | Model FLOP definition and step time matter critically |
| HFU | What fraction of peak did the hardware's raw FLOPs reach? | Counts unhelpful work like recomputation |
| Effective training time | What portion of total elapsed wall-clock time was spent training? | Checkpointing and failure overheads must be accounted for separately |

### 19.2. Sanity Check for 1,024 H100s at 38% MFU

If we use the theoretical dense BF16 ceiling of 990 TFLOP/s/GPU:

$$
C_{\mathrm{peak,total}}=1024\times990\ \mathrm{TFLOP/s}
\approx1.014\ \mathrm{EFLOP/s}
$$

If MFU is 38%:

$$
C_{\mathrm{useful}}\approx1.014\times0.38
\approx0.385\ \mathrm{EFLOP/s}
$$

This does not imply that **62% of the GPUs are physically powered off**. That 62% represents theoretical peak capacity that failed to translate into useful model work. Memory stalls, collective communication, pipeline bubbles, sub-optimal kernel efficiency, and software framework overhead all contribute to this gap in varying proportions.

<svg viewBox="0 0 720 226" role="img" aria-label="Bar of the theoretical BF16 dense peak of 1,024 H100s, about 1.014 exaFLOP/s. 38 percent, about 0.385 exaFLOP/s, becomes useful model FLOPs. The other 62 percent is potential not turned into useful model math; it does not mean 62 percent of the GPUs are switched off. Candidate causes, whose shares are unknown until profiled: memory-bound kernels, collective waits, pipeline bubbles, small or unfused kernels, recomputation that counts in HFU but not MFU, and data or checkpoint stalls." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">1,024 H100 at 38% MFU: what the 62% is — and is not</text>
<text x="16" y="50" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">theoretical peak: 1,024 × 990 TFLOP/s ≈ 1.014 EFLOP/s (BF16 dense)</text>
<rect x="16" y="60" width="261.44" height="40" rx="4" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="146.72" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">useful model FLOPs</text>
<text x="146.72" y="94" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">38% ≈ 0.385 EFLOP/s</text>
<rect x="277.44" y="60" width="426.56" height="40" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="490.71999999999997" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">62% not turned into useful model math</text>
<text x="490.71999999999997" y="94" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">GPUs are NOT 62% switched off</text>
<text x="277.44" y="124" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">candidate causes — shares are unknown until profiled:</text>
<rect x="277.44" y="134" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.44" y="148.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">memory-bound kernels</text>
<rect x="491.44" y="134" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594.44" y="148.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">collective waits</text>
<rect x="277.44" y="162" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.44" y="176.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">pipeline bubble</text>
<rect x="491.44" y="162" width="206" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:.1;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="594.44" y="176.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">small / unfused kernels</text>
<rect x="277.44" y="190" width="206" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:.1;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="380.44" y="204.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">recomputation (HFU, not MFU)</text>
<rect x="491.44" y="190" width="206" height="22" rx="4" style="fill:var(--c-danger);fill-opacity:.1;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="594.44" y="204.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">data / checkpoint stalls</text>
</svg>

### 19.3. If MFU Is 38%, What Data Should We Collect First?

When confronted with a performance issue, our immediate knee-jerk reaction should not be to throw more GPUs at the cluster or blindly increase batch size. First, we must pinpoint where time is actually spent. Do not treat diagnosis like taking a patient's temperature with only a thermometer; comprehensive diagnostic telemetry is required.

Let us dissect a single training step onto a timeline: in which intervals are matrix kernels executing, where is collective communication occurring, when is the dataloader running, and when is storage I/O taking place? Once serialized stalls and wait times become visible, the solution path emerges clearly.

Before attempting to resolve any performance bottleneck, we must inspect the unified **GPU and host execution timeline**.

Recommended observability layers:

1. **Model KPIs:** tokens/s, steps/s, MFU, sequence length, global batch size, microbatch size, gradient accumulation steps.
2. **Kernel KPIs:** GEMM shapes, Tensor Core instructions, achieved FLOP/s, DRAM throughput, cache hit rates, register spills.
3. **Communication KPIs:** NCCL AllReduce/AllGather/ReduceScatter/AllToAll durations, byte volumes, overlap percentage.
4. **Pipeline KPIs:** Compute, transfer, and bubble time per stage; *straggler* stage distribution.
5. **I/O KPIs:** Data wait time, CPU parse/decode duration, checkpoint pause time, durable write latency, filesystem p95 latency.
6. **Reliability KPIs:** Interruption count and root causes, restart/recovery time, age of the last valid checkpoint.

The standard diagnostic tool suite includes **NVIDIA Nsight Systems**, **Nsight Compute**, **DCGM**, **NCCL tests**, PyTorch Profiler, system I/O profilers, and network fabric telemetry. Each provides insights at a different scale:

- Nsight Systems: Unified timeline of GPU/CPU kernels, CUDA streams, runtime API calls, and communication events.
- Nsight Compute: Granular breakdown of a CUDA kernel's compute, cache utilization, memory hierarchy traffic, and warp occupancy.
- DCGM: Continuous GPU health, thermal status, power, and operational performance telemetry.
- NCCL tests: Empirical throughput and latency baselines for collective communication primitives.

### 19.4. Symptom → Potential Root Cause → Controlled Experiment

| Observed Symptom | Likely Bottleneck | Controlled Experiment to Run |
|---|---|---|
| High HBM throughput, low TFLOP/s | Memory-bound kernels | Sweep batch size and weight precision; evaluate Roofline model |
| Low Tensor Core utilization, small GEMMs | Sub-optimal / fragmented matrix dimensions | Benchmark tiling, kernel fusion, and degree of parallelism |
| Excessive NCCL AllReduce latency | Network fabric, collective algorithm, or packet pacing | Run intra-node vs. cross-node NCCL microbenchmarks |
| Step slows down as TP increases | TP scaling threshold exceeded (communication dominates) | Compare TP=4/8/16 mapped against physical topology |
| Certain PP stages remain idle | Pipeline bubbles / inter-stage imbalance | Vary microbatch count and test alternative schedules (e.g., 1F1B) |
| GPUs pause periodically | Synchronous checkpointing or dataloader starvation | Measure memory snapshot pause vs. durable persist time separately |
| A single GPU stalls the entire step | Straggler: thermal throttling, ECC recovery, or host-side jitter | Log per-rank step times, SM clocks, and hardware error counters |
| Slow decode, robust prefill | Memory bandwidth bound by weight reads or KV cache access | Compare concurrency levels, continuous batching, and KV quantization |
| OOM errors, constrained user concurrency | Weights + KV footprint + memory fragmentation | Analyze KV cache sizing, GQA efficiency, and PagedAttention allocation |

**Golden rule:** Never modify ten hyperparameters simultaneously. Isolate a single variable per experiment; jointly report the optimization's impact across throughput, latency, convergence quality, infrastructure cost, and system resilience.

### 19.5. Divergent Decision Matrices for Training vs. Inference

| Architectural Choice | Impact on Training | Impact on Inference |
|---|---|---|
| Faster Tensor Cores | Accelerates large GEMMs in forward and backward passes | Primarily benefits the prefill phase and high-concurrency batching |
| Higher HBM Bandwidth | Accelerates memory-bandwidth-bound and elementwise kernels | Critical bottleneck for low-batch autoregressive decode |
| Larger HBM Capacity | Accommodates activations, optimizer states, and large models | Accommodates larger KV caches and higher concurrent request counts |
| NVLink / NVSwitch Fabric | Accelerates high-frequency TP and local collectives | Enables low-latency multi-GPU model serving across ranks |
| Compute Fabric / RDMA | Scales DP, PP, EP, and CP collectives cross-node | Enables disaggregated serving topologies and distributed inference |
| High-Burst Storage | Minimizes checkpointing pause and fast-tracks recovery | Accelerates model weight loading and container cold starts |
| GPUDirect Storage (GDS) | Eliminates bounce-buffer copies in compliant I/O pipelines | Conditionally speeds up weight streaming and multimodal data loading |
| FP8 / INT8 / INT4 Quantization | Constrained by training stability, numerical recipe, and precision | Unlocks drastic weight memory savings, higher throughput, and lower latency |

### 19.6. An Operational Morning: How Would We Diagnose 38% MFU?

Suppose the operations team sends an alert first thing in the morning: “Last night we rolled out a new checkpointing strategy; MFU dropped from 42% down to 38%.” Hearing this, rather than arbitrarily altering the GPU allocation, we must ask a disciplined sequence of diagnostic questions.

**Question 1: Did the underlying workload change?** Are the global batch size, sequence length, precision, and model FLOP formula identical? If context length was extended, the MFU shift might reflect algorithmic scaling properties rather than a software regression. If the benchmark baselines are not strictly equivalent, attributing the delta to a single architectural element will lead to false conclusions.

**Question 2: Is the MFU degradation sustained or periodic?** If timeline traces reveal wide synchronous gaps appearing every few minutes, periodic tasks such as checkpointing or dataloader prefetching are prime suspects. If NCCL collectives stall at the tail end of every step, fabric congestion or sub-optimal collective topology is indicated. If GPU kernels execute continuously without idle gaps yet yield low FLOP/s, Roofline positioning and GEMM shape efficiency are the appropriate starting points.

**Question 3: Is a single straggler holding back the entire collective?** Compare execution profiles across ranks. In a cluster of thousands of GPUs, if even a few nodes experience thermal throttling, retry on corrected ECC errors, or lag on host I/O, every synchronous barrier will be dragged down to the speed of the slowest worker. Averaged cluster-wide metrics will mask this entirely.

**Question 4: What is the minimal controlled experiment to validate a hypothesis?** For example, execute intra-node NCCL benchmarks across 8 GPUs and contrast them with inter-node fabric sweeps; disable checkpointing for a handful of steps to isolate its overhead on step latency; or test batch size adjustments while keeping tensor parallelism fixed.

**Final Question: Does this yield tangible value for the end user?** For training, evaluate useful tokens/s and overall time-to-convergence; for inference, inspect TTFT, TPOT, overall throughput, and p95/p99 tail latency. A higher MFU is a positive indicator, but it is never the sole objective metric of success.

In this manner, we establish a systematic diagnostic methodology: **measure → locate the bottleneck → test an isolated hypothesis → validate genuine business value.** Now, let us ground this methodology with empirical figures from modern Blackwell benchmarks.


### 19.7. Don't Just Read MFU, Calculate It from First Principles

Let us step back from the telemetry dashboard and calculate MFU ourselves. Understanding *the exact numerical inputs* that produce a metric is the single most reliable way to interpret it correctly.

We require three quantities: the estimated FLOPs required to train one token through the model, the aggregate token throughput processed by the cluster per second, and the theoretical peak FLOP/s of the GPUs in the target numeric precision.

$$
\mathrm{MFU}=\frac{F_{\mathrm{model/token}}\times R_{\mathrm{tokens/s}}}{N_{\mathrm{GPU}}\times C_{\mathrm{peak/GPU}}}
$$

Here, **token throughput** ($R_{\mathrm{tokens/s}}$) is the *actual* count of tokens processed per second across the entire training cluster. **Peak FLOP/s** ($C_{\mathrm{peak/GPU}}$) is the theoretical non-sparse ceiling of a single GPU in the specified precision. For an H100 SXM, dense BF16 peak throughput is approximately 989 TFLOP/s. This number cannot be applied to B200, B300, GB200, or GB300; one must select the exact specification dictated by the architecture and numeric format. [K16]

For a dense decoder-only Transformer, the ubiquitous first-order approximation is:

$$
F_{\mathrm{model/token}}\approx 6N_{\mathrm{param}}
$$

**Why 6?** Consider that a multiply-accumulate operation during the forward pass costs approximately **2 FLOPs** per parameter. During the backward pass, computing gradients with respect to both inputs and weights involves two comparable passes, contributing roughly **4 FLOPs**. This yields approximately 6 FLOPs/parameter/token. This canonical heuristic does not capture the sequence-length-dependent scaling of attention, embedding/unembedding layers, or architectural nuances. At long context lengths, a more granular analytical model is required; for this reason, Lambda's report incorporates an explicit attention correction term into the baseline formula. [K12]

**A concrete example:** Suppose we are training a 70B dense model on a cluster of **128 H100 SXM** GPUs, achieving an aggregate throughput of **100,000 tokens/second**. These values are not drawn from a published benchmark, but are selected to illustrate the arithmetic step by step.

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
\mathrm{MFU}\approx\frac{42}{126.6}=\mathbf{33.2\%}
$$

When someone cites "33% MFU," your immediate reaction should no longer be merely *why so low?*, but rather: **What was the exact token throughput, the FLOP/token accounting methodology, the arithmetic precision, the GPU count, and the measurement window?**

> **Measurement Rule:** For a given model and global batch size, do not conflate the transient MFU of an isolated kernel benchmark with the end-to-end *wall-clock* efficiency of a multi-week training campaign. Always disclose whether warm-up intervals, data loading stalls, checkpoint serialization, transient pauses, and recovery restarts are incorporated into the denominator.

### 19.8. There Is No Magic Number for 'What Constitutes a Good MFU?'

We cannot evaluate a vehicle's stop-and-go city fuel economy against its top speed on a racetrack. The same fallacy applies to MFU benchmarks: **a compact dense model, a multi-hundred-billion-parameter dense model, and a Mixture-of-Experts (MoE) architecture belong to entirely distinct comparison classes.** Even for an identical model, MFU measured at an 8K sequence length will diverge markedly from one measured at 128K.

The numbers below should therefore not be viewed as **universal target thresholds, but rather as empirical, published measurements under specific experimental configurations**:

| Study / System | Workload and Hardware | Reported Result | What This Number Conveys |
|---|---|---:|---|
| **Meta Llama 3** [K5] | 405B dense, 16,384 H100; varying context / parallelism setups | **38–43% MFU** | Robust production baseline at hyper-scale; highly context-dependent |
| **MegaScale (NSDI 2024)** [K13] | 175B dense, **12,288 GPUs** | **55.2% MFU** | Disproves the myth that '>45% MFU is unattainable on 10k+ GPUs' |
| **Lambda / Llama 3.1 8B** [K12] | 8× B200, BF16, **8K** sequence | **55.34% MFU** | Highly tuned dense training near single-node scale |
| **Lambda / Llama 3.1 8B** [K12] | 8× B200, BF16, **32K** sequence | **60.48% MFU** | Higher arithmetic intensity at longer sequences drives elevated efficiency |
| **Lambda / Llama 3.1 70B** [K12] | 16× B200, BF16, **8K** sequence | **50.20% MFU** | Massive uplift achieved once out-of-the-box defaults were tuned |
| **Lambda / Llama 3.1 405B** [K12] | 128× GB300, BF16, **8K** sequence | **52.69% MFU** | Modern rack-scale fabric and co-design sustain efficiency even on massive models |

A critical nuance: MegaScale's **55.2%** result and Lambda's **60.48%** figure do not form a straightforward "who wins?" leaderboard. The underlying hardware architectures, model geometries, FLOP definitions, sequence lengths, and measurement scopes differ substantially. Moreover, these represent **published empirical data points from 2024–2025**; they should not be construed as an exhaustive, definitive world-record leaderboard across all systems as of October 2026.

Nevertheless, a powerful conclusion emerges: **there is no law of physics dictating that ultra-large models can never surpass 40% MFU.** Rigorous communication overlap, well-architected parallelism strategies, and kernel tuning can dramatically push this boundary. [K13]

### 19.9. Doubling MFU Often Requires Configuration Tuning, Not More Silicon

Let us parse Lambda's empirical results through an engineering lens. When training **Llama 3.1 70B across 16× HGX B200 GPUs in BF16 at an 8K sequence length**, default TorchTitan configurations yielded only **23.83%** MFU. Following disciplined optimization, that figure surged to **50.20%**. [K12]

$$
\text{Improvement factor}=\frac{50.20}{23.83}\approx\mathbf{2.11\times}
$$

In this benchmark, neither the model architecture nor the training dataset was altered. The primary interventions included **dropping the tensor parallelism degree from 8 down to 1**, **transitioning from full activation checkpointing to selective activation checkpointing**, and refining batch and kernel tuning. The massive memory headroom and compute capabilities of Blackwell hardware illustrate why blindly inheriting configuration presets tuned for older GPU generations can severely impair efficiency.

Consider an intuitive analogy: rather than running a meeting where eight people must continuously synchronize every micro-decision, if the entire workload can fit onto a single worker's desk, we eliminate coordination overhead entirely. Of course, assigning the task to a single worker requires that their local desk space and processing speed be genuinely sufficient. This does not mean **TP=1 is universally optimal for every cluster architecture**; it simply highlights that parallelism overhead must be continually re-evaluated against hardware capacity.

| Configuration Evaluated on B200 Infrastructure | MFU | Commentary |
|---|---:|---|
| Llama 3.1 8B, 8× B200, default | 44.46% | Baseline TorchTitan configuration |
| Llama 3.1 8B, 8× B200, optimized | 55.34% | **1.24×** relative improvement |
| Llama 3.1 70B, 16× B200, default | 23.83% | Dominated by communication / memory overheads |
| Llama 3.1 70B, 16× B200, optimized | 50.20% | **2.11×** relative improvement |

*Source: Lambda, BF16/8K results from their 2025 MFU Technical Report [K12]. These figures are not automatically guaranteed for every workload.*

Let us examine this subtle nuance: boosting MFU from **23.83% to 50.20%** implies that, provided the workload and FLOP definitions remain constant, the useful token processing throughput on the identical silicon cluster increases by approximately **2.11×**. However, if training hyperparameters affect convergence dynamics, or if numeric precision is altered, the true end-to-end *time-to-quality* gain must be measured independently.

### 19.10. Overlap: Hiding Stalls by Layering Concurrent Work

Throughout this discussion—from warp schedulers to NVLink fabrics—we have encountered the identical design paradigm. Let us now materialize it in distributed training.

Suppose a given layer requires 8 ms of compute and 5 ms of inter-GPU collective communication. If executed sequentially:

$$
T_{\mathrm{serial}}=8+5=13\ \mathrm{ms}
$$

If independent compute and communication can be fully overlapped, the critical path latency becomes:

$$
T_{\mathrm{overlap}}\gtrsim\max(8,5)=8\ \mathrm{ms}
$$

This does not mean that the **five milliseconds of communication magically disappeared**. Rather, by executing independent compute during that window, the apparent latency of the communication is effectively *hidden*. In practice, strict data dependencies, SM contention, and network link arbitration often prevent 100% overlap.



<svg viewBox="0 0 720 254" role="img" aria-label="A layer with 8 ms of compute and 5 ms of communication. Run serially it takes 13 ms. If the independent communication fully overlaps the compute, the critical path is at least the maximum of 8 and 5, so 8 ms. That is a 38 percent shorter step, or 1.63 times the throughput on the same GPUs; the 5 ms of communication still happen, only off the critical path. Real overlap is partial because of dependencies, shared SMs and HBM, and network contention." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="om-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="om-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">8 ms compute + 5 ms communication: serial vs. fully overlapped</text>
<text x="16" y="74" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">serial</text>
<rect x="150" y="58" width="288" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="76.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">compute 8 ms</text>
<rect x="438" y="58" width="180" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="528.0" y="76.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">comm 5 ms</text>
<text x="626" y="77" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">= 13 ms</text>
<text x="16" y="136" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">overlapped</text>
<rect x="150" y="120" width="288" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="138.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">compute 8 ms</text>
<rect x="222" y="152" width="180" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="312.0" y="167.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">comm 5 ms (hidden)</text>
<text x="446" y="139" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">≥ max(8, 5) = 8 ms</text>
<line x1="438" y1="50" x2="438" y2="184" style="stroke:var(--c-success);stroke-width:1.2;stroke-dasharray:4 3"/>
<line x1="618" y1="50" x2="618" y2="96" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="16" y="196" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="216" text-anchor="start" style="fill:var(--c-text);font-size:12px">Step time −38% (13 → 8 ms) = 1.63× throughput on the same GPUs; the 5 ms still happen, just off the critical path.</text>
<text x="28" y="233" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Real overlap is partial: dependencies, shared SMs/HBM and network contention eat into it.</text>
</svg>



This paradigm has three critical production implementations:

**FSDP Prefetching:** Initiating the *AllGather* collective for the subsequent Transformer layer's sharded parameters before the current layer finishes its compute pass. While the current layer executes, parameters for the next are fetched across the fabric. The trade-off is transient memory overhead, as prefetched weights temporarily consume additional HBM headroom. In Lambda's 8B benchmarks, aggressive prefetching configurations were instrumental in driving MFU beyond 60%. [K12]

**Asynchronous Tensor Parallelism:** Pipelining and overlapping certain TP collective operations with independent GEMM or attention computations. This does not eliminate the physical bandwidth tax, but masks its latency when software and hardware capabilities (such as dedicated copy engines and stream priority) allow. [K12]

**DualPipe:** The bidirectional pipeline schedule developed for DeepSeek-V3 MoE training, which orchestrates forward/backward microbatch chunks alongside expert dispatch/combine AllToAll communication. DeepSeek's technical report specifies **PP=16**, **EP=64**, and **ZeRO-1 DP**, while intentionally avoiding expensive cross-node TP. The breakthrough of DualPipe stems not merely from scheduling theory, but from the tight co-design of custom AllToAll kernels mapped directly onto physical NVLink and InfiniBand topologies. [K9]

> **Intuitive Distinction:** A pipeline bubble represents a worker at an assembly station standing idle while waiting for upstream parts. Communication overlap is that same worker assembling a secondary, already-available batch of parts while a transport truck is en route. DualPipe applies this concept symmetrically between forward and backward tasks in an interleaved pipeline. It is inaccurate to claim that it "eliminates all bubbles under every conceivable circumstance."

### 19.11. FP8, BF16, and MFU: Fewer Bits Can Yield Higher Throughput while MFU Drops

BF16 numbers use 16 bits; FP8 numbers use 8 bits. At first glance, one might assume: "Data precision halved, so MFU will surely double." In reality, **MFU reflects a ratio of two moving targets:** while useful FLOP/s may increase, the hardware's **theoretical peak FLOP/s** ceiling also doubles (or quadruples) for that numeric format.

For example, in NVIDIA's non-sparse specifications, an H100 delivers approximately **989 TFLOP/s** in BF16 and **1,979 TFLOP/s** in FP8; a B200 delivers **2,250 TFLOP/s** in BF16 and **4,500 TFLOP/s** in FP8. [K16]

Suppose that by switching to FP8, your achieved compute rate climbs to 1,200 TFLOP/s:

- In BF16: 800 / 989 ≈ **81%** MFU.
- In FP8: 1,200 / 1,979 ≈ **61%** MFU.

In the second scenario, your **actual execution speed increased by 50%**, yet your **MFU percentage plummeted by 20 percentage points**! This thought experiment explains why cross-precision performance cannot be evaluated solely by comparing raw MFU percentages.

It is precisely for this reason that Lambda chose to report **achieved TFLOP/s** rather than raw MFU in their mixed-precision BF16+FP8 studies. In their 70B benchmark on B200 silicon, the default configuration yielded approximately **812 TFLOP/s**, whereas their optimized setup reached approximately **1,783 TFLOP/s**—with the precise metrics qualified by the study's measurement boundaries. [K12]

FP8 training also introduces substantial operational complexity: dynamic scaling factors (*scales*), on-the-fly format casting, selective layer-by-layer precision policies, and loss convergence safeguards. **Quantization** is never simply about shrinking memory footprints; it is the joint orchestration of data movement, specialized Tensor Core micro-kernels, and numerical error budgets. [K17]

Looking at the formats themselves explains why scale factors are unavoidable:

| Format | Sign | Exponent | Mantissa | Approx. largest value | Typical role |
|---|---:|---:|---:|---:|---|
| FP16 | 1 | 5 | 10 | 65,504 | Older mixed precision, inference |
| BF16 | 1 | 8 | 7 | ~3.4 × 10³⁸ | Default for pretraining: same exponent range as FP32 |
| FP8 E4M3 | 1 | 4 | 3 | 448 | Forward pass: weights and activations |
| FP8 E5M2 | 1 | 5 | 2 | 57,344 | Backward pass: gradients (wider range) |
| NVFP4 (E2M1) | 1 | 2 | 1 | 6 (× an E4M3 scale per 16-value block) | Low-bit GEMMs on Blackwell, mainly inference |

Read it as: exponent bits set the representable range, mantissa bits set the precision. Because E4M3 tops out at 448, tensors are first squeezed into range with a scale factor; get the scale wrong and large values overflow while small ones flush to zero. NVFP4 takes this a step further: every 16-value block carries its own FP8 (E4M3) scale and the whole tensor carries an extra FP32 scale, a finer grain than MXFP4's 32-value blocks with power-of-two scales. [K17][K57]

**Key Takeaway:** When comparing FP8 against BF16, MFU must be evaluated alongside **useful tokens/s, model quality benchmarks, loss convergence curves, the exact precision accounting methodology, and cost per million tokens**. Above all, the identical FLOP/token heuristic cannot be indiscriminately applied across Dense and MoE architectures.

### 19.12. Kernel Fusion: Preparing the Dish in One Pan Instead of Three

Imagine a professional kitchen where a cook chops onions, thoroughly cleans the counter, chops tomatoes, cleans the counter again, and then prepares peppers. Each independent operation incurs tool-switching overhead and reset delays. Consolidating compatible prep work onto the same cutting board dramatically slashes total prep time.

Similarly, **kernel fusion** on a GPU merges compatible sequential operations into a single kernel launch. This minimizes CPU-side launch overhead and eliminates the round-trip memory bandwidth tax of spilling intermediate tensors to HBM only to read them back immediately.

For instance, in Transformer multi-head attention, the Query, Key, and Value projections share the identical input activation tensor. With an aligned weight layout:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

rather than launching three distinct GEMM kernels, the projection weight matrices can be concatenated:

$$
[Q,K,V]=X[W_Q,W_K,W_V]
$$

allowing the operations to execute as a single large GEMM. This is widely known as **QKV merging**. Other standard examples include fused optimizer updates (combining weight decay, momentum, and parameter updates) and chained elementwise operations (such as Bias-GELU or LayerNorm-Residual fusions). [K12]

**FlashAttention-3** represents a far more sophisticated realization of this principle: by tiling softmax and matrix multiplications into cache-sized blocks, it eliminates materialization of the full $N \times N$ attention matrix in HBM, exploiting Hopper's asynchronous data movement and TMA (Tensor Memory Accelerator) pipelines. However, claiming that "the entire attention computation resides purely in SRAM" is technically inaccurate; tiles and partial accumulation states are dynamically staged across memory hierarchies. [K8]

**`torch.compile`** captures supported PyTorch execution subgraphs to synthesize JIT-compiled, fused inductor kernels. However, it does not magically turn every model into a single uninterrupted static computation graph under all circumstances; *graph breaks*, recompilation hitches, and device-side fallbacks can occur. Claiming that it automatically drives Python overhead to **zero** is an overstatement. [K14]

In Lambda's profiling traces for an optimized 8B model, the share of execution time occupied by **elementwise operations dropped from approximately 14.8% down to 1.5%**, while the relative share of matrix multiplications shifted from **56.4% to 68.7%**. These are **percentages of total elapsed step duration**; one must not prematurely jump to the conclusion that absolute matmul duration increased or decreased on its own, as batch sizing and compilation passes shifted concurrently. [K12]

### 19.13. The Activation Checkpointing Paradox: Less VRAM Does Not Guarantee Higher MFU

During training, we could theoretically retain every intermediate activation tensor generated during the forward pass in memory. Doing so, however, rapidly exhausts available HBM. **Activation checkpointing** mitigates this by discarding selected intermediate activations and recomputing them on-the-fly during the backward pass.

Intuitively, think of a student solving a complex derivation: rather than writing down every trivial arithmetic step in their notebook, they only record pivotal intermediate milestones. When checking their work later, they re-derive the omitted steps. The notebook stays slim, but the re-derivation consumes additional time.

Here, we must reinforce the clear demarcation between **disk checkpointing** and **activation checkpointing**: the former persists model training state to durable storage for fault recovery, whereas the latter recycles transient GPU memory required for backward autodiff graphs.

| Strategy | HBM Memory Savings | Recomputation Overhead | Expected Directional Impact on MFU |
|---|---|---|---|
| Cache All Activations | None | Zero | Typically optimal if memory capacity allows |
| **Selective** Checkpointing | Moderate | Minimal | Frequently strikes an optimal trade-off |
| **Full** Checkpointing | High | Substantial (~33% more compute) | MFU often degrades to free up VRAM |

The crucial nuance: **FLOPs executed during recomputation do not count toward useful model FLOPs in the MFU numerator.** Consequently, while HFU may rise, MFU can actively fall. Nevertheless, if recomputation frees enough memory to enable a significantly larger microbatch size or avoid Out-Of-Memory (OOM) crashes, aggregate training throughput may still improve. PyTorch's official activation checkpointing documentation elaborates extensively on this memory-compute trade-off. [K15]

This is precisely what the Lambda benchmarks underscore: first, minimize unnecessary recomputation whenever memory permits; if memory tightens, evaluate a **selective** checkpointing strategy; and only if memory constraints remain prohibitive, escalate to full recomputation or distributed model parallelism. **This is not an invariant dogma**, but an engineering decision roadmap. [K12]

### 19.14. Why MoE MFU Cannot Be Directly Compared to Dense Models

In a dense model, nearly every parameter in the architecture participates in computation for every single token. In a **Mixture-of-Experts (MoE)** architecture, a dynamic gating router directs each token to a specialized subset of experts. Consequently, the model's **total parameter count** diverges dramatically from its **active parameter count** per token.

In the case of DeepSeek-V3, the architecture encompasses **671B total parameters**, but activates only approximately **37B active parameters** per token. The technical report cites a training budget of 14.8 trillion tokens consumed across 2.788 million H800 GPU hours. [K9]

Applying the simplistic dense heuristic to calculate $6\times671B$ FLOPs would be fundamentally invalid: not all experts execute for every token. Conversely, computing simply $6\times37B$ underestimates total work, failing to capture dense shared attention layers, router overhead, multi-token prediction heads, and heavy communication costs.

In MoE distributed architectures, we must rigorously distinguish the following components:

- **Routing:** Algorithmic assignment of incoming tokens to designated top-$k$ experts.
- **AllToAll:** Inter-GPU exchange routing token activations to target expert ranks and recombining expert outputs.
- **Load balancing:** Dynamic mechanisms preventing expert starvation or hotspot congestion.
- **Expert Parallelism (EP):** Partitioning discrete experts across GPUs in the cluster.
- **DualPipe overlap:** Overlapping AllToAll transfer latencies with independent dense forward and backward GEMM operations.

Regarding claims circulating in secondary analysis stating that **"DeepSeek-V3 achieved exactly 21.4–23% MFU," I cannot verify this specific percentage directly from the primary technical report**. That figure appears to be an extrapolation derived in third-party literature; without explicit disclosure of the exact FLOP/token definition and the FP8 peak baseline used in the denominator, it should not be treated as an official verified measurement. What can be authoritatively cited from DeepSeek's primary paper are the architectural specifications, the FP8 mixed-precision framework, the DualPipe scheduling mechanics, the total GPU hours, and the dataset token volume. [K9]

In other words, **an MoE model displaying a lower nominal MFU does not indicate that it is more expensive, slower, or poorly optimized**. Active compute per token, downstream model benchmark quality, aggregate training duration, and total dollar cost must be interpreted holistically.

### 19.15. The Methodology Behind an MFU Optimization Campaign

If presenting these findings to an architecture steering committee, proposing an arbitrary goal like "let's hit 60% MFU" is ineffective; we must structure our efforts around a rigorous, measurable experimental plan.

**Phase 1 — Establish a Reproducible Baseline:** Execute a steady-state run across a sufficient number of steps using identical model checkpoints, sequence lengths, global batch sizes, precision formats, GPU topologies, and data sampling seeds. Record both the theoretical FLOP/token formula and actual tokens/second. Document explicitly how warm-up steps and checkpoint I/O are handled.

**Phase 2 — Map the Critical Path of Time:** In detailed profiler traces, isolate durations for GEMMs, attention kernels, pointwise operations, optimizer steps, NCCL collectives, pipeline bubbles, and dataloader waits. Identify the dominant bottleneck before touching code.

**Phase 3 — Implement Low-Risk, High-Reversibility Tweaks:** Begin with low-hanging software optimizations: evaluate `torch.compile`, verify fused kernels and QKV merging, confirm pinned host memory and proper CPU core affinity, and sweep microbatch dimensions. Vary only one parameter per test.

**Phase 4 — Re-evaluate Parallelism and Memory Hierarchies:** Do not copy tensor parallelism degrees from prior-generation clusters. If HBM capacity allows, reduce TP degree, benchmark selective activation checkpointing, and prioritize data parallelism. If memory constrained, evaluate hybrid sharded parallelism (FSDP/HSDP), calibrated TP, and Context Parallelism (CP) tailored to long sequences. [K12]

**Phase 5 — Verify Empirically that Overlap Occurs:** Enabling configuration flags for FSDP prefetch, async TP, or DualPipe schedules does not guarantee success. Profile execution timelines in Nsight Systems to measure precisely how much collective communication latency remains *exposed* on the critical path.

**Phase 6 — Co-optimize Precision with Throughput and Numerical Quality:** Do not compare BF16 against FP8 solely on nominal MFU percentages. Report achieved TFLOP/s, end-to-end token throughput, loss convergence stability, and validation evaluation metrics concurrently. [K12][K17]

**Phase 7 — Audit Real Business Impact:** Lifting MFU from 40% to 50% is meaningless if cluster MTBF plummets, checkpoint stalls balloon, or loss diverges. The ultimate North Star metrics must remain **time-to-quality, total GPU hours consumed, operational dollar cost, and cluster resilience**.

> **Core Intuition in One Sentence:** First, measure where time is genuinely wasted; then bring data closer to compute, eliminate unneeded communication, overlap inevitable stalls with productive work, and re-benchmark gains under strictly identical conditions.

---

### 19.16. Beyond MFU: Which Metrics Define a Truly Superior Architecture?

Suppose a manufacturing line runs at 90% utilization but churns out defective or obsolete goods. Another plant runs at 65% machine utilization, yet delivers the exact products customers need on time and at lower total cost. Which system is better engineered?

MFU is an invaluable yardstick for **training computational efficiency**, but it cannot serve as the overarching success metric of an entire enterprise AI infrastructure.

Let us delineate metrics across their proper domains:

| Metric | What It Measures | What It Fails to Reveal in Isolation |
|---|---|---|
| **Job MFU** | Ratio of achieved model FLOPs to theoretical peak for an active job | Cluster-wide fleet utilization or idle capacity |
| **Fleet allocation/utilization** | Percentage of total cluster GPU capacity assigned/busy | Whether allocated jobs execute arithmetic efficiently |
| **Training tokens/s** | Raw ingestion and processing rate of tokens | Whether the model is learning effectively or converging properly |
| **Time-to-quality** | Wall-clock time required to reach a target validation loss or benchmark score | Micro-level kernel execution efficiency |
| **Serving goodput** | Useful inference requests completed strictly within predefined SLO targets | The magnitude of compute capacity wasted on SLO-violating requests |
| **Cost per million tokens** | Unit economic expenditure for serving inference | Output quality degradation or tail latency violations |
| **Tokens per joule** | Computational energy efficiency per completed token | Multi-tenant fairness or systemic operational risk |

**Goodput** represents the rate of *useful* system output that strictly adheres to agreed Service Level Objectives (SLOs). For instance, if an inference server produces a high aggregate token volume but consistently breaches Time-to-First-Token (TTFT) thresholds, raw throughput looks impressive while the actual user experience is unacceptable. The MLPerf Inference benchmark methodology emphasizes this principle by jointly profiling interactivity and throughput frontiers across varying concurrency loads. [K34]

**Architectural Objective for Training:** Compare which system topology reaches the target evaluation threshold with the lowest overall wall-clock time, GPU-hour expenditure, energy footprint, and failure risk.

**Architectural Objective for Inference:** Under strictly matched quality and safety parameters and representative prompt/generation token distributions, measure which serving architecture satisfies p95/p99 TTFT and TPOT targets at the lowest unit cost per token.

> **Food for Thought:** Upgrading model architecture, pruning prompts, improving KV cache hit rates, or implementing semantic request routing can frequently unlock greater gains in *perceived service quality* than procuring more GPUs. Making that determination requires profiling not only hardware counters, but production workload distributions.

### 19.17. Benchmarking Is Not a One-Off Run, but Experimental Design

To rigorously benchmark AI platform performance, execute these four systematic steps:

**1 — Lock Down Experimental Conditions:** Unless model commit hash, precision format, kernel revisions, driver/CUDA/NCCL stack, GPU SKU and count, network topology, input/output token length distributions, dataset splits, and client concurrency are held invariant, directly comparing two benchmark runs is invalid.

**2 — Isolate and Benchmark Each Subsystem Layer:** Execute targeted microbenchmarks: GEMM and memory bandwidth for the GPU; NCCL collective sweeps for the interconnect; parallel asynchronous I/O runs for storage; and representative steps for the end-to-end framework. A single end-to-end latency metric cannot locate the root cause.

**3 — Capture Tail Latency and Temporal Variance:** An average step duration may look stellar, but if every hundredth step stalls by 2×, scheduling jitter, garbage collection, or network stragglers are at play. In inference serving, tracking p95 and p99 distributions for TTFT and TPOT is non-negotiable.

**4 — Design Counterfactual Experiments per Optimization:** When scaling from TP=4 to TP=8, hold every other parameter strictly constant and measure the delta in step time, memory footprint, NCCL exposure, and numerical stability. Concurrently altering multiple variables muddies the causal source of performance gains.

A structured observability stack can be organized as follows:



<svg viewBox="0 0 720 302" role="img" aria-label="Observability stack. Four signal sources: GPU and node telemetry from DCGM, such as power, ECC, Xid and clocks; fabric telemetry from NICs, RDMA counters and NCCL RAS; training job metrics such as step time, MFU, loss and restarts; and serving metrics such as TTFT, TPOT, queue and KV usage. They flow through the DCGM exporter and Prometheus or OpenTelemetry into one metrics and trace store keyed by a trace ID per request, which drives action: SLO burn-rate alerts, node quarantine, capacity planning and cost per useful token. The DCGM exporter covers GPU fields only; the rest must be instrumented separately." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ob-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ob-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Observability: four signal sources, one place to correlate them</text>
<rect x="16" y="44" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="126.0" y="66.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU &amp; node</text>
<text x="126.0" y="80.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">DCGM: power, ECC, Xid, clocks</text>
<path d="M236 68 C 262 68, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="102" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="126.0" y="124.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Fabric</text>
<text x="126.0" y="138.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NIC, RDMA counters, NCCL RAS</text>
<path d="M236 126 C 262 126, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="160" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="182.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Training job</text>
<text x="126.0" y="196.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">step time, MFU, loss, restarts</text>
<path d="M236 184 C 262 184, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="218" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="126.0" y="240.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Serving</text>
<text x="126.0" y="254.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">TTFT, TPOT, queue, KV usage</text>
<path d="M236 242 C 262 242, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="292" y="116" width="190" height="88" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="387.0" y="143.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Metrics + traces</text>
<text x="387.0" y="158.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">DCGM exporter →</text>
<text x="387.0" y="172.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Prometheus / OpenTelemetry</text>
<text x="387.0" y="187.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">one trace_id per request</text>
<line x1="482" y1="160" x2="512" y2="160" marker-end="url(#ob-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="514" y="106" width="190" height="108" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="609.0" y="136.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Act</text>
<text x="609.0" y="150.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">SLO burn-rate alerts</text>
<text x="609.0" y="165.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">node quarantine</text>
<text x="609.0" y="179.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">capacity planning</text>
<text x="609.0" y="194.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">cost per useful token</text>
<text x="16" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DCGM exporter covers GPU fields only; network, application and user-experience metrics must be instrumented separately.</text>
</svg>



The NVIDIA **DCGM Exporter** exposes low-level GPU hardware telemetry counters in Prometheus-compatible exposition formats. However, it does not capture software-level framework metrics, network fabric switch counters, or end-user application latency; these require deliberate instrumentation across application and infrastructure layers. [K35]

#### A Concrete Sizing Heuristic: Little's Law

Consider an inference service receiving an arrival rate of $\lambda = 6$ requests per second, where each request resides in the system for an average latency of $W = 8$ seconds. Assuming steady-state conditions and long-term averages:

$$
L=\lambda W=6\;\mathrm{request/s}\times8\;\mathrm{s}=48\;\mathrm{request}
$$

This indicates that the serving system holds an average of **48 concurrent in-flight requests**; it does not mean that 48 GPUs are required. A single GPU worker can serve multiple concurrent requests via continuous batching, and individual requests exhibit varying KV cache footprints.

**Important:** Sizing cluster capacity strictly against average concurrency is a recipe for operational failure. Ingress traffic spikes, queuing policies, and tight p99 tail latency guarantees necessitate stress testing and deliberate headroom buffers.

---

---

### Visualizing MFU, GPU Utilization, and Realized Value on a Unified Horizon

Consider two parallel training runs. Run A achieves 55% MFU, but after 10,000 steps its validation loss diverges or lags behind expectations. Run B clocks 45% MFU, but maintains superior numerical stability and converges to the target evaluation loss significantly faster in wall-clock time. Which setup is superior? Judged solely on MFU, Run A appears victorious; evaluated on true time-to-quality, Run B delivers vastly greater engineering value.

For this reason, performance engineering spans three distinct tiers:

1. **Hardware Efficiency:** GPU utilization percentage, Tensor Core throughput, memory bandwidth saturation, and network fabric utilization.
2. **Model Progress Efficiency:** MFU/HFU, step latency, training tokens/s, loss convergence trajectories, and time-to-target-accuracy.
3. **Product Efficiency:** Cost per serving request, p95/p99 TTFT, request completion rate, and useful tokens/s delivered to downstream users.

A stellar showing in Tier 1 may fail completely to translate into Tier 2 or Tier 3 outcomes. This is why MLPerf Training mandates convergence to an exact quality target as an invariant requirement, preventing benchmarks from degenerating into empty raw-compute races. [K50]

### Why p99 Blows Up Despite Solid Throughput: A Simple Queuing Experiment

Imagine a service counter capable of processing an average service rate of $\mu = 10$ customers per second. Contrast two different arrival rates: $\lambda = 8$ customers/s versus $\lambda = 9.5$ customers/s. In a simplified $M/M/1$ queue, the average time spent in the system is governed by:

$$
W=\frac{1}{\mu-\lambda}.
$$

For $\mu=10$, when $\lambda=8$, $W = 0.5$ seconds; when $\lambda=9.5$, $W = 2$ seconds. While arrival traffic grew by only 18.75%, the average residence time quadrupled.

This toy model does not fully encapsulate the nuances of modern GPU serving engines—it abstracts away continuous batching dynamics, variable token sequence lengths, multi-worker parallelism, KV cache exhaustion, and scheduler preemption policies. Yet it illustrates a fundamental queuing reality: **as system load approaches saturation capacity, marginal increases in request arrival rates induce exponential tail latency spikes.** This is precisely why p99 latency serves as an essential architectural signal.

### Why Measurement Granularity Matters

Running `nvidia-smi` may confirm that "the GPU is running hot and active"; however, answering "which kernel, in what time slice, stalled by memory bandwidth or blocked by collective communication?" requires granular timeline traces and hardware performance counters. NVIDIA Nsight Systems coordinates host CUDA API calls, GPU engine executions, network exchanges, and CPU runtime events on a synchronized timeline. Nsight Compute inspects low-level warp scheduler metrics and pipeline counters within specific kernels. [K47]

In any rigorous engineering analysis, every architectural assertion should be accompanied by its corresponding **level of evidence**:

| Technical Claim | Sufficient Evidence Source? | Rationale |
|---|---|---|
| “The GPU is active” | GPU utilization | Confirms coarse activity, nothing more |
| “Tensor Core utilization is sub-optimal” | Detailed kernel profile | Directly inspects functional MMA execution units |
| “We are hitting an interconnect bottleneck” | NCCL timeline + NIC/fabric telemetry | Proves exposed stalls are bound to network transit |
| “Storage is bottlenecking training progress” | Checkpoint timeline + storage telemetry | Distinguishes snapshot memory pause from durable write times |
| “System A is strictly superior to System B” | Convergence quality + SLO compliance + TCO + replicated runs | Ties architectural performance directly to delivered business value |

## 20. Source-Verified Large-Scale Case Studies

Theory teaches a great deal, but when designing large-scale systems, one must also examine reported real-world empirical results. The two examples below demonstrate two different approaches to the same underlying challenge: maintaining system integrity during massive dense model training, and intelligently overlapping computation with communication in MoE models.

### 20.1. Meta Llama 3 405B: Balanced Progress Across Massive GPU Fleets

Meta's technical report indicates that the 405-billion-parameter model was trained across 16,384 H100 GPUs across various context lengths. Among the reported configurations, approximately **41% MFU** at an 8K context and **38% MFU** at a 128K context were achieved. The system leveraged **Grand Teton**-based servers, a **RoCE** compute fabric, **Tectonic** distributed storage, and automated fault recovery mechanisms. [K5]

This case study illustrates a success born not from a single silver bullet, but from a **cohesive matrix of engineering decisions**: model partitioning, network topology, checkpointing, and failure automation all converged toward the same overarching goal.

### 20.2. DeepSeek-V3: Overlapping Computation with Network Communication in MoE

In MoE architectures, tokens must traverse the network fabric to reach their assigned experts. When the individual experts are relatively small, the computational workload per token and the communication overhead of token dispatching can reach parity.

The DeepSeek-V3 report details the use of **DualPipe** to overlap forward/backward compute passes with AllToAll dispatch/combine collectives and pipeline parallelism bubbles. The report cites a 671B total parameter architecture with 37B active parameters per token, trained at a cost of 2.788 million H800 GPU hours. These figures reflect the specific training setup reported in the paper rather than invariant metrics transferable to arbitrary clusters. [K9]

The report's training layout is concrete: 16-way PP, 64-way EP spread across 8 nodes and ZeRO-1 DP, with no costly TP at all. DualPipe feeds microbatches from both ends of the pipeline at once and overlaps forward and backward compute with dispatch/combine communication in every chunk; the report states that this keeps the cross-node AllToAll overhead **near zero** even with fine-grained experts. With a global batch of 15,360 sequences × 4,096 tokens ≈ 62.9 million tokens, each of the 2,048 H800s processes ~30,720 tokens per step: that is how a run approaches the MoE-inflated critical batch of §13.6. [K9]

**Instructive contrast:** In dense models, large GEMMs saturate compute units efficiently, but the sheer volume of weights imposes severe memory constraints. MoE substantially curtails active computation per token, yet intensifies expert routing overhead and AllToAll network traffic. Mitigating one bottleneck invariably exposes another.

---

### 20.3. Lambda and MegaScale: How Much Difference Does Better Utilization of the Same GPUs Make?

The Meta case study established a robust large-scale baseline. Now let us examine a second class of engineering success: orchestrating a superior execution pipeline on identical hardware without altering the underlying model architecture.

Lambda's **November 2025** benchmark report evaluates the 8B, 70B, and 405B variants of Llama 3.1 across **HGX B200** and **GB300 NVL72** clusters. Operating in BF16 with an 8K sequence length, the 70B model's MFU surged from 23.83% under baseline TorchTitan settings to 50.20% with a tuned training recipe. Similarly, for the 405B model evaluated across 128× GB300 GPUs, MFU rose from 28.62% to 52.69%. In the **8B/32K** configuration, Lambda recorded an MFU of **60.48%**. [K12]

The most insightful takeaway from these findings is not merely the peak numbers, but **the clear diagnosis of why performance lagged initially**: overly small micro-batches, excessive tensor parallelism (TP), aggressive activation checkpointing, stalled FSDP AllGather collectives, and unnecessary pointwise/optimizer kernel overhead. Rather than scaling out with more GPUs, the solution lay in systematically eliminating these specific execution taxes.

MegaScale addresses a complementary question: *Do such optimizations hold up when scaling across tens of thousands of GPUs?* Published at NSDI 2024, the study demonstrates **55.2% MFU** for a **175B parameter model trained across 12,288 GPUs**, achieving a **1.34× speedup** over baseline Megatron-LM. The architecture jointly tackles computation-communication overlapping, operator-level fusion, data pipeline bottlenecks, and automated straggler mitigation. [K13]

The conclusion is unequivocal: **MFU is not an intrinsic property of the GPU hardware alone, but the emergent product of end-to-end full-stack co-design.** Software refinements on identical silicon can yield more than a 2× performance delta; nor does hyperscale cluster sizing inherently preclude achieving sustained MFUs exceeding 40% to 50%.

## 21. Five Major Misconceptions and Their Technical Realities

At this juncture, it is critical to debunk several catchphrases that, while catchy and easy to memorize, fatally misdirect engineering decisions. Pithy assertions stick in memory during technical presentations, but overgeneralized rules of thumb frequently lead to misguided hardware capital expenditures or counterproductive optimization campaigns.

| Flawed or Overly Rigid Claim | Nuanced Technical Reality |
|---|---|
| **"H100 decode MFU is strictly 1–2%."** | While the theoretical arithmetic intensity of single-token GEMV is minimal, real-world MFU depends heavily on concurrency (batching), model FLOP definitions, and surrounding kernel execution. |
| **"TP can never scale past 8 GPUs."** | An 8-GPU NVLink domain represents the optimal physical locality for H100 DGX nodes, but it does not represent an inescapable universal upper limit for tensor parallelism. |
| **"SHARP accelerates every AllReduce by exactly 2×."** | While offloading reduction operations reduces network transit volume and latency for supported operations, end-to-end application speedup is strictly workload- and topology-dependent. |
| **"GDS completely eliminates the CPU from the data path."** | GDS bypasses host bounce buffers for bulk DMA transfers, but CPU-driven I/O orchestration, metadata management, and fallback paths remain essential components. |
| **"Writing a 7.5 TB checkpoint in one second solves all persistence problems."** | Practical recovery guarantees require sustained storage/fabric throughput, scalable metadata formats, fault domain resilience, and minimized job pause latencies. |

Two additional critical clarifications:

- **405B state = 7.5 TB** is not a universal constant. The footprint varies with the inclusion of weights, gradients, optimizer states, FP32 master weights, and checkpoint snapshots. Assuming standard 16 bytes per parameter (FP16 weights + FP16 grads + FP32 master weights + Adam first/second moments), the baseline model state equals **6.48 TB**.
- **419 vs. 466 interruptions** are not competing metrics. In the Llama 3 technical report, out of 466 total training interruptions, 419 were classified as **unexpected/unplanned** outages. [K5]

---

## 22. Observability: What Metrics and Traces Reveal About a System

The case scenarios detailed below do not constitute a rigid troubleshooting playbook; rather, they are structured to illustrate how distinct underlying physical failure modes can produce identical macroscopic symptoms.

The health of a GPU cluster cannot be assessed through a single lens. **Telemetry** answers "what is happening" across aggregated fleet health; **profiling** zooms in to the kernel level to answer "why it is happening"; and **distributed tracing** tracks how an individual user request or training iteration traverses microservices and hardware boundaries.

### 22.1. Three-Tier Visibility Architecture

| Tier | Representative Tools | Key Question Addressed |
|---|---|---|
| Hardware | DCGM, NVML, NIC telemetry counters | Is the GPU throttling; what are the error rates on HBM, NVLink, and PCIe? |
| Runtime / Kernel | Nsight Systems, Nsight Compute, PyTorch Profiler | Which kernel, launch gap, or collective communication call is dominating step time? |
| Service | Prometheus, OpenTelemetry, distributed tracing | Why did TTFT spike by 3 seconds for this specific tenant request? |

DCGM's sampled telemetry cannot resolve per-kernel micro-timings, just as Nsight Systems execution traces cannot be run continuously in production without non-trivial profiling overhead. **Continuous fleet telemetry** and **targeted profiling captures** serve distinct operational roles and must not be conflated. [K36][K37]

### 22.2. Minimum Viable Schema for an Inference Request Trace

```text
trace_id, request_id, tenant_id_hash, model_version, runtime_version,
prompt_token_count, output_token_count, image_count, context_length,
auth_ms, safety_ms, data_access_ms, queue_ms, prefill_ms,
first_decode_ms, inter_token_p95_ms, e2e_ms,
cache_hit, gpu_pool, worker_id, result_status, cancel_reason
```

This represents an **exemplar schema** for structuring telemetry spans. Because tenant IDs and request metadata may contain sensitive or identifying information, trace collection must adhere strictly to organizational data minimization and retention policies.

### 22.3. Diagnostic Logic: Root-Cause Layer Mapping for Common Symptoms

<svg viewBox="0 0 720 364" role="img" aria-label="Root-cause triage tree. Start from the symptom that an SLO or MFU dropped. First ask whether the error rate rose, such as Xid, ECC or restarts; if yes, treat it as a hardware or driver fault and quarantine the node. If not, find where the wait is in the profile timeline: in the queue, pointing at admission and pool capacity; in GPU compute, pointing at kernel memory and batch size; in the network, pointing at NCCL collectives and NIC counters; or in storage and I/O, pointing at the dataloader and checkpoint writes. Then test one hypothesis with one variable changed." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dt-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dt-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Root-cause triage: is it broken, or is it waiting?</text>
<rect x="16" y="128" width="150" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="91.0" y="154.0" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SLO / MFU dropped</text>
<text x="91.0" y="169.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">symptom</text>
<rect x="212" y="52" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="307.0" y="78.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Error rate up?</text>
<text x="307.0" y="92.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Xid, ECC, restarts</text>
<rect x="212" y="204" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="307.0" y="230.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Where is the wait?</text>
<text x="307.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">profile timeline</text>
<path d="M166 146 C 190 146, 190 80, 210 80" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M166 166 C 190 166, 190 232, 210 232" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="52" width="254" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="577.0" y="78.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Hardware / driver fault</text>
<text x="577.0" y="92.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">quarantine the node (see §17.5)</text>
<line x1="402" y1="80" x2="448" y2="80" marker-end="url(#dt-b)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<text x="425" y="74" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px;font-weight:600">yes</text>
<path d="M307 108 V198" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="315" y="160" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">no</text>
<rect x="450" y="136" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="154.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Queue</text>
<text x="577.0" y="169.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">admission, pool capacity</text>
<path d="M402 232 C 426 232, 426 157, 448 157" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="186" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="577.0" y="204.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU compute</text>
<text x="577.0" y="219.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kernel memory, batch size</text>
<path d="M402 232 C 426 232, 426 207, 448 207" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="236" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="577.0" y="254.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Network</text>
<text x="577.0" y="269.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NCCL collectives, NIC counters</text>
<path d="M402 232 C 426 232, 426 257, 448 257" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="286" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="577.0" y="304.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Storage / I/O</text>
<text x="577.0" y="319.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">dataloader, checkpoint writes</text>
<path d="M402 232 C 426 232, 426 307, 448 307" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Then the same discipline every time: one hypothesis, one variable changed, before/after measured.</text>
</svg>

### 22.4. Case Scenario A: Why Would MFU Collapse from 38% to 22%?

**Step 1: Metric Verification.** Confirm identical baseline conditions: model architecture, global/micro batch size, sequence length, floating-point precision, and active GPU count. A change in the theoretical peak denominator will skew MFU calculations. Library version updates or instrumentation discrepancies can also artificially alter measured throughput.

**Step 2: Timeline Phase Breakdown.** Isolate the step time into distinct execution phases on the profiler timeline: compute kernels, inter-GPU collective communication, data loader ingest, checkpoint I/O serialization, and optimizer weight update kernels.

**Step 3: Straggler vs. Systemic Localization.** If a subset of ranks lags behind, diagnose straggler nodes (e.g., PCIe degradation, thermal throttling, bad NIC). If all GPUs exhibit uniform deceleration, evaluate shared infrastructure bottlenecks (storage fabric congestion, spine switch packet loss, or a software regression).

**Step 4: Controlled Rollback and Isolation.** Execute an A/B regression test against the last known good container image and configuration snapshot. Never adjust multiple NCCL environment tuning knobs simultaneously without systematically benchmarking each isolated parameter.

### 22.5. Case Scenario B: Why Is Token Throughput Low Despite 100% `nvidia-smi` Utilization?

**Probable Causes:** Batch=1 autoregressive decode running in a strict memory-bandwidth-bound regime, sub-optimal batch sizes, KV cache allocation contention, CPU-side token pre/post-processing bottlenecks, or un-fused, inefficient quantization kernels. Coarse GPU utilization percentages reported by `nvidia-smi` simply reflect that a kernel is active on the SMs, not that execution units are saturated with useful FLOPs. Segment the workload by prompt and output length; simultaneously correlate output token/s, time per output token (TPOT), effective HBM throughput, and kernel-level SM activity.

### 22.6. Case Scenario C: Why Does NCCL `all_reduce` Hang?

First, confirm whether distributed execution has truly stalled and pinpoint which ranks are unresponsive. Depending on the version, NCCL's **RAS** (Reliability, Availability, and Serviceability) telemetry can surface active job states and uncommunicative ranks. Newer NCCL releases also incorporate active diagnostic routines; note that passive RAS merely exposes counters and does not actively inject test traffic. [K38]

Follow a systematic diagnostic hierarchy: rank environment consistency -> NIC and GPU visibility -> binding to the correct network interface (`NCCL_IB_HCA`) -> GPU-to-NIC PCIe/NUMA affinity -> physical link flap and error counters -> two-node `nccl-tests` -> multi-node rack-scale benchmarks. This methodical progression isolates whether the root cause stems from runtime configuration errors, fabric routing failures, or localized hardware degradation.

```bash
# Initial read-only sanity checks. Exact commands may vary depending on environment.
nvidia-smi -L
nvidia-smi topo -m
# Single-node AllReduce sweep using nccl-tests (if compiled)
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

Do not expose NCCL RAS telemetry endpoints unauthenticated across the enterprise network; restrict diagnostic interfaces strictly to privileged administrative host and fabric boundaries.

### 22.7. Case Scenario D: Why Does p99 TTFT Spike While Average Latency Appears Healthy?

Examine micro-burst queue build-ups, pending prefill token volume, cold model worker instantiations, tail prompt lengths, upstream retrieval service timeouts, and KV prefix cache hit rates. Tail latency (p99) cannot be diagnosed or resolved by inspecting average metrics. It requires decomposing requests by context length buckets and tenant traffic priority tiers.

### 22.8. Case Scenario E: Why Does Checkpoint Write Time Stretch into Minutes?

First, measure the **actual bytes written to disk**, the total shard count, and the number of concurrent rank writers. Next, inspect parallel filesystem metadata contention, storage fabric oversubscription, storage target disk saturation, conflicting concurrent I/O jobs, and compression overhead. While verifying whether GPUDirect Storage (GDS) is actively engaged is essential, GDS alone will not resolve underlying network congestion or filesystem metadata bottlenecks.

### 22.9. Six Cardinal Principles for Interpreting Empirical Benchmarks

1. **Archive the baseline measurement:** Persist library versions, environment variables, and git commit hashes.
2. **Standardize the workload:** Replay identical request traces and lock random seeds wherever applicable.
3. **Isolate variables:** Modify only a single variable or hypothesis at a time to establish definitive causal attribution.
4. **Control cluster state:** Rigorously account for warmup iterations, cache states (KV / page cache), and background fabric traffic.
5. **Analyze tail and variance:** Evaluate median, p95, and p99 distributions alongside repeated trials and confidence intervals.
6. **Validate systemically:** Ensure claimed throughput gains do not introduce model convergence regressions, numerical divergence, or hidden operational costs.

---

## 23. FinOps and Sustainable Capacity: Why Is the Cloud Bill Sky-High Despite High MFU?

The operational success of an AI infrastructure platform is not measured by seeing 100% utilization in `nvidia-smi`. An infrastructure cluster burning power 24 hours a day without generating proportional business value is little more than an extraordinarily expensive industrial space heater.

### 23.1. What Constitutes Total Cost of Ownership (TCO)?

**TCO (Total Cost of Ownership)** extends far beyond the raw GPU compute invoice or acquisition price. It encapsulates hardware amortization or cloud reservation leases, NICs and leaf/spine switches, optical transceivers and cabling, rack infrastructure, power delivery, facility cooling, parallel storage tiers, software licensing, maintenance and spare parts, transit/backbone egress bandwidth, and specialized operations headcount.

A simplified annualized cost model:

$$
C_{\mathrm{annual}}=C_{\mathrm{capex/amort}}+C_{\mathrm{power}}+C_{\mathrm{network}}+C_{\mathrm{storage}}+C_{\mathrm{software}}+C_{\mathrm{ops}}
$$

For annual energy consumption:

$$
E_{\mathrm{annual}}\text{ (kWh)}\approx P_{\mathrm{IT}}\text{ (kW)}\times PUE\times8760
$$

This formulation provides a first-order approximation assuming steady-state power draw. In production environments, dynamic power capping, idle power draws, seasonal PUE fluctuations, and duty cycle variations must be factored in.

### 23.2. Unit Economics: Which Metrics Should Replace Raw GPU-Hours?

Consider two deployment options:

- **Model A:** Highly capable, low latency, but computationally intensive and expensive to serve.
- **Model B:** Compact and cost-effective; resolves the vast majority of routine queries with acceptable quality.

Comparing flat `$/GPU-hour` rates is fundamentally deceptive. Business-aligned metrics such as **$/1M successful output tokens**, **$/1,000 SLO-compliant requests**, or **$/target-loss training completion** provide genuine financial visibility. Squeezing 5% extra throughput out of a cluster by burning 20% more power while degrading request success rates yields a net-negative return.

### 23.3. Model Routing as an Economic Architectural Lever

Rather than routing every incoming request to a monolithic frontier model, tiered routing architectures (dispatching simple queries to compact models and escalating complex prompts to frontier models) dramatically slash aggregate serving costs. However, router classification introduces its own latency and classification error overhead; misrouted queries degrade end-user quality. Consequently, **routing policies must be governed as an integral component of the product quality gate** rather than a purely financial optimization.

### 23.4. On-Premises vs. Cloud: Dispelling Dogma

On-premises deployments entail steep upfront capital commitments, long lead times for facility procurement, and deep operational responsibilities, counterbalanced by sovereignty, deterministic latency, and predictable unit economics at sustained scale. Cloud environments offer rapid provisioning and elastic bursting, but can incur punishing egress fees, rigid multi-year reservation lock-ins, and high baseline rental rates. The optimal decision framework demands:

1. **Categorize the workload profile:** Classify demands into baseline continuous, periodic batch, and highly volatile bursting.
2. **Model realistic 12-to-36-month utilization curves:** Avoid relying on best-case linear growth projections.
3. **Normalize for parity:** Ensure both on-prem and cloud scenarios are modeled against identical uptime SLOs, networking tiers, and response latency targets.
4. **Include full-spectrum overheads:** Quantify fabric, high-performance storage, operational headcount, data egress, and N+1/N+2 hardware redundancy buffers.
5. **Perform sensitivity analysis on duty cycles:** Rigorously stress-test financial viability across varying utilization percentages.

**Architectural Takeaway:** A hardware acquisition that appears economically compelling at an assumed 80% duty cycle turns into a catastrophic financial liability if actual sustained utilization hovers at 20%. Rather than collapsing uncertainty into a single optimistic estimate, architects must present transparent best-case, baseline, and worst-case scenario envelopes.

### 23.5. Carbon-Aware and Power-Aware Scheduling

For workloads with flexible SLAs, batch training runs can be scheduled during periods of low spot electricity tariffs or favorable ambient temperatures for free-air cooling. However, such scheduling must be weighed against training completion deadlines, data residency regulations, and wide-area data synchronization overheads. Crucially, **lower instantaneous wattage** is not synonymous with **lower kWh per unit of work**: an optimized run operating at elevated instantaneous power may complete so much faster that it expends substantially less total energy to reach convergence.

---

---

**PART VII — The Holistic Platform: From Security to Data Products**

*We now step back from the physical GPU interconnects to view the macro architecture: production workloads, security boundaries, user experience, and economic sustainability.*

## 24. The AI Architect's Perspective: How Technical Components Converge in a Single System

Let us conduct a thought experiment. A company asks you to "design an AI platform with 1,024 H100 GPUs." You have the hardware specifications, network ports, and GPU price tags in hand. Do you immediately begin sketching a rack layout?

I would not. First, I would ask five fundamental questions: **What workload will this platform serve? By what metric will success be measured? Who owns the data and models? How much downtime can we tolerate? What is the upper bound on total cost for all of this?** Because an architecture drafted before answering these questions might be a visually appealing cabling diagram, but it is by no means guaranteed to be the right AI platform.

### 24.1. Requirements First: The Same GPUs, Three Distinct Products

| Requirement | Large Model Pre-training | Fine-Tuning / Research | Enterprise Online Serving |
|---|---|---|---|
| Primary Deliverable | Checkpoint meeting target quality | Experiment results and customized model | Secure, timely response |
| Key Performance Metric | Time-to-quality, tokens/s | Experiment turnaround time, cost | TTFT/TPOT, goodput, p95/p99 latency |
| Resource Strategy | Homogeneous, topology-aware, gang scheduling | Flexible quotas, short-job queuing | Always-on reserved capacity, admission control |
| Critical Memory Footprint | Weights, gradients, optimizer states, activations | Model + optimizer/adapters | Weights + KV cache + transient scratch buffers |
| Critical Failure Scenario | Full restart of large-scale training run | Loss of experiment, lack of reproducibility | Disruption of live user requests |
| Primary Controls | Checkpointing + health/RAS | Experiment tracking + data/model versioning | Rate limiting, failover, multi-tenant isolation |

This table illustrates why a solitary "GPU utilization" metric is entirely insufficient for managing all three product archetypes.

### 24.2. Why Decouple the Control Plane from the Data Plane?

Consider an airport. On one side, there is the air traffic control tower and dispatch center deciding which aircraft departs from which gate; on the other, the physical ground operations moving passengers and baggage. The control tower does not carry passengers on its shoulders; it governs and coordinates the system.

The **control plane** determines to whom, when, and under what policies and configurations resources are allocated. Schedulers, job queues, model registries, permissions, deployment policies, and progressive rollout controls belong to this domain.

The **data plane** is the runtime execution path where actual training or inference computations take place on GPUs, KV cache blocks are transferred, and user responses are generated.

Decoupling these two domains does not imply that "every single request must round-trip through the control system"; rather, the control plane enforces governing decisions, while the data plane executes the latency-sensitive computational path.



<svg viewBox="0 0 720 318" role="img" aria-label="Three planes. The control plane on top holds IAM and policy, the scheduler and queue, the model registry, rollout control, and FinOps; it decides placement, quotas, routing policy and rollout without sitting in every request. The training data plane moves dataset shards through a GPU cluster running FSDP and NCCL into a checkpoint store; it is throughput-bound and runs for hours to weeks. The serving data plane moves requests through a gateway and router, a prefill pool and a decode pool, and streams to the user; it is latency-bound, in milliseconds. Telemetry from both data planes flows back to the control plane." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="pl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Control plane decides; the two data planes do the work</text>
<rect x="16" y="40" width="688" height="74" rx="10" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">Control plane · decisions, quotas, orchestration</text>
<rect x="28" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="91.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">IAM &amp; policy</text>
<rect x="162" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="225.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Scheduler / queue</text>
<rect x="296" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="359.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Model registry</text>
<rect x="430" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="493.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Rollout control</text>
<rect x="564" y="72" width="126" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="627.0" y="91.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">FinOps &amp; cost</text>
<path d="M190 114 V140" marker-end="url(#pl-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M530 114 V140" marker-end="url(#pl-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="198" y="132" text-anchor="start" style="fill:var(--c-accent-2);font-size:10.5px">placement, quotas</text>
<text x="538" y="132" text-anchor="start" style="fill:var(--c-accent-2);font-size:10.5px">routing policy, rollout</text>
<rect x="16" y="142" width="338" height="132" rx="10" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-warn);font-size:12.5px;font-weight:700">Training data plane</text>
<rect x="28" y="174" width="98" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="77.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Dataset</text>
<text x="77.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">shards</text>
<rect x="140" y="174" width="100" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="190.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU cluster</text>
<text x="190.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">FSDP / NCCL</text>
<rect x="254" y="174" width="88" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="298.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Checkpoint</text>
<text x="298.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">store</text>
<line x1="126" y1="201" x2="138" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="240" y1="201" x2="252" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="28" y="254" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">throughput-bound · hours to weeks</text>
<rect x="366" y="142" width="338" height="132" rx="10" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="378" y="162" text-anchor="start" style="fill:var(--c-success);font-size:12.5px;font-weight:700">Serving data plane</text>
<rect x="378" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="413.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Gateway</text>
<text x="413.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">router</text>
<rect x="458" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="493.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Prefill</text>
<text x="493.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">pool</text>
<line x1="448" y1="201" x2="457" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="538" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="573.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Decode</text>
<text x="573.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">pool</text>
<line x1="528" y1="201" x2="537" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="618" y="174" width="70" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="653.0" y="198.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">User</text>
<text x="653.0" y="213.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">stream</text>
<line x1="608" y1="201" x2="617" y2="201" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="378" y="254" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">latency-bound · milliseconds</text>
<path d="M190 274 V292 H530 V276" marker-end="url(#pl-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="360" y="306" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">telemetry flows back up to the control plane</text>
</svg>



This approach serves as a **pedagogical reference architecture**, inspired in part by worker, router, and management components found in systems such as NVIDIA Dynamo; it is not intended as an exact turnkey installation manual for a specific proprietary product. [K20]

In smaller systems, prefill and decode stages may be colocated on the same worker; in large-scale training systems, Slurm, Kubernetes, or dedicated distinct clusters may be employed. The concrete implementation of the identity and policy layer likewise depends on the enterprise security model.

### 24.3. A Pedagogical Layout Calculation for 1,024 H100 GPUs

Suppose we have 128 eight-GPU servers. When training a massive dense model, suppose we configure **TP=8, PP=8, DP=16**:

$$
N_{\mathrm{GPU}}=8\times8\times16=1024
$$

- **TP=8:** We confine the matrix shards of each individual layer within a single 8-GPU server, maximizing the use of high-speed scale-up interconnects (NVLink/NVSwitch).
- **PP=8:** We divide the sequential layers of each model replica across eight pipeline stages, which can traverse multiple server boundaries.
- **DP=16:** We replicate the training workload across 16 model-parallel groups operating on distinct data batch slices, enforcing parameter consistency through the gradient synchronization strategy.

This is strictly a **topological illustration matching the raw GPU count**. In practice, without rigorously validating the balanced partition of layers across PP stages, activation memory footprints, Context Parallelism (CP) requirements, network oversubscription ratios, optimizer state overhead, checkpointing latency, and total HBM headroom, one cannot claim this layout is optimal or even directly executable.

For instance, the raw BF16 weights of a 405B dense model alone occupy roughly 810 GB. Assuming an ideal, perfectly uniform split across $\text{TP}\times\text{PP}=64$ model shards, this yields **approximately 12.7 GB/GPU** in static weights. However, optimizer states, gradients, and activation buffers have not yet entered the equation. One cannot conclude that "since the model weights fit, training will definitely run."

**The topological verdict:** Rather than scattering an identical count of GPUs across arbitrary nodes, we must pin TP groups within ultra-fast local interconnect domains, route PP stage handoffs across an appropriately dimensioned fabric, and place DP groups to balance cross-rack bisection bandwidth against fault-domain blast radii.

### 24.4. Security and Governance Are Inseparable from Performance

In enterprise AI, the ability of a user to receive an inference response does not merely depend on GPU readiness. We must also verify that the request originates from an authenticated, authorized entity, that it routes to a permitted model endpoint, and that it adheres strictly to data access boundaries.

A **model registry** tracks approved model versions and their deployment locations. **RBAC (Role-Based Access Control)** defines access perimeters based on user and service roles. **Secrets management** provides the secure storage, injection, and rotation of API keys and access tokens without hardcoding them into application artifacts.

An enterprise data plane design must formally define the following safeguards:

- **Model and data provenance (lineage):** Which data sources, licensing terms, and evaluation benchmarks vetted this model for production?
- **Authorization:** Which specific models, tenants, and downstream data products is the requesting user permitted to access?
- **Data minimization:** Are unnecessary personally identifiable information (PII) or confidential attributes being injected into prompts?
- **Isolation:** How rigorously are boundaries enforced across KV caches, request logs, debug dumps, GPU memory spaces, persistent storage, and model artifacts?
- **Observability vs. privacy:** Is audit-level operational telemetry captured without logging raw, sensitive user payloads into centralized log sinks?
- **Supply chain integrity:** Are container images, model weights, custom tokenizers, and runtime dependencies cryptographically verifiable, signed, and strictly versioned?

Security cannot be reduced to a single downstream DLP filter. Prompt injection, sensitive data leakage, harmful model generation, training data provenance/copyright, and enterprise access control represent distinct threat planes. NIST's Generative AI Profile framework recommends systematically addressing these domains across the entire AI system lifecycle, explicitly incorporating data provenance and privacy risks. [K18]

#### Disentangling the AI Gateway, AI Safety, and Data Access Layers

An enterprise inference request can be visualized much like a passenger moving through an airport: first, identity and credentials are verified; next, compliance with security screening policies is confirmed; then, the passenger is routed to the designated gate (model). When strictly necessary, enterprise data systems are consulted; **not every inference request requires a retrieval query**.

- **API Gateway / Ingress Layer:** Enforces authentication (AuthN), authorization (AuthZ), quota tracking, rate limiting, and intelligent service routing.
- **AI Safety Layer:** Executes input/output guardrail policies, PII detection/redaction, content safety checks, and prompt-injection mitigations. No single heuristic or classifier provides an absolute security guarantee.
- **Inference Engine:** Executes the deep neural network; oversees batching schedulers, KV cache management, prefill/decode disaggregation, and tensor-parallel execution.
- **Data Product Gateway / Data Service:** Furnishes authorized, observable, and fully auditable access to enterprise data stores strictly on demand. This layer does not need to be coupled as a subordinate module of the AI Gateway.

This separation is also fundamentally a matter of the **latency budget**. For instance, when establishing an end-to-end TTFT SLO, ingress auth, safety screening, optional retrieval queries, queue queuing delays, and GPU prefill latency must be measured cumulatively. Furthermore, failure modes—whether requests fail-closed (blocked) or fail-open under safety service degradation—must be explicitly decided ahead of time based on risk classification; silently relaxing access controls during outages is never a secure default.

**Note on the Data Product Gateway / Data Service:** In enterprise architectures, mediating authenticated, audited access to contextual data required by a model or agent is an architectural responsibility distinct from running tensor inference. This data service is invoked strictly as needed; not every inference call hits a database or vector index. Disentangling it clarifies both access governance and latency accounting.

### 24.5. Economics: Beyond GPU Purchase Price to Total Cost of Ownership (TCO)

The total cost framework formalized in Section 23.1 serves as the financial bedrock of system design. For an AI Architect, the primary challenge is translating aggregate TCO into operational **unit economics**:

- **Cost per million tokens:** Must be evaluated not against raw generated tokens, but against *effective goodput tokens* delivered strictly within target SLO latency bounds.
- **Make-or-buy break-even thresholds:** Workloads exhibiting sporadic traffic or sustained fleet utilization below ~60% favor elastic cloud reservation; sustained, predictable workloads exceeding ~75% utilization frequently justify dedicated on-prem infrastructure once facility CapEx and operational staffing are accounted for.
- **SLA violation exposure:** The operational and financial penalties incurred when tail latency spikes or cluster outages breach customer SLAs can rapidly dwarf marginal accelerator hardware savings.

Consequently, financial planning shifts from "Which GPU server is cheaper?" to "Which architecture delivers 1 million SLO-compliant useful tokens at the lowest sustainable unit cost?"


### 24.6. Evaluating Architectural Choices: Which Benchmarks Answer Which Questions?

An architecture diagram visualizes component relationships, but it does not in itself constitute proof of performance or reliability. For this reason, architectural evaluations require understanding exactly which questions each tier of benchmarking answers. The table below outlines an exemplary assessment progression; it is not a prescriptive, universal deployment procedure.

| Evaluation Phase | Focal Question | Representative Evidence / Metric |
|---|---|---|
| **1. Workload Definition** | What model, data, and SLA/SLOs govern the workload? | ISL/OSL distributions, concurrency, model/dataset versions, target quality thresholds |
| **2. GPU Microbenchmarking** | What is the real behavior of GEMMs, attention kernels, and HBM? | Kernel profiles evaluated against hardware roofline limits |
| **3. Fabric Validation** | Are NICs, transceivers, and collective communication paths performing to spec? | `nccl-tests` logs, RDMA bandwidth/latency, topology maps |
| **4. I/O and Checkpointing** | Are storage burst throughput and restore latencies adequate? | Full-scale checkpoint save/load and simulated restart drills |
| **5. Small-Scale Pilot** | Is the model converging properly and stably? | Loss curves, numeric stability checks, time-to-quality projections |
| **6. Progressive Scaling** | What is the scaling efficiency from 8 → 64 → 256 → 1,024 GPUs? | Strong/weak scaling curves, straggler detection, MFU, effective unit cost |
| **7. Load and Chaos Testing** | How does the cluster behave under network partition, traffic spikes, or node drops? | SLO compliance under load, error budgets, failover times, MTTR |
| **8. Operational Handoff** | When incidents occur, who responds and with what runbooks? | Production telemetry dashboards, playbooks, automated remediation procedures |

Here, **strong scaling** measures how much runtime decreases as more GPUs are applied to a fixed total workload size. **Weak scaling** measures how well per-GPU throughput or step execution time is maintained as workload size scales proportionally with GPU count. Both tests are essential, as the practical impact of "doubling the GPU count" yields drastically different outcomes depending on the operational scaling regime.

### 24.7. Fundamental Questions Behind Architectural Decisions

When comparing alternative architectural blueprints, the following questions bring the technical assumptions underpinning each design decision into sharp focus:

| Architectural Dimension | Guiding Question |
|---|---|
| Model / Precision | Under what target quality criteria and numerical verification was precision selected? |
| GPU / HBM Capacity | Do weights + KV/activations + scratch buffers + safety headroom fit comfortably? |
| Parallelism Strategy | Why were TP/PP/DP/CP/EP mapped specifically to this interconnect topology? |
| Network Fabric | What is the observed delta between empirical NCCL/RDMA bandwidth and theoretical line rate? |
| Storage Subsystem | Have checkpoint blocking stall time, restore latency, and data integrity been empirically validated? |
| Orchestration | How are gang scheduling, quotas, preemption, and resource fragmentation governed? |
| Reliability / RAS | How are degraded nodes detected, quarantined, drained, and recertified? |
| Serving Engine | What are the strict p95/p99 targets for TTFT, TPOT, KV cache hit rate, concurrency, and goodput? |
| Data Governance | Are dataset/model lineage and sensitive data access patterns fully auditable? |
| Security & Compliance | Are zero-trust identity, data perimeter policies, audit trails, and multi-tenant cache isolation enforced? |
| Physical Facilities | Are rack-level power delivery, cooling thermal dissipation (air/liquid), and redundant feeds verified? |
| Unit Economics | What is the total cost per useful output unit, and what are the underlying workload growth assumptions? |

**The architectural takeaway of this treatise:** The best AI platform is neither the one boasting the highest theoretical peak FLOPS nor the one achieving isolated peak MFU in synthetic runs. **It is the platform that reliably delivers target model quality, latency SLOs, security, and operational resilience at an acceptable total cost of ownership.** True scaling engineering begins only when hardware, algorithms, data topologies, and operations are designed as tightly coupled facets of a singular, coherent system.

---

---

### Two Architectural Views of the Same Service: The "Hardware Map" and the "Responsibility Map"

An AI platform cannot be understood solely by examining how GPUs are cabled together. The hardware map reveals **how physical bits move across compute, memory, and networking domains**. The responsibility map, on the other hand, delineates **where decisions are made and governed**: Where is identity authenticated? Who owns model routing and policy dispatch? Under what credentials are external databases queried? Who mediates and inspects the model's tool calls and execution sandbox?



<svg viewBox="0 0 720 260" role="img" aria-label="Responsibility map of an enterprise AI request. A user or app sends an authorized request to the AI gateway, which checks identity, quota and rate limits; a safety and DLP layer filters prompts and data; the model service runs the LLM or agent on GPUs. Only if data is needed does the model service call a data access layer that performs authorized, audited RAG and tool calls. The output passes an output check for tool calls, leakage and policy, and the reply is streamed back to the user. These are logical responsibilities, not product boundaries." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="en-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="en-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="en-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Responsibility map of one enterprise AI request</text>
<rect x="16" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">User / app</text>
<text x="96.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">authorized client</text>
<rect x="192" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="272.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">AI Gateway</text>
<text x="272.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">identity, quota, rate limit</text>
<line x1="176" y1="72" x2="191" y2="72" marker-end="url(#en-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="448.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Safety &amp; DLP</text>
<text x="448.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">prompt / data filters</text>
<line x1="352" y1="72" x2="367" y2="72" marker-end="url(#en-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="44" width="160" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="624.0" y="70.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Model service</text>
<text x="624.0" y="84.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">LLM / agent on GPUs</text>
<line x1="528" y1="72" x2="543" y2="72" marker-end="url(#en-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="380" y="150" width="220" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="490.0" y="176.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Data access layer</text>
<text x="490.0" y="190.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">authorized RAG + tools, audited</text>
<path d="M654 100 V178 H602" marker-end="url(#en-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="646" y="143" text-anchor="end" style="fill:var(--c-accent-2);font-size:10.5px">only if data is needed</text>
<rect x="150" y="150" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="245.0" y="176.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Output check</text>
<text x="245.0" y="190.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tool calls, leakage, policy</text>
<path d="M564 100 V124 H245 V148" marker-end="url(#en-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M150 178 H96 V102" marker-end="url(#en-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<text x="100" y="196" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">streamed reply</text>
<text x="16" y="232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Logical responsibilities, not product boundaries: the data layer is called only when needed,</text>
<text x="16" y="248" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">and agent tool calls may need checks after the model too.</text>
</svg>



This is a **logical** decoupling; it neither prescribes rigid vendor boundaries nor dictates an invariant, linear request path. For example, in agentic workflows, security validation of tool executions may occur downstream of initial model generation. The data access layer is not invoked unconditionally on every cycle; it intervenes strictly when external contextual retrieval is required, guaranteeing authorized, audited access across enterprise data perimeters. Conflating these responsibilities complicates design and weakens governance, even when application servers and GPU clusters reside within the exact same data center facility.

### Why the Enterprise Security Layer Cannot Be Divorced from GPU Performance

In an enterprise RAG pipeline, prior to generating the very first token, the system may perform document retrieval, neural reranking, access privilege checks, and content safety filtering. Even if the GPU delivers flawless throughput, the end user may still experience unacceptable latency. This occurs because total latency is never confined to GPU execution alone:

$$
T_{request}=T_{auth}+T_{retrieval}+T_{queue}+T_{prefill}+T_{decode}+T_{postprocess}.
$$

While certain stages may partially overlap or pipeline in advanced runtime designs, this equation provides a vital conceptual decomposition. One fundamental reality emerges: **doubling GPU execution speed does not automatically halve end-to-end request turnaround time.** System designers must rigorously quantify the latency and cost overheads introduced by upstream security guardrails and downstream data retrieval stages.

### The Dual Realities of a 70B Model: Evaluating Workload Delivery, Not Just Static Memory Footprint

While the ~140 GB BF16 weights of a 70B model can fit across two 80 GB GPUs, this provisioning resolves only the *static weight containment* requirement. In production serving, feasibility is dictated entirely by the dynamic operating envelope under live user concurrency:

- **Static Boundary:** Weight placement across HBM (e.g., ~35 GB via INT4 quantization, or 70 GB/GPU under BF16 TP=2).
- **Dynamic Boundary:** Aggregate KV cache consumption across concurrent sessions, transient activation footprints, and framework runtime scratchpads.

An architect must move beyond the trivial question "Do the weights fit in memory?" to the decisive operational metric: "Under target concurrency and sequence length distributions, can the system sustain required token delivery velocities?" Where a model fits is a static capacity check; how reliably and fast it serves is an end-to-end systems engineering problem.


### Final Synthesis: The Benchmarking Hierarchy Is Not a Rigid Recipe, but a Chain of Causality

When an end user reports that "responses are slow," the right question is rarely "are our GPUs underpowered?" First, we must dissect *the specific nature of the degradation*: Is Time-to-First-Token (TTFT) delayed, is inter-token streaming (TPOT) stuttering, does latency spike exclusively on long multimodal inputs, or is the request stalled waiting in an upstream identity gateway or vector retrieval service?

Similarly, when a data center operator observes that "GPUs are sitting idle," there is never a single root cause. If a kernel is memory-bound within local HBM, optimizing NVLink fabrics will not move the needle. If nodes are blocked in NCCL collective synchronization barriers, upgrading to GPUs with higher peak Tensor Core FLOPS offers no relief. If training is throttled by storage I/O bottlenecks during checkpoint dumps, quantizing to lower numeric precision will not solve the stall.

Therefore, the physical and logical layers explored throughout this treatise are multifaceted expressions of a single core reality: **Large-scale AI systems are governed far less by peak theoretical compute capacity than by how dependencies and data movement are orchestrated across the hierarchy.**

## 25. From Technical Capacity to Workload: Which Questions Define an AI Architecture?

Up to this point, we have traversed the full technical spectrum—from intra-GPU computation to inter-cluster fabric communication. Now let us examine these identical technical building blocks from a different angle: Why might two systems behave fundamentally differently even when provisioned with the exact same number of GPUs? Because the delivered service, user traffic profile, data access patterns, and latency expectations fundamentally reshape architectural behavior. The examples below are not intended to dictate a rigid deployment blueprint, but rather to illuminate the **cause-and-effect dynamics underlying technical design choices**.

When an organization states, "Let's procure 300 GPUs," an architect's primary response should never be a GPU SKU recommendation. Instead, you must first establish the **service-level contract**. An architect designing a hotel determines the required room capacity, peak-season occupancy surges, emergency fire exits, and operating budget long before drawing physical floor plans. In an AI infrastructure, the question "how many GPUs?" is fundamentally incomplete without a concrete service specification.

### 25.1. Same GPU, Different Requirements: Seven Representative Questions

| Question | Representative Answer | Architectural Impact |
|---|---|---|
| What will the product do? | Chat, RAG, and document analysis with a 70B model | Requires serving, retrieval, and safety layers |
| What is the traffic profile? | Average 8 req/s, peak 25 req/s; bursty traffic | Requires queueing, autoscaling, and capacity headroom |
| Input/output length? | p50: 1k/200 tokens, p95: 8k/1k tokens | Prefill, decode, and KV memory are impacted differently |
| Success metric? | p95 TTFT ≤ 2 s, p95 TPOT ≤ 60 ms, failed requests ≤ 0.5% | Forms benchmark acceptance criteria |
| Where does data reside? | On-prem, specific classified data stores | Identity, network segmentation, and data access design |
| Fault tolerance? | Limited capacity degradation upon single GPU/node failure | N+1, multi-replica, routing, and recovery |
| Budget and timeline? | Specific kWh, rack, licensing, and operational budget | TCO and architectural trade-off evaluation |

All of these figures represent **illustrative requirements**. An architect's success lies not in memorizing a sample table, but in extracting these concrete parameters from the actual workload.

Partitioning operational demands into four primary workload archetypes streamlines architectural decision-making:

- **Batch training:** Long-running jobs lasting hours or days; priorities are time-to-quality, checkpointing efficiency, interconnect topology, and GPU-hour cost.
- **Interactive inference:** Users waiting synchronously; priorities are p95/p99 latency, goodput, queueing behavior, and overall availability.
- **Offline inference/embedding:** Nightly batch processing of millions of documents; large batch sizes and cost-efficient capacity often make more sense.
- **Research/fine-tuning:** Highly variable resource footprints; short queue times, experiment reproducibility, and shared cluster pools are critical.

### 25.2. SLI, SLO, and SLA: Why Do the Three Differ?

A **Service Level Indicator (SLI)** is a quantifiable metric directly observed: p95 TTFT, error rate, or the percentage of jobs completing within a target timeframe. A **Service Level Objective (SLO)** represents our internal target for that metric: for example, "99% of requests over any rolling 30-day window must complete in under five seconds." A **Service Level Agreement (SLA)** is a formal contractual commitment made to external customers, typically including legal remedies or financial credits upon breach. Violating an internal SLO does not necessarily trigger a contractual SLA breach.

A robust SLO never relies solely on average latency. Even if the arithmetic mean is a respectable 1 second, an enterprise application has failed if five out of every hundred users endure a 15-second hang. For this reason, architectures must be governed by **p95** (the boundary below which 95% of requests complete) and **p99** (the performance tail representing the slowest 1%).

### 25.3. Why Use Architecture Decision Records (ADRs)?

Architectural documentation extends far beyond boxes and arrows on a slide deck. Six months down the line, it must provide a defensible answer to: "Why did we engineer the system this way?"

```text
ADR-007 — Aggregated vs. disaggregated serving for production inference
Status: Proposed / To be validated empirically
Context: Long prompt p95 12k tokens; concurrency is increasing.
Alternative A: Unified prefill + decode within a single worker
Alternative B: Disaggregated worker pools for prefill and decode
Decision criteria: p95 TTFT/TPOT under identical quality and traffic,
                   successful req/s/GPU, KV transfer overhead, operational risk
Experiment: A/B load testing; cold/warm prefix; 4 traffic profiles
Rollback: Revert to Alternative A if gains are not substantiated
Owner: AI Platform Team
```

**Conceptual distinction:** An unmeasured advantage remains nothing more than a **hypothesis**. Rather than asserting "RDMA will definitely accelerate our workload," framing the decision as "if KV cache transfers account for an empirically measured 18% of total latency, evaluating an RDMA-based alternative is warranted" reflects disciplined, evidence-based engineering.

### 25.4. Limits of a Reference Architecture

Every production system operates across three distinct architectural planes: **logical architecture** (what capabilities exist?), **physical architecture** (which servers, NICs, and racks host the services?), and **operational architecture** (who monitors, who intervenes, and how are rollbacks orchestrated?). Most AI architectural diagrams illustrate only the logical tier; in production, outages almost invariably manifest in the seams between the physical and operational tiers.

---


## 26. RAG, Agents, and Enterprise Data Architecture: Bottlenecks Outside the GPU

When an enterprise employee queries, "Why did our product sales decline last quarter?", the model's parametric memory will almost certainly lack the specific corporate context. The system must query proprietary corporate data repositories. This is the domain of **Retrieval-Augmented Generation (RAG)**.

Intuitive analogy: Much like a physician consulting an authoritative medical guideline rather than relying entirely on rote memorization, a RAG system first gathers corroborated evidence and then prompts the model to synthesize an answer grounded strictly in that evidence. However, retrieving "a document deemed relevant by similarity" does not automatically guarantee that it is the "correct and authoritative document."

### 26.1. How Many Layers Does RAG Actually Have?



<svg viewBox="0 0 720 260" role="img" aria-label="RAG has two pipelines. Offline ingestion takes approved sources with ACL labels, chunks them into semantic pieces, embeds them into vectors, and stores vectors with ACL metadata in a vector index. Online, each question arrives with the user&#x27;s identity, tenant and role; filtered retrieval queries the index with the user&#x27;s permissions using an ACL filter and hybrid search; reranking picks the evidence; and the LLM answers with citations grounded in that evidence. Filtering only the model&#x27;s answer is too late, because unauthorized text would already be in the prompt." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rg2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rg2-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="rg2-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">RAG has two pipelines — and authorization belongs before retrieval</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:700">offline: ingestion</text>
<rect x="16" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Approved sources</text>
<text x="96.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">docs + ACL labels</text>
<rect x="192" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="272.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Chunking</text>
<text x="272.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">semantic pieces</text>
<line x1="176" y1="85" x2="191" y2="85" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="448.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Embedding</text>
<text x="448.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">vectors</text>
<line x1="352" y1="85" x2="367" y2="85" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="60" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="624.0" y="82.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Vector index</text>
<text x="624.0" y="97.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">vectors + ACL metadata</text>
<line x1="528" y1="85" x2="543" y2="85" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:700">online: per question</text>
<rect x="16" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Question + identity</text>
<text x="96.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tenant, role</text>
<rect x="192" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="272.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Filtered retrieval</text>
<text x="272.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">ACL filter + hybrid search</text>
<line x1="176" y1="175" x2="191" y2="175" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="448.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Reranking</text>
<text x="448.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">pick the evidence</text>
<line x1="352" y1="175" x2="367" y2="175" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="150" width="160" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="624.0" y="172.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LLM + citations</text>
<text x="624.0" y="187.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">answer grounded in evidence</text>
<line x1="528" y1="175" x2="543" y2="175" marker-end="url(#rg2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M624 110 V124 H272 V148" marker-end="url(#rg2-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<text x="448" y="139" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">index queried with the user&#x27;s permissions</text>
<rect x="16" y="214" width="688" height="34" rx="6" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="28" y="236" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">Filtering only the model&#x27;s answer is too late: unauthorized text has already entered the prompt.</text>
</svg>



**Embedding** transforms a textual chunk into a high-dimensional vector whose geometric proximity reflects semantic similarity. A **vector index** accelerates nearest-neighbor search across millions of these embeddings. **Hybrid retrieval** mitigates purely dense retrieval blind spots by pairing vector similarity with lexical algorithms such as BM25. A **reranker** takes the initial pool of candidates and scores them using a more compute-intensive, cross-attentive model to produce high-precision ranking.

### 26.2. The Most Dangerous RAG Antipattern: Post-Retrieval Authorization

If a user lacks read authorization for a document, that document must never enter the retrieval result set in the first place. Attempting to filter out unauthorized content from the model's generated response post-hoc fails to eliminate the vulnerability of injecting confidential text into the LLM context prompt. Access controls—including tenant isolation, departmental boundaries, document classifications, and row/column-level security—must be enforced strictly at the **retrieval or data-service layer**.

### 26.3. Evaluating RAG Quality

Evaluating a RAG system purely on the perceived eloquence of the final LLM response is fundamentally flawed. First, measure **retrieval recall@k** to determine whether the ground-truth evidence is present within the top-$k$ returned passages. Next, quantify **context precision**, attribution accuracy, faithfulness, answer semantic quality, pipeline latency, and token consumption. Because some of these metrics can be subjective or LLM-judge-driven depending on the evaluation methodology, high-stakes enterprise workflows require continuous human validation.

### 26.4. Agent Architecture: Models Don't Just Reason, They Trigger Actions

Autonomous agents plan and execute tool invocations such as external search, code execution, database mutations, and ticket generation. At this juncture, **tool authorization**, **approval gates**, **idempotency**, **audit trails**, and the principle of **least privilege** become just as critical as raw latency optimizations. A model proposing an API call does not inherently render that call authorized.

### 26.5. Defining Boundaries: AI Gateway vs. Data Product Gateway

- **AI Gateway:** Controls who can invoke which AI models; enforces rate limits, routing, tenant authentication, and request policies.
- **AI Safety:** Enforces input/output safety classifiers, PII masking, malicious payload detection, and escalation triggers for human review.
- **Data Product Gateway:** Mediates access to enterprise data sources based on user/service identity and data product contracts.
- **Serving Engine:** Manages GPU tensor computation, KV caching, dynamic continuous batching, and inference parallelization.

These represent logical architectural responsibilities; they need not be deployed as four separate commercial products. However, if these policy boundaries remain ambiguous, teams cannot answer the fundamental audit question: "The model generated a response, but who authorized access to that corporate data?"

---


## 27. Security and Isolation: What New Risks Emerge from Shared Resources?

While co-locating two different workloads on the same physical GPU may be technically feasible, **resource sharing** is fundamentally distinct from **security isolation**. In enterprise LLM and vision-language systems, sensitive data continuously traverses training corpora, user prompts, tool parameters, cache structures, debug logs, and generated outputs.

### 27.1. Threat Modeling: What Are We Protecting, and from Whom?

| Asset | Potential Risk | Architectural Control |
|---|---|---|
| GPU compute capacity | One tenant starving others of throughput | Quotas, fair-share schedulers, dedicated pools, or hardware partitioning |
| KV/prefix cache | Cross-tenant data leakage or state bleeding | Tenant/salt isolation boundaries, cache invalidation policies, automated regression testing |
| Model weights and adapters | Unauthorized access, theft, or tampering | Cryptographically signed artifacts, RBAC, strict version pinning, and SHA-256 verification |
| Dataset / RAG index | Unauthorized document retrieval | Data-level authorization, encryption at rest/in transit, and audit logging |
| Agent tools | Unauthorized tool execution or lateral movement | Tool allowlisting, scoped credentials, human-in-the-loop approval gates |
| Containers and drivers | Supply chain vulnerabilities or host privilege escalation | Image vulnerability scanning, digest pinning, and least-privilege runtime security |
| Telemetry and debug | Sensitive prompts or credentials leaking into logs | PII redaction, strict retention limits, and role-based access to telemetry |

### 27.2. MIG, MPS, and Time-Slicing Are Not Synonymous with Security

- **Time-slicing:** Multiple processes take turns executing on the GPU; offers zero physical resource isolation, making strict latency and memory bandwidth guarantees impossible.
- **MPS (Multi-Process Service):** Concurrently schedules kernels from multiple CUDA processes; while it provides certain resource allocation ceilings, it should never be conflated with hardware-level security partitioning.
- **MIG (Multi-Instance GPU):** On supported architectures, physically partitions compute engines, memory controllers, and cache slices into isolated GPU instances. However, security across the host OS, device drivers, control plane, and network fabric must still be engineered explicitly. [K45]

Consequently, in regulated or multi-tenant environments, one cannot simply declare: "We enabled MIG, so all isolation risks are resolved." Each environment's isolation boundaries must be validated against the hardware vendor's security architecture and the workload's specific threat model.

### 27.3. Why Prompt Injection Defenses and Data Access Controls Are Not Substitutes

Prompt injection occurs when adversarial input manipulates an LLM into ignoring system guardrails or taking unintended actions. If the tools and databases accessible to the model are already constrained by **least privilege**, the potential blast radius of such manipulation is fundamentally bounded. Appending "ignore previous instructions" checks to system prompts does not constitute a robust technical security perimeter.

Example: An indexed RAG document might contain adversarial text designed to spoof system prompt instructions. That text must never escalate the end-user's actual operational privileges. Retrieved document contents must be treated strictly as untrusted data; the agent's authorization to invoke downstream tools must be verified by an external policy engine and tool gateway.

### 27.4. End-to-End Security Across the Data Lifecycle

Security controls must follow data throughout its entire lifecycle: permission validation and data classification during ingestion, tenant sharding across vector indexes, prompt minimization during context assembly, caller identity validation on API endpoints, safety classification on model outputs, and data retention/pruning across system logs. A **right-to-erasure request** does not end with deleting the raw file: all derived vector embeddings, KV caches, database snapshots, analytical replicas, and disaster backups must be expunged according to a well-defined lifecycle policy.

### 27.5. Sample Security Review Checklist

```text
[ ] Model / tokenizer / adapter hashes and origins are cryptographically verified
[ ] Training and serving container images are signed and scanned
[ ] Secrets are dynamically fetched from a central vault via short-lived credentials
[ ] Runtime host privileges and GPU device visibility are strictly constrained
[ ] Tenant isolation boundaries are verified across KV / prefix caches / RAG indexes
[ ] All tool invocations traverse an external authorization policy enforcement layer
[ ] Prompt injection and data exfiltration resilience tests have been executed
[ ] Raw sensitive content complies with logging and distributed tracing retention rules
[ ] Incident response, access revocation, and credential rotation runbooks are tested
```

---


## 28. Walking Through a 70B Model: Components of an Enterprise AI System

Let us synthesize these architectural dimensions through a concrete design scenario. An enterprise requires both conversational search and RAG workflows powered by a 70B dense language model. User prompts range from short interactive queries to large document uploads. Classified enterprise data must reside strictly on-premises. All operational figures in this section are **pedagogical assumptions**; production deployments demand rigorous empirical benchmarking.

### 28.1. Requirements

- Model: 70B dense Transformer; initial baseline in BF16, with evaluation pathways for INT8/FP8 quantized weights.
- Traffic: Nominal 5 req/s, peaking at 15 req/s during business hours; input distribution p50 1k tokens, p95 8k tokens; output distribution p50 250 tokens, p95 1k tokens.
- User experience: p95 TTFT ≤ 2 s, p95 TPOT ≤ 70 ms; dedicated latency budgets allocated for RAG retrieval overhead.
- Security: Model inference gated behind authenticated endpoints; enterprise document retrieval mediated strictly by a separate data service.
- High availability: Capacity degradation is acceptable upon the loss of a single serving node; complete service interruption is unacceptable.

### 28.2. Initial Memory Sizing

In BF16 precision, model weights occupy approximately 140 GB. While two 80 GB GPUs offer 160 GB of raw HBM, this leaves minimal headroom for runtime engine overhead and KV cache allocation. Therefore, the **initial baseline architecture** should explore a 4-way tensor-parallel (TP=4) configuration across four 80 GB GPUs. The physical interconnect between these four GPUs (NVLink vs. PCIe) must be validated; selecting TP=4 simply to meet aggregate capacity does not ensure efficiency if inter-GPU bandwidth is constrained.

Recall our earlier sizing rule of ~2.5 GiB logical KV cache per 8,192 tokens. Concurrently servicing 32 such requests demands approximately 80 GiB of logical KV memory across a single replica. This KV pool is sharded across the four GPUs by the serving runtime; modern block allocators and prefix caching alter the physical allocation footprint. The crucial architectural question is not merely "Do we have 80 GiB of aggregate HBM?", but rather: **How evenly is cache distributed across GPUs, and how many concurrent sequences can we reliably host at p95 context lengths without triggering KV eviction or thrashing?**

### 28.3. Baseline Reference Architecture



<svg viewBox="0 0 720 344" role="img" aria-label="Starting reference architecture for one 70B service. A client calls the AI gateway for identity and rate limiting, then a safety layer for input and output checks, then a model router that handles queueing and admission. The router feeds 70B serving replicas on different servers, so losing one node costs partial capacity rather than an outage; each replica runs on four H100 80 GB GPUs with tensor parallelism 4 over NVLink. BF16 weights of about 140 GB take roughly 35 GB per GPU, leaving the rest of the 320 GB for KV cache and runtime buffers. A data gateway provides authorized RAG access only when needed, and a control plane with FinOps handles telemetry, rollout and cost per token. Variants to test: BF16 aggregated TP 4, lower-precision weights, and disaggregated serving only with a fast KV fabric." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rf7-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rf7-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Starting reference architecture for one 70B service</text>
<rect x="16" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Client</text>
<text x="96.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">app / user</text>
<rect x="192" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="272.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">AI Gateway</text>
<text x="272.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">identity, rate limit</text>
<line x1="176" y1="70" x2="191" y2="70" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="368" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="448.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Safety layer</text>
<text x="448.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">input / output checks</text>
<line x1="352" y1="70" x2="367" y2="70" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="544" y="44" width="160" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="624.0" y="68.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Model router</text>
<text x="624.0" y="82.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">queue, admission</text>
<line x1="528" y1="70" x2="543" y2="70" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="624.0" y1="96" x2="624.0" y2="122" marker-end="url(#rf7-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="124" width="688" height="104" rx="10" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="28" y="144" text-anchor="start" style="fill:var(--c-accent-2);font-size:12.5px;font-weight:700">70B serving replicas on different servers: losing one node = partial capacity, not an outage</text>
<rect x="28" y="154" width="324" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">replica 1: 4 × H100 80 GB, TP = 4</text>
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
<text x="380" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">replica 2: 4 × H100 80 GB, TP = 4</text>
<rect x="380" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="405.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 0</text>
<rect x="436" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="461.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 1</text>
<rect x="492" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="517.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 2</text>
<rect x="548" y="182" width="50" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="573.0" y="197.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">GPU 3</text>
<text x="608" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">NVLink</text>
<text x="28" y="248" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">BF16 weights ≈ 140 GB → ~35 GB per GPU; the rest of 320 GB holds KV cache + runtime buffers.</text>
<rect x="16" y="264" width="336" height="46" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="184.0" y="284.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Data gateway</text>
<text x="184.0" y="299.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">authorized RAG access, called only when needed</text>
<rect x="368" y="264" width="336" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="536.0" y="284.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Control plane &amp; FinOps</text>
<text x="536.0" y="299.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">telemetry, rollout, cost per token</text>
<text x="16" y="332" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Test variants on the same traffic: A = BF16 aggregated TP 4 · B = lower-precision weights · C = disaggregated, only with a fast KV fabric.</text>
</svg>



This diagram illustrates a logical architecture. The Data Product Gateway is not an intrinsic sub-component of the AI Gateway. How retrieval outputs are assembled into the model prompt, where security policy boundaries reside, and how data access contracts are enforced must be explicitly defined in the detailed system design.

### 28.4. Capacity Benchmarking Experiment

Under identical synthetic traffic loads, evaluate three deployment configurations:

- **A:** BF16 weights, aggregated prefill/decode, TP=4.
- **B:** Lower-precision quantized weights (FP8/INT8), aggregated prefill/decode; validated against strict quality regression benchmarks.
- **C:** Disaggregated prefill and decode worker pools based on Configuration A, provisioned only if a dedicated high-bandwidth KV interconnect fabric is available.

For each variant, under varying concurrency, short vs. long prompts, and warm vs. cold prefix caches, measure **TTFT, TPOT, p99 latency, request throughput (req/s), aggregate output tokens/s, peak KV footprint per request, power consumption, and error rates**. Configuration C introduces architectural complexity and infrastructure cost; it should only be adopted if it delivers quantifiable, non-marginal improvements within our SLO envelope.

### 28.5. Failure Injection and Recovery Testing

First, simulate a worker node crash to verify that the ingress load balancer immediately diverts incoming traffic away from the degraded replica. Next, examine how active streaming connections terminate and verify that client applications receive graceful, well-structured error payloads. Finally, evaluate zero-downtime rolling upgrades to confirm that sufficient warm serving capacity remains online to maintain SLO compliance during model updates.

### 28.6. Documenting Architecture Decisions for This Case

```text
Product: Enterprise LLM + RAG
Workload: p50/p95 prompt-output token distributions and peak QPS attached
Selected model/runtime: Git commit SHA and runtime container digests recorded
GPU & topology: Interconnect benchmarks and capacity reserve sizing attached
Quality gate: Accuracy, domain benchmarks, long context recall, hallucination tests
Performance gate: p95 TTFT, p95 TPOT, goodput, cost per million tokens
Reliability: Failover test logs, recovery drills, and rollback mechanisms validated
Security: IAM, tenant isolation, data product contracts, logging retention policies
Operations: Telemetry dashboards, alerting rules, on-call runbooks, maintenance schedules
Sign-off: Date / Technical Owner / Review Trigger Conditions
```

The true output of an architectural design review is not a vendor slide deck, but this comprehensive body of verifiable engineering evidence.

---


## 29. Evolution of AI Platforms Over Time: Organization and Operations

Beyond physical silicon and network fabrics, an AI platform is shaped by organizational dynamics. As systems expand, the collaboration between GPU infrastructure, networking, data engineering, ML research, security, and product teams becomes paramount. The operational cadence and timelines outlined below do not prescribe a rigid enterprise deployment schedule; they serve as an **illustrative mental model** to structure organizational complexity.

### 29.1. Division of Responsibilities Across Teams

| Responsibility | Primary Owner | Collaborating Teams |
|---|---|---|
| GPU/driver/fabric health | Platform / SRE | DC / Networking |
| Dataset & ingestion quality | Data Engineering | ML Research |
| Model training & quality | ML Engineering / Research | GPU Platform |
| Serving p95/p99 SLOs | Inference Platform | Product / SRE |
| Identity, data access, auditing | Security / Data Governance | Platform / Product |
| TCO & capacity forecasting | Platform / FinOps | Product / Finance |

This table represents an **initial RACI template**, rather than a universal organizational structure. What matters is that for every failure mode, security gate, and architectural decision, a single accountable team is explicitly documented.

> **Timeline Note:** The phase milestones referenced below (e.g., 30/60/90 days) do not prescribe an inflexible installation schedule; they provide a pedagogical framework for explaining architectural maturity progression.

### 29.2. Initial Phase Example: Establishing the Baseline

The initial phase of deploying an AI platform focuses on comprehensive asset discovery and baseline benchmarking: mapping GPU/NUMA/NIC topologies, characterizing storage throughput, cataloging software dependency trees, measuring baseline model accuracy, and recording unconstrained p95 latencies. This exploratory phase emphasizes why system behavior cannot be interpreted without rigorous baseline metrics.

### 29.3. Intermediate Phase Example: Stability and Automation

As platform maturity increases, operational priorities shift toward golden machine images, unified scheduling policies, centralized model registries, automated CI/CD pipelines, end-to-end observability, structured post-mortems, and reliable checkpoint restore mechanisms. These are not disparate software tools; they represent the unified pillars of version consistency, reproducibility, and fault tolerance. For instance, rigorous failover drills reveal the gap between merely possessing redundant hardware and actually sustaining uninterrupted service delivery.

### 29.4. Maturation Phase Example: Balancing Performance and Cost

In later stages of maturity, engineering efforts center on holistic performance and cost optimization. Elevating MFU from 40% to 45%, for example, is not universally more valuable than shaving 300 ms off tail TTFT. The impact of advanced techniques—such as custom kernel fusion, mixed-precision quantization, aggressive prefix caching, and disaggregated prefill/decode architectures—must be evaluated holistically against actual traffic distributions, accuracy budgets, and operational expenditures.

### 29.5. Operational Observability Across Multiple Time Horizons

- **Weekly Cadence:** SLO adherence, incident post-mortem reviews, capacity fluctuations, and active high-priority risks.
- **Monthly Cadence:** Cost per goodput token, GPU failure patterns, active model version distribution, and technical debt accumulation.
- **Quarterly Cadence:** Network and storage bottleneck trends, hardware lifecycle planning, disaster recovery drills, and adversarial security red-teaming.

These review intervals do not represent an immutable operational standard. They illustrate that daily, weekly, and quarterly telemetry surface fundamentally different classes of architectural insights. Furthermore, converting every operational metric into a high-stakes performance target can incentivize suboptimal, metric-gaming engineering practices.

---

## 30. Making sense of numbers: Small Python experiments and assumptions

The following example uses dependency-free Python 3 code to produce **didactic lower bounds**. It does not query a live GPU cluster, nor does it replace real runtime benchmarking. However, it quickly helps test whether figures cited during design discussions make sense in terms of order of magnitude.

```python
from math import sqrt


def weights_gb(params_billion: float, bits: float) -> float:
    """Decimal GB; excludes quantization metadata / runtime overhead."""
    return params_billion * 1e9 * bits / 8 / 1e9


def kv_gib(layers: int, kv_heads: int, head_dim: int,
           tokens: int, bytes_per_element: int = 2) -> float:
    """Total logical KV; prior to TP/PP sharding."""
    n_bytes = 2 * layers * kv_heads * head_dim * tokens * bytes_per_element
    return n_bytes / (1024 ** 3)


def checkpoint_interval_min(snapshot_tb: float,
                            write_gbs: float, mtbf_hours: float) -> float:
    """Young approximate optimum; assuming blocking checkpoint."""
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
    """TP degree at which T_math = T_comm (Megatron MLP, first order)."""
    return ffn_dim * link_bytes_s / peak_flops


def dp_min_tokens_per_gpu(peak_flops: float = 990e12,
                          link_bytes_s: float = 400e9,
                          experts: int = 1, active: int = 1) -> float:
    """Per-GPU token batch that hides gradient sync; x E/k for MoE."""
    return experts / active * peak_flops / link_bytes_s


print("TP limit, NVLink 450 GB/s:", round(tp_max(28672), 2))
print("TP limit, NIC 50 GB/s:", round(tp_max(28672, link_bytes_s=50e9), 2))
print("DP tokens/GPU across nodes:", round(dp_min_tokens_per_gpu()))
print("MoE E=128, k=4 tokens/GPU:",
      round(dp_min_tokens_per_gpu(experts=128, active=4)))
print("Decode ceiling, 8xH100 TP=8 (tokens/s):", round(8 * 3350 / 140))
```

Expected output:

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

The checkpoint result here is obtained assuming $7.5\,\mathrm{TB}/250\,\mathrm{GB/s}=30\,\mathrm{s}$ and $M=3\,\mathrm{h}$. If the numbers appear consistent, the next step is to **transition to real workload profiling and benchmarking**.

### 30.1. Eight sample symptoms and underlying potential mechanisms

| Symptom | Associated Metric or Variable | Potentially Flawed Inference |
|---|---|---|
| Training MFU suddenly dropped | Configuration, profiler timeline, stragglers | Arbitrarily increasing TP |
| NCCL hang | RAS, rank state, GPU/NIC/topology | Increasing all timeouts |
| TTFT p99 spiked | Queue, long prompt, cold start | Simply adding GPUs |
| TPOT degraded | Decode HBM, batch size, and cache | Optimizing prefill alone |
| Slow checkpointing | Effective write BW, sharding, and metadata | Assuming GDS resolves everything |
| Insufficient HBM | Weights, KV, activations breakdown | Applying quantization without identifying root causes |
| MoE imbalanced | Expert load and AllToAll distribution | Benchmarking against dense models |
| Model output quality degraded | Model/precision/data change | Continuing rollout merely because MFU is high |

**Common takeaway across examples:** A single symptom rarely points to a single root cause. This table illustrates how different hardware and software layers can converge to impact the same end-user experience.

---




---

**PART VIII — Synthesizing what we have learned**

*We will consolidate principles across various topics through several thought experiments and a comprehensive glossary.*


## 31. Testing our understanding: Six brief thought experiments

Now let us cultivate an engineering reflex: whenever we encounter a number, let us immediately ask which assumptions produced it. The questions below are not intended as a test; rather, their purpose is to ground our understanding through several concise calculations.

### Question 1 — Fitting a 70B model into memory in BF16

If we have 70 billion parameters, each 2 bytes, what is the weight-only memory footprint in GB?

**Solution:**

$70\times10^9\times2=140\times10^9$ bytes = **140 GB**. This excludes KV cache, activations, and runtime scratch buffers. Although eight 80 GB H100s offer a theoretical total of 640 GB HBM, the physical partitioning scheme and memory fragmentation remain critical.



### Question 2 — Roofline threshold

Given 990 TFLOP/s and 3.35 TB/s, what is the H100 BF16 arithmetic intensity threshold?

**Solution:**

$990/3.35\approx\mathbf{295.5}$ FLOP/byte. This value serves as the approximate boundary for dense BF16 Tensor Core peak compute and the HBM3 pathway.



### Question 3 — How do you partition 16 GPUs?

In a setup that splits the model into four pipeline stages and uses a two-GPU tensor parallel group per stage, how many data parallel replicas remain? Assume CP=1.

**Solution:**

$DP=16/(TP\times PP)=16/(2\times4)=\mathbf{2}$. Each data-parallel replica occupies 8 GPUs.



### Question 4 — Ring AllReduce transfer volume

For 8 GPUs and 8 GB of data per GPU, what is the ideal ring AllReduce send volume in GB/GPU?

**Solution:**

$2\times(8-1)/8\times8=\mathbf{14}$ GB/GPU. This reflects the payload model and excludes message latency and protocol overhead.



### Question 5 — Why accelerating storage reduces failure overhead

Cluster MTBF = 3 hours, blocking checkpoint duration = 30 seconds. What is the approximate optimal checkpoint interval?

**Solution:**

$\sqrt{2\times30\times10800}=804.98$ seconds ≈ **13.4 minutes**. This formula represents the first-order Young–Daly approximation.



### Question 6 — Is buying the most powerful GPU enough?

A team upgrades to a GPU offering twice the peak FLOP/s, but memory bandwidth remains unchanged. If the decode workload is memory-bound, why might it fail to achieve a 2x speedup?

**Solution:**

Because if the product $\alpha W_{\mathrm{mem}}$ does not change, the memory roof remains identical. Additional compute units continue to stall waiting for data. Improving throughput requires data reuse, batching, compression, or higher effective memory bandwidth.



---


## Conclusion: Shared principles from GPUs to supercomputers

We started from the smallest execution unit of a GPU—the warp—and journeyed all the way to multi-thousand GPU clusters, checkpoint design, and MFU. Moreover, as demonstrated in the Lambda and MegaScale studies, software orchestration and timing on the exact same hardware can create a far greater difference than what is expected from a GPU generation upgrade alone. Along this path, we repeatedly encountered the same two recurring principles: **keeping data close** and **filling inevitable idle time with useful work whenever possible**.

Consider this: the warp scheduler inside an SM switches to another warp when the current one is stalled waiting for data. Tiling within the GPU avoids redundant trips to HBM. Within a DGX node, NVLink keeps bandwidth-heavy inter-layer communication close. At ultra-large cluster scales, PP and communication-overlap techniques similarly strive to eliminate idle bubbles. GDS and asynchronous checkpointing carry that very same principle over to the storage tier.

The 'weakest link' analogy is useful, yet fundamentally incomplete on its own. That is because certain operational phases can be overlapped: computation can proceed during network transfers, checkpoints can be written asynchronously in the background, and another warp can execute while one waits on memory. Excellent AI infrastructure engineering is precisely the art of architecting these **overlap and data locality** opportunities.

Whatever the warp scheduler resolves within a single GPU, pipeline scheduling attempts to solve at massive cluster scales: **filling idle stall cycles with productive computation**. Just as shared memory tiling minimizes round-trips to HBM, confining TP within an NVLink domain represents the distributed systems counterpart of the identical principle: **keeping data as close as physically possible**.

Therefore, the right question is never 'how many GPUs should we buy?' The right questions are: *Which workload? Which precision? Which matrix shapes? What batch size and concurrency? Which communication topology? What checkpoint and recovery targets? And against which metrics will we validate them?*

For an **AI Architect**, one final question remains at the end of the journey: *Does the system preserve the target user experience not only at peak benchmark moments, but on stressed, fault-ridden production days?* Because a great platform is not defined merely by record-breaking benchmarks; it demands traceable dataset lineage, secure serving pathways, robust schedulers, realistic fault-recovery plans, sustainable power budgets, and measurable unit costs.

**Closing:** Understanding the performance of a 1,024-GPU cluster requires far more than glancing at peak TFLOP/s figures. One must examine which data traverses which memory hierarchy levels, when GPUs communicate, and which resources are consumed across training and inference. That is the foundational thesis of this article: AI infrastructure is the interconnected sum of myriad physical and software mechanisms.

---




---

### What questions can the reader now answer independently?

Let us return to the 1,024 H100 and 38% MFU example from the introductory page. That figure no longer implies "the rest of the GPU is shut down." Instead, it raises an engineering question mark that we can now dissect across three distinct scales: which kernel is constrained within the GPU; what communication dependencies exist across multiple GPUs; and how do cluster-wide data, reliability, and serving targets impact the whole system?

To answer this, we do not need to memorize a single magic formula or vendor list. The questions: **Where does a byte originate, how many times is it reused, and where is it transported; which dependent result is a computation waiting for; and ultimately, what quality and latency targets does it unlock?** form the starting point for nearly every architectural discussion.

This work is not an *exhaustive troubleshooting encyclopedia*. New GPU generations, novel Transformer variants, VLMs, and serving methodologies will continue to evolve. Yet viewing data, compute, communication, and user experience through this unified mental model makes understanding emerging systems vastly simpler.



---



## The whole story in six lines

- On a 1,024-H100 supercomputer, a theoretical 1 ExaFLOP on paper stalls at 38% MFU unless physical transfer rates across compute (SM/Tensor Cores), memory (HBM3), networking (NVLink/InfiniBand), and storage (GDS/I-O) are strictly balanced.
- H100 SXM's 1,979 TFLOP/s BF16 peak compute and 3.35 TB/s HBM3 bandwidth dictate a critical arithmetic intensity (Roofline) of $\alpha_{crit} \approx 295$ FLOP/byte; any operation below this threshold condemns the GPU to stall on memory transfers.
- Intra-node 900 GB/s NVLink and NVSwitch crossbars keep 8-way Tensor Parallelism (TP) pinned to dedicated hardware pipelines, whereas scaling TP across nodes collapses against a 9x bandwidth chasm (50 GB/s InfiniBand), grinding training to a halt.
- In distributed LLM training, the ultimate goal of orchestrating 4D parallelism (TP, PP, DP/FSDP, CP, EP) is to hide latency by overlapping collective communication and data ingestion behind ongoing computation, driving GPU stall cycles toward zero.
- GPUDirect Storage and asynchronous checkpointing bypass host CPU bounce-buffer overhead to stream terabytes of model state directly between GPU HBM and NVMe storage, tearing down the I/O wall.
- In inference serving, prefill is compute-bound via dense matrix multiplications, whereas decode is memory-bound due to reading the entire KV cache from HBM for every generated token; disaggregating these workloads into dedicated GPU pools optimizes TTFT and TPOT independently.

## Glossary

- **ALU (Arithmetic Logic Unit)** — Circuit responsible for fundamental arithmetic and comparison operations
- **AllGather (Gathering Shards to All)** — Collective operation where every worker acquires the complete concatenated dataset across all ranks
- **AllReduce (Reduction + Distributing Result to All)** — Collective operation where every worker computes and receives the global sum/reduction
- **AllToAll (Scatter/Gather from All to All)** — Collective operation where each participant sends distinct chunks to every other participant
- **Arithmetic intensity (FLOP / Transferred Byte)** — Metric measuring data reuse: the ratio of floating-point operations performed per byte moved across the memory bus
- **Backward (Backward Pass)** — Propagation phase computing gradients of the loss with respect to model parameters
- **Bandwidth (Data Transfer Rate)** — Volume of data transmitted per unit of time across an interconnect or memory bus
- **Bisection bandwidth (Bisection Network Capacity)** — Minimum available bandwidth across the network when partitioned into two equal halves
- **BMC (Baseboard Management Controller)** — Dedicated service processor monitoring server hardware independently of the host operating system
- **Checkpoint (Persistent Training State)** — Durable snapshot of model weights, optimizer states, and metadata to resume execution following failures
- **Collective (Multi-Participant Communication)** — Coordinated communication primitive enabling distributed GPUs to exchange or aggregate data
- **CP (Context Parallelism)** — Parallelization strategy partitioning long token sequences across multiple GPUs along the sequence dimension
- **Critical batch size (Per-GPU Token Threshold for Data Parallelism)** — The per-GPU token batch, about C/W, above which compute outlasts gradient synchronization; it grows by E/k for MoE.
- **CUDA core (General-Purpose GPU Arithmetic Unit)** — Standard execution unit executing scalar and vector floating-point and integer operations
- **Dataloader (Training Data Ingestion Pipeline)** — Pipeline responsible for fetching, augmenting, batching, and staging samples into GPU memory
- **DMA (Direct Memory Access)** — Hardware mechanism enabling subsystem memory transfers without involving the host CPU
- **DP (Data Parallelism)** — Parallel execution of identical model copies over distinct data mini-batches
- **EP (Expert Parallelism)** — Distributed routing placing distinct expert sub-networks on different GPUs in MoE architectures
- **Fabric (Interconnect Network Mesh)** — High-performance interconnect switching topology linking servers together
- **FLOP/s (Floating-Point Operations per Second)** — Standard measure of computational execution rate
- **FSDP (Fully Sharded Data Parallelism)** — Data parallelism framework sharding model parameters, gradients, and optimizer states across ranks to minimize VRAM footprint
- **GDS (GPUDirect Storage)** — Direct DMA data path between NVMe storage and GPU HBM bypassing CPU bounce buffers
- **GEMM (General Matrix–Matrix Multiplication)** — Canonical compute-heavy matrix multiplication operation
- **GEMV (General Matrix–Vector Multiplication)** — Memory-bandwidth-sensitive matrix-by-vector multiplication operation
- **GQA (Grouped-Query Attention)** — Attention mechanism where multiple query heads share a single key/value head to reduce KV cache size
- **HBM (High Bandwidth Memory)** — Stacked high-throughput primary memory subsystem integrated directly on the GPU substrate
- **HFU (Hardware FLOPs Utilization)** — Ratio of total floating-point operations executed by hardware relative to theoretical peak compute
- **I/O (Input/Output)** — Data transfer operations between memory, storage, and external networks
- **Kernel (GPU Execution Function)** — Function compiled for execution across massive GPU thread grids
- **KV cache (Attention Key/Value Cache)** — Stored key and value representations of past tokens retained across auto-regressive generation steps
- **Latency (Response Delay)** — Time elapsed between dispatching a request and receiving the result
- **L1/L2 (Cache Hierarchies)** — On-chip static memory hierarchies buffering frequently accessed data near execution units
- **MFU (Model FLOPs Utilization)** — Ratio of theoretical minimal floating-point operations required for model execution relative to peak hardware FLOP/s
- **DualPipe (Bidirectional Pipeline Scheduling)** — Pipeline scheduling technique overlapping forward and backward execution phases with communication to reduce pipeline bubbles
- **Exposed communication (Unoverlapped Network Time)** — Collective communication duration that cannot be masked behind compute, resulting in idle GPU stall cycles
- **FP8 (8-Bit Floating-Point Representation)** — Low-precision numeric format reducing memory traffic and doubling compute throughput, requiring careful numerical scaling
- **HSDP (Hybrid Sharded Data Parallelism)** — Two-tiered distributed strategy applying parameter sharding within dense intra-node domains and replication across nodes
- **Kernel fusion (Combining Multiple GPU Operations)** — Compilation technique merging adjacent operations into a single kernel to eliminate intermediate DRAM round-trips
- **MoE (Mixture-of-Experts)** — Sparse architecture routing each token dynamically to a small subset of specialized expert networks
- **QKV merging (Fused Attention Projections)** — Fusing Query, Key, and Value projection linear layers into a single large GEMM to maximize Tensor Core utilization
- **Selective activation checkpointing (Selective Activation Recomputation)** — Strategy storing cheap-to-keep tensors while recomputing memory-heavy activation layers during backward passes
- **Time-to-quality (Time to Target Model Quality)** — True benchmark metric measuring wall-clock time required to achieve target loss or evaluation metrics, beyond raw throughput
- **MTBF (Mean Time Between Failures)** — Statistical expectation of operational elapsed time between unexpected hardware or cluster faults
- **NUMA (Non-Uniform Memory Access)** — Multi-socket host architecture where memory access latency depends on the physical distance to the memory controller
- **NVLink (NVIDIA High-Speed GPU Interconnect)** — Dedicated high-bandwidth, cache-coherent interconnect link connecting GPUs within a node
- **NVSwitch (NVLink Switching Hardware)** — Crossbar switch fabric establishing full-bandwidth, non-blocking all-to-all connectivity across NVLink-attached GPUs
- **NVFP4 (NVIDIA 4-bit Floating Point)** — Blackwell's E2M1 format with an FP8 scale per 16-value block and an FP32 tensor scale; trades range for 4× fewer bytes than BF16.
- **Occupancy (Active Warp Occupancy Ratio)** — Ratio of active resident warps on a Streaming Multiprocessor relative to theoretical hardware limits
- **Pipeline bubble (Pipeline Idle Stall Region)** — Operational dead time where a pipeline stage sits idle awaiting upstream activations or downstream gradients
- **PP (Pipeline Parallelism)** — Partitioning consecutive layers of a neural network sequentially across multiple GPUs
- **Prefill (Prompt Processing Phase)** — Initial inference phase executing parallel GEMMs over the entire input context to generate the first token
- **Register (Thread-Local Fast Register Storage)** — Ultra-low-latency on-chip storage directly attached to thread execution units
- **RDMA (Remote Direct Memory Access)** — Direct memory access across network adapters without host operating system or CPU intervention
- **ReduceScatter (Reduction and Shard Distribution)** — Collective operation reducing data across ranks and distributing disjoint resulting segments to each participant
- **RoCE (RDMA over Converged Ethernet)** — Network protocol routing RDMA traffic across standard converged Ethernet infrastructures
- **Roofline (Operational–Memory Performance Model)** — Visual performance model delineating whether an operation is bound by memory bandwidth or compute capacity
- **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** — In-network computing technology offloading collective reductions directly to InfiniBand switch hardware
- **SM (Streaming Multiprocessor)** — Primary multithreaded core compute engine of NVIDIA GPUs
- **SP (Sequence Parallelism)** — Technique sharding non-tensor-parallel operations (such as LayerNorm and Dropout) along the sequence length dimension
- **Tensor Core (Specialized Matrix Multiplication Engine)** — Hardware execution unit executing mixed-precision matrix multiply-accumulate operations in single clock cycles
- **TMA (Tensor Memory Accelerator)** — Hardware engine in Hopper/Blackwell moving multidimensional tensor blocks asynchronously between global memory and shared memory
- **TP (Tensor Parallelism)** — Intralayer model parallelism splitting weight matrices and linear layers across multiple GPUs
- **Warp (32-Thread Execution Bundle)** — Fundamental scheduling unit of 32 parallel threads executing instructions in lockstep
- **Warp divergence (Execution Path Divergence within a Warp)** — Performance penalty occurring when threads in the same warp diverge along conditional execution branches
- **WGMMA (Warpgroup Matrix Multiply-Accumulate)** — Hopper instruction family enabling 128-thread warp groups to execute asynchronous matrix multiply-accumulate operations
- **Young–Daly (Optimal Checkpoint Interval Model)** — Analytic formula balancing checkpoint creation overhead against expected recomputation cost after node failures
- **Admission control (Capacity-Aware Request Ingestion)** — Ingress gateway mechanism throttling or rejecting requests to prevent cluster oversubscription
- **Control plane (Management and Orchestration Plane)** — System layer managing resource allocation, scheduling policies, and cluster state
- **Data plane (Execution and Data Path)** — Low-level execution plane processing active model computation, tensor operations, and data transfers
- **Dataset lineage (Training Data Provenance Tracking)** — Complete audit trail tracking versions, tokenization, filtering, and transformations of training data corpora
- **Disaggregated serving (Decoupled Prefill and Decode Pools)** — Serving architecture separating prefill and decode tasks onto independently scaled, specialized GPU pools
- **Gang scheduling (All-or-Nothing Distributed Scheduling)** — Scheduling policy ensuring all interdependent worker processes are allocated and started concurrently
- **Goodput (Service-Compliant Useful Throughput)** — Volume of requests or tokens successfully processed within strict SLO latency boundaries
- **GQA (Grouped-Query Attention)** — Architectural design where multiple query heads share fewer key/value heads to conserve memory bandwidth during decoding
- **ISL / OSL (Input / Output Sequence Length)** — Token counts defining prompt length and generated output completion length for inference requests
- **KV-aware routing (Cache-Locality-Aware Request Routing)** — Serving gateway routing strategy directing queries to workers that already hold matching prompt prefix KV caches
- **Little's Law (Queueing Law $L = \lambda W$)** — Fundamental theorem establishing that average system concurrency equals arrival rate multiplied by average residence time
- **MIG (Multi-Instance GPU)** — Hardware partitioning technology dividing a physical GPU into fully isolated, hardware-enforced instances
- **PagedAttention (Paged KV Memory Management)** — Memory management paradigm allocating KV cache into non-contiguous physical pages to eliminate external memory fragmentation
- **PUE (Power Usage Effectiveness)** — Data center efficiency metric representing the ratio of total facility power consumed to IT equipment power
- **RPO / RTO (Recovery Point Objective / Recovery Time Objective)** — Metrics defining acceptable training step loss and maximum permissible downtime during disaster recovery
- **RAS (Reliability, Availability, Serviceability)** — Engineering framework for fault detection, resilient operation, and non-disruptive cluster maintenance
- **SDC (Silent Data Corruption)** — Undetected calculation errors or memory bit-flips producing corrupt model states without triggering system exceptions
- **SLO (Service Level Objective)** — Contractual quantitative target defining acceptable latency, throughput, and error rate performance thresholds
- **Speculative decoding (Draft-and-Verify Token Generation)** — Latency-reduction technique using a lightweight draft model to generate candidate tokens verified in parallel by a target model
- **Strong scaling (Fixed-Problem Scaling)** — Scaling behavior when increasing the number of accelerators while keeping total problem size constant to reduce total completion time
- **TCO (Total Cost of Ownership)** — Comprehensive financial calculation incorporating capital hardware expenditures, energy, cooling, networking, and ongoing operational costs
- **Tenant isolation (Workload and Boundary Partitioning)** — Security and performance isolation ensuring independent users or workloads do not compromise shared cluster resources
- **Time-to-quality (Convergence Duration to Target Metric)** — Total training duration required to reach a specific validation loss or benchmark capability, rather than mere iteration speed
- **Weak scaling (Proportional Problem Scaling)** — Scaling behavior when expanding cluster resources proportionally with problem size to maintain constant completion time

## References and verification links

A portion of the references below represents **original technical context**, while another portion provides **independent vendor documentation or research paper verification**. In no case are marketing theoretical upper limits treated as automatically equivalent to real-world application throughput.

- **[K1] Primary source:** Engin Bozaba, [*Understanding GPUs and LLM Scaling*](https://muhendis.github.io/en/blog/post.html?slug=understanding-gpus-and-llm-scaling) — [direct Markdown source](https://github.com/muhendis/muhendis.github.io/blob/master/en/blog/posts/understanding-gpus-and-llm-scaling.md). Compute–network–storage axes, SM → DGX → SuperPOD hierarchy, and didactic hardware context.
- **[K2] Hardware micro-architecture:** NVIDIA, [*Hopper Architecture In-Depth*](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — H100 SXM SM/Tensor Core/L2/HBM/NVLink components.
- **[K3] Distributed model strategies:** NVIDIA, [*Megatron Core Parallelism Strategies Guide*](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) and [*Context Parallelism*](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html). TP, PP, CP, EP, and FSDP.
- **[K4] DGX SuperPOD reference:** NVIDIA, [*Network Fabrics*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html) and [*Major Components*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/components.html). Rail topology, 127 nodes, 32 leaf + 16 spine switches, dedicated storage fabric.
- **[K5] Large-scale training:** Dubey et al., [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783), 2024. 16,384 H100s, 4D parallelism, MFU ranges, Tectonic, and 54-day interruption statistics.
- **[K6] GPU–storage data path:** NVIDIA, [*GPUDirect Storage Design Guide*](https://docs.nvidia.com/gpudirect-storage/design-guide/) and [*Overview Guide*](https://docs.nvidia.com/gpudirect-storage/overview-guide/). CPU bounce buffer and compatibility mode details.
- **[K7] In-network computing:** NVIDIA, [*SHARP In-Network Computing*](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing). Collective acceleration.
- **[K8] Attention kernel optimization:** Shah et al., [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/abs/2407.08608), 2024; [PyTorch technical blog](https://docs.pytorch.org/blog/flashattention-3/).
- **[K9] MoE and DualPipe:** DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) and [DualPipe repository](https://github.com/deepseek-ai/DualPipe).
- **[K10] Checkpoint theory:** [*A survey on checkpointing strategies: Should we always checkpoint à la Young/Daly?*](https://www.sciencedirect.com/science/article/pii/S0167739X24003777), 2024; assumptions of the classic $\sqrt{2MC}$ approximation.
- **[K12] Concrete Blackwell results in MFU optimization:** Lambda, [*Reproducible MFU Optimization Techniques to Boost Your Training Efficiency*](https://lambda.ai/hubfs/4.%20Resources/White%20Papers/Lambda%20MFU.pdf), November 2025; [benchmark summary](https://lambda.ai/llama-3-1-blackwell-mfu-benchmarks). Llama 3.1 8B/70B/405B, TorchTitan, TP, selective activation checkpointing, torch.compile, prefetch, and BF16/FP8 comparisons.
- **[K13] MFU on 10k+ GPUs:** Jiang et al., [*MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs*](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng), NSDI 2024; [technical paper PDF](https://www.usenix.org/system/files/nsdi24-jiang-ziheng.pdf). 55.2% MFU on 175B parameters across 12,288 GPUs.
- **[K14] `torch.compile` boundaries:** PyTorch, [*torch.compile API*](https://docs.pytorch.org/docs/stable/generated/torch.compile) and [*Introduction to torch.compile*](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). Graph breaks, kernel optimization, and dynamic shapes.
- **[K15] Activation memory and recomputation:** PyTorch, [*Current and New Activation Checkpointing Techniques*](https://pytorch.org/blog/activation-checkpointing-techniques/). Selective vs. full checkpointing and the memory-compute trade-off.
- **[K16] Peak theoretical FLOPs across precisions:** NVIDIA, [*Exemplar Performance — Peak Theoretical Throughput*](https://github.com/NVIDIA/exemplar-performance/blob/main/README.md). Non-sparse BF16/FP8 baselines for H100, B200, B300, GB200, and GB300.
- **[K17] FP8 and mixed-precision constraints:** NVIDIA, [*Transformer Engine Data Type Support Matrix*](https://nvidia.github.io/TransformerEngine/support_matrix.html) and [*GEMM Profiling Tutorial*](https://docs.nvidia.com/deeplearning/transformer-engine/examples/gemm_profiling/gemm_profiling.html). FP8 support matrix, non-GEMM operations, and low-precision overheads.
- **[K11] Further reading on scaling:** Austin et al., [*How to Scale Your Model: How to Think About GPUs*](https://jax-ml.github.io/scaling-book/gpus/). Quantified pedagogical examples on GPU architecture, collective communication, and model parallelism.
- **[K18] Enterprise risk and data governance:** NIST, [*AI Risk Management Framework: Generative AI Profile*](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), 2024. Data provenance, privacy, model risks, and lifecycle governance.
- **[K19] Decode optimization:** vLLM, [*Speculative Decoding*](https://docs.vllm.ai/en/latest/features/speculative_decoding/). Acceptance rate, methods, and performance dependence on traffic patterns.
- **[K20] Production inference components:** NVIDIA, [*Dynamo Documentation*](https://docs.nvidia.com/dynamo/latest/). Router, prefill/decode disaggregation, KV management, and scaling.
- **[K21] Architectural parameters for KV cache sizing:** [*Llama 3.1 70B model configuration*](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json). 80 layers, 8 KV heads, hidden size 8192, and 64 attention heads; the calculation formula is a didactic derivation.
- **[K22] Paged attention and KV cache:** vLLM, [*Automatic Prefix Caching*](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [*Cache Design / Isolation*](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md), and [*vLLM Security*](https://docs.vllm.ai/en/latest/usage/security/). KV block management, prefix reuse, and isolation.
- **[K23] Disaggregated inference topology:** NVIDIA Dynamo, [*Disaggregated Serving*](https://docs.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview) and [*KV-Aware Routing*](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving). Prefill/decode workers and KV transmission overheads.
- **[K24] Training job scheduling:** Kubernetes SIGs, [*Kueue All-or-Nothing Scheduling*](https://github.com/kubernetes-sigs/kueue/blob/main/site/content/en/v0.20/docs/concepts/all_or_nothing.md). Co-allocating resources for distributed workers.
- **[K25] GPU partitioning:** NVIDIA, [*MIG User Guide*](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/introduction.html). GPU partitioning mechanisms and support matrices.
- **[K26] System power reference:** NVIDIA, [*DGX H100/H200 System Specifications*](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). 10.2 kW maximum system power specification.
- **[K27] Physical facility design:** Uptime Institute, [*AI embraces liquid cooling*](https://journal.uptimeinstitute.com/ai-embraces-liquid-cooling-but-enterprise-it-is-slow-to-follow/), 2025. Rack density, cooling methods, and power capacity planning.
- **[K28] Training data quality:** Meta, [*Introducing Meta Llama 3*](https://ai.meta.com/blog/meta-llama-3), 2024. Data filtering, deduplication, and quality-driven curation.
- **[K29] Training data and model management:** NVIDIA, [*Megatron Bridge Documentation*](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html) and [*Data Preparation*](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/training/data-preparation.md). Training recipes and indexed dataset approaches.
- **[K30] Scaling laws:** Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556), 2022. Interplay among parameter count, token budget, and fixed compute envelopes.
- **[K31] Measurement methodology:** MLCommons, [*MLPerf Training*](https://mlcommons.org/benchmarks/training/) and [*Training Rules*](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc). Measuring time-to-target model quality.
- **[K32] Collective communication health:** NVIDIA, [*NCCL RAS Documentation*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html). RAS diagnostics during active distributed workloads.
- **[K33] GPU node isolation:** NVIDIA, [*DGX SuperPOD Health Checks*](https://docs.nvidia.com/dgx-superpod/administration-guide-dgx-superpod/latest/system-health.html). Health auditing, node cordoning, and NCCL validation.
- **[K34] Production inference benchmarking:** MLCommons, [*What is MLPerf Endpoints?*](https://docs.mlcommons.org/endpoints-submission-docs/understand/what-is-mlperf-endpoints/). Concurrency–throughput–latency Pareto frontiers.
- **[K35] GPU telemetry:** NVIDIA, [*DCGM Exporter*](https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html). Exposing GPU metrics in Prometheus format.
- **[K36] DCGM health monitoring:** NVIDIA, [*Health Monitoring*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/health-monitoring.html) and [*Diagnostics*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/dcgm-diagnostics.html). Xid events, ECC errors, page retirement, PCIe metrics, and stress diagnostics.
- **[K37] DCGM telemetry / field constraints:** NVIDIA, [*Profiling*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html) and [*DCGM Exporter Metrics*](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html). Metric support bounds, sampling rates, multiplexing, and exporter configurations.
- **[K38] NCCL diagnostics:** NVIDIA, [*NCCL RAS*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html) and [*Diagnostics*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/diagnostics.html). Passive and active diagnostics, rank inconsistencies, and communication stalls.
- **[K39] NCCL microbenchmarking:** NVIDIA, [*nccl-tests*](https://github.com/NVIDIA/nccl-tests). Single-node and multi-node AllReduce performance benchmarking examples.
- **[K40] Disaggregated inference trade-offs:** NVIDIA Dynamo, [*Disaggregated Serving Overview*](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/disaggregated-serving/overview.md). Decoupled prefill/decode pools, KV transmission overheads, and contraindications.
- **[K41] Training performance and quality:** MLCommons, [*MLPerf Training Reference Implementations*](https://github.com/mlcommons/training). Time-to-target convergence criteria and reproducible benchmark workflows.
- **[K42] Inference benchmarking methodologies:** MLCommons, [*MLPerf Inference Benchmarks*](https://docs.mlcommons.org/inference/benchmarks/) and [*Submission Guide*](https://docs.mlcommons.org/inference/submission/). Latency and accuracy constraints across diverse load profiles.
- **[K43] Request-level serving benchmarking:** SGLang, [*Serving Benchmark Guide*](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx). TTFT/TPOT/ITL metrics, request throughput, and input/output token distributions.
- **[K44] Multimodal model processing:** Hugging Face, [*Multimodal Chat Templates*](https://huggingface.co/docs/transformers/chat_templating_multimodal). Image/video preprocessors, multimodal processors, and frame sampling parameters.
- **[K45] GPU sharing and isolation:** NVIDIA, [*MIG Concepts*](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/concepts.html) and [*Introduction*](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html). Streams, MPS, and MIG functional and isolation boundaries.
- **[K46] Kubernetes accelerator scheduling architecture:** Kubernetes, [*Dynamic Resource Allocation*](https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/) and [*v1.34 DRA GA*](https://kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/). DeviceClass, ResourceClaim, and version-dependent hardware allocation capabilities.
- **[K47] Profiling and instrumentation limits:** NVIDIA, [*Nsight Systems*](https://developer.nvidia.com/nsight-systems) and [*User Guide*](https://docs.nvidia.com/nsight-systems/UserGuide/). CUDA tracing, GPU/NVLink/DRAM hardware counters, multi-node timelines; for kernel-level analysis, see [*Nsight Compute Profiling Guide*](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- **[K48] FSDP2 mechanics:** PyTorch, [*Fully Sharded Data Parallel — `fully_shard`*](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) and [*Getting Started with FSDP2*](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html). DTensor sharding, pre-forward AllGather, backward ReduceScatter, and communication prefetching.
- **[K49] Seminal PagedAttention paper:** Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://doi.org/10.1145/3600006.3613165), SOSP 2023. KV cache paging, virtual memory allocation, eliminating memory fragmentation, and vLLM throughput experiments.
- **[K50] Throughput vs. quality distinction:** MLCommons, [*MLPerf Training Rules*](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc). Attaining target accuracy and wall-clock training duration.
- **[K51] Rack-scale NVLink domain:** NVIDIA, [*GB300 NVL72 specifications*](https://www.nvidia.com/en-us/data-center/gb300-nvl72/). 72 Blackwell Ultra GPUs, Grace CPUs, and NVLink interconnect domain; evaluate vendor peak performance figures independently of sparsity assumptions.
- **[K52] Measured Hopper latencies:** Luo et al., [*Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*](https://arxiv.org/abs/2402.13499), 2024. Load latencies: SMEM ~29, L1 ~41, L2 ~263, global memory ~479 cycles.
- **[K53] NVSwitch fabric bring-up:** NVIDIA, [*Fabric Manager for NVIDIA NVSwitch Systems*](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). How each GPU's 18 NVLinks split 5 + 4 + 4 + 5 over four NVSwitches, and why the NVLink fabric needs Fabric Manager.
- **[K54] InfiniBand switch:** NVIDIA, [*QM97X0 NDR 400Gb/s InfiniBand Switch Systems User Manual*](https://docs.nvidia.com/networking/display/qm97x0um/introduction). 64 NDR ports per QM9700 and SHARP in-network reduction.
- **[K55] SuperPOD scaling:** NVIDIA, [*DGX SuperPOD Architecture (DGX H100)*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html). Two tiers up to 8 SUs, a core tier from 16 SUs; 512 leaf, 512 spine and 256 core switches for 2,048 DGX.
- **[K56] GPU generation datasheets:** NVIDIA, [V100](https://www.nvidia.com/en-us/data-center/v100/), [A100](https://www.nvidia.com/en-us/data-center/a100/), [H200](https://www.nvidia.com/en-us/data-center/h200/) and [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/). HBM capacity and bandwidth, dense tensor peaks.
- **[K57] 4-bit floating point:** NVIDIA, [*Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/). E2M1 values, 16-value blocks with E4M3 scales, an FP32 tensor scale, and the comparison with MXFP4.
