An AI hardware platform is not merely a collection of GPU servers racked side by side. It is a balanced system capable of absorbing concurrent stress across compute, HBM, interconnects, storage, and power facilities.

**The central idea:** The right hardware is not chosen by picking the "fastest GPU"; it is selected by calculating where tensors are generated, where they reside, and which interconnects they traverse during training, fine-tuning, and inference.

We first translate workloads into physical resource demands. Then we assemble the GPU node, scale-out network fabric, storage tiers, power infrastructure, and failure domains into a concrete reference design anchored around a 70B parameter model.

> **What we carry forward:** [The cluster diagnosis article](post.html?slug=from-gpus-to-ai-supercomputers) mapped low MFU, slow checkpoints, and queue spikes to physical bottlenecks. In this final article of the series, we convert those telemetry insights into procurement and architectural placement decisions.

> **By the end of this article**
>
> - You will analytically map pre-training, full fine-tuning, PEFT, and inference workloads to exact hardware footprints.
> - You will evaluate HBM capacity, HBM bandwidth, NVLink, scale-out fabric, storage, and power budgets within a unified architectural framework.
> - You will mathematically derive why serving nodes and fine-tuning nodes for a 70B model require fundamentally divergent designs.
> - You will execute a multi-tier benchmark and verification ladder before signing off on a multi-million-dollar GPU cluster procurement.

**In this article**

- [1. Translating a workload into a hardware signature](#1-translating-a-workload-into-a-hardware-signature)
- [2. Designing the GPU node](#2-designing-the-gpu-node)
- [3. From node to cluster: Network and physical placement](#3-from-node-to-cluster-network-and-physical-placement)
- [4. Physical limits beyond the GPU: Storage, power, and failures](#4-physical-limits-beyond-the-gpu-storage-power-and-failures)
- [5. A hardware reference design for a 70B model](#5-a-hardware-reference-design-for-a-70b-model)
- [6. The validation ladder before procurement](#6-the-validation-ladder-before-procurement)
- [7. Testing what we learned](#7-testing-what-we-learned)
- [What questions can the reader now answer independently?](#what-questions-can-the-reader-now-answer-independently)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [References and verification links](#references-and-verification-links)

---

## 1. Translating a workload into a hardware signature

The most expensive mistake in AI platform design is treating all workloads (pre-training, fine-tuning, inference) as homogeneous "GPU demand." In reality, a workload's hardware signature—its compute intensity, memory footprint, and network communication frequency—demands radically different physical architectures.

Across the bottom-up scaling ladder, each tier replicates the components below it while the primary architectural bottleneck physically shifts:

```
Thread -> Warp (32) -> SM -> GPU (HBM3) -> DGX Node (NVLink) -> SU (Leaf) -> SuperPOD (Spine) -> Multi-Pod Cluster
```

<!-- restored-svg:en_full:2 -->
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

### 1.1. Why pretraining must be designed with the network

Foundational model pre-training from scratch represents the most punishing stress test for systems architecture.
- **Compute Profile:** Continuous $6P$ FLOPs per token compute crunch spanning billions of tokens. Tensor Cores operate near full load for months.
- **Communication Demand:** Intra-layer Tensor Parallelism triggers two AllReduces per Transformer block, Pipeline Parallelism streams activations across nodes, and Data Parallelism fires hundreds of megabytes of gradient synchronization per step.
- **Network-First Architecture:** In pre-training, the network is not a cabling afterthought; it is the central nervous system. Without a 1:1 non-blocking bisection bandwidth fabric, thousands of H100 GPUs waste half their time stalled at collective synchronization barriers, crashing cluster MFU below 20%.

### 1.2. Why full fine-tuning and PEFT have different capacity models

When an enterprise adapts a 70-billion parameter model to internal data, two main paths exist: **Full Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**. Their hardware footprints are polar opposites:

<!-- restored-svg:en_full:39 -->
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

The memory mathematics of both approaches:

1. **Full Fine-Tuning (16 Bytes per Parameter):**
   Every model parameter is updated. Under AdamW mixed-precision (FP16/BF16), static memory demands:
   - Model Weights (FP16/BF16): $2\text{ bytes/param}$
   - Gradients (FP16/BF16): $2\text{ bytes/param}$
   - AdamW States (FP32 master weights, momentum, variance): $4 + 4 + 4 = 12\text{ bytes/param}$
   - **Total Static State:** Exactly **$16\text{ bytes}$** per parameter!
   For a 70B model, static state consumes:
   $$M_{\mathrm{static}} = 70 \times 10^9 \times 16\text{ bytes} \approx 1,120\text{ GB} = 1.12\text{ TB}$$
   Factoring in activation tensors and workspace buffers requires at least $1.5\text{ TB}$ HBM. This workload demands at least two 8-GPU DGX H100 nodes ($16 \times 80\text{ GB} = 1,280\text{ GB}$) orchestrated via FSDP/ZeRO-3.
2. **PEFT / LoRA (Low-Rank Adaptation):**
   Original model weights are frozen ($2\text{ bytes/param}$). Only inserted low-rank decomposition matrices ($A$ and $B$, rank $r=16$ or $64$) are trained.
   - Frozen Base Weights (FP16): $70\text{B} \times 2\text{ bytes} = 140\text{ GB}$ (or $35\text{ GB}$ under 4-bit QLoRA).
   - Trainable Parameters: Typically 0.1% to 0.5% of the model ($\approx 350\text{ million}$ parameters).
   - Optimizer State: Tracked exclusively for adapters ($350\text{M} \times 16\text{ bytes} \approx 5.6\text{ GB}$).
   - **Conclusion:** With LoRA, a 70B model trains comfortably on a single 8-GPU node—or even on one or two 80 GB GPUs with QLoRA!

### 1.3. Why inference is more than fitting weights in memory

A recurring trap in production inference planning: "A 70B FP16 model occupies $140\text{ GB}$; I have two 80 GB GPUs ($160\text{ GB}$), so I can easily serve it."
In production, this assumption **guarantees instant crashes**.

Inference memory divides into three distinct tiers:
$$
M_{\mathrm{inference}} = M_{\mathrm{weights}} + M_{\mathrm{KV\_cache}} + M_{\mathrm{runtime\_buffers}}
$$

1. **Weights ($M_{\mathrm{weights}}$):** $140\text{ GB}$.
2. **Runtime Buffers ($M_{\mathrm{runtime\_buffers}}$):** CUDA graphs, NCCL buffers, and scratch activation memory require $10 - 15\text{ GB}$.
3. **Remaining Memory ($M_{\mathrm{KV\_cache}}$):** Across two GPUs, only $160 - 140 - 10 = 10\text{ GB}$ remains!
   - In a 70B model (Grouped-Query Attention, $8$ KV heads, $128$ head dim, $80$ layers), 1 token's KV cache requires:
     $$\text{KV per token} = 2 \times 2\text{ bytes} \times 80\text{ layers} \times 8\text{ heads} \times 128 = 327,680\text{ bytes} \approx 320\text{ KB/token}$$
   - A $10\text{ GB}$ buffer holds only $32,000$ tokens total. As soon as 8 concurrent users submit $4\text{K}$ context queries, the engine crashes with an Out-of-Memory (OOM) fault!
4. **The Fix:** Serving a 70B model requires not two, but at least **four 80 GB GPUs** ($320\text{ GB}$ HBM), leaving $160\text{ GB}$ of dedicated dynamic KV cache to support dozens of concurrent requests at low latency.

---

## 2. Designing the GPU node

Selecting an AI server is not about chasing headline TFLOP/s ratings; it is about balancing intra-node subsystems to prevent internal chokepoints.

### 2.1. Separate HBM capacity from HBM bandwidth

HBM presents two orthogonal physical dimensions that constrain different phases of execution:
- **HBM Capacity (GB):** Dictates how many weights and how large a KV cache pool fit on the device (The capacity ceiling).
- **HBM Bandwidth (TB/s):** Dictates the velocity at which weights stream from memory into Tensor Cores (The throughput ceiling).

As established in Part 2:
- **Prefill Phase:** Large prompt batches generate matrix-matrix multiplications (GEMM), operating at high arithmetic intensity in the **Compute-Bound** regime (governed by Tensor Core TFLOP/s).
- **Decode Phase:** Generating tokens sequentially requires streaming all $140\text{ GB}$ of weights from HBM to SMs for every single token (GEMV). Arithmetic intensity is near unity ($\approx 1$), placing execution strictly in the **Memory-Bandwidth Bound** regime.
- The throughput difference between an H100 SXM5 ($3.35\text{ TB/s}$ HBM) and an H200 ($4.8\text{ TB/s}$) during decode is driven not by compute FLOPs, but directly by memory bandwidth.

### 2.2. Precision is a data-path decision

Migrating from FP32 to FP16/BF16, and onward to FP8 and NVFP4, is not merely an algorithmic accuracy choice; it is an architectural decision that alters traffic across every data path:

$$
\text{Bandwidth Requirement} \propto \text{Precision (Bytes)}
$$

- Adopting FP8 ($1\text{ byte}$) instead of FP16 ($2\text{ bytes}$):
  1. Halves memory traffic from HBM during decode, doubling token generation speed theoretically.
  2. Reduces activation volume across NVLink and PCIe by 50%, accelerating collective barriers.
  3. Doubles physical Tensor Core throughput ($990 \to 1,979\text{ TFLOP/s}$).

### 2.3. Which work belongs inside an NVLink domain?

NVIDIA NVSwitch delivers $900\text{ GB/s}$ bidirectional all-to-all bandwidth at $0.5\ \mu\text{s}$ latency across 8 GPUs in a chassis. This domain is a precious physical resource:
- **Must Stay Intra-Node:** Tensor Parallelism (TP). Frequent intra-layer AllReduces cannot tolerate PCIe or network fabric latency.
- **Can Span Inter-Node Fabrics:** Pipeline Parallelism (PP) and Data Parallelism (DP/FSDP). These collectives fire only at stage boundaries or step completion and can be scheduled over InfiniBand or RoCE.

### 2.4. Which physical problem do MIG, MPS, and time-slicing solve?

In enterprise environments, not all workloads occupy an entire 8-GPU chassis. For smaller embedding models, evaluation benchmarks, or developer sandboxes, partitioning strategies include:
1. **MIG (Multi-Instance GPU):** Partitions an H100 into up to 7 hardware-isolated instances, each with dedicated SMs, memory controllers, and crossbar paths. It is the gold standard for secure, multi-tenant enterprise isolation.
2. **MPS (Multi-Process Service):** Allows multiple CUDA processes to share an address space concurrently on the same hardware. Ideal for co-locating low-batch inference workloads.
3. **Time-Slicing:** Round-robin GPU sharing; generally avoided for deep learning due to severe context-switch latencies.

### 2.5. Which worksheet defends a node choice?

When presenting a GPU node selection to technical leadership, evaluate against this 5-dimensional scorecard:

| Architectural Dimension | Minimum Threshold | Engineering Justification |
|---|---|---|
| **GPU Form Factor** | SXM5 (over PCIe) | $900\text{ GB/s}$ all-to-all NVSwitch mesh (PCIe is capped at $128\text{ GB/s}$) |
| **HBM per GPU** | $\ge 80\text{ GB}$ (HBM3/HBM3e) | Accommodates 70B models at TP=4/TP=8 without KV-cache exhaustion |
| **Host System Memory** | $\ge 2\text{ TB}$ (2x GPU HBM) | Buffers dataloader queues and enables two-tier asynchronous checkpointing |
| **Fabric Interconnect** | 8x 400G OSFP ($400\text{ Gb/s}$ per GPU) | Single-hop rail-optimized RDMA in multi-node clusters |
| **Power and Thermal** | $10.2\text{ kW}$ sustained per chassis | Ensures GPUs sustain boost clocks without thermal throttling |

---

## 3. From node to cluster: Network and physical placement

Once nodes are properly configured, they must be woven into a balanced cluster fabric.

### 3.1. Every parallelism dimension creates physical traffic

Parallelism is not an abstract algorithm; it is physical bitstreams traversing copper and optics:
- $TP$ generates high-frequency AllReduce storms inside each layer.
- $PP$ streams microbatch activations sequentially between adjacent stages.
- $DP / FSDP$ floods the fabric with heavy gradient AllGather and ReduceScatter traffic.
- $EP$ (MoE) creates an all-to-all shuffle spanning every expert GPU across the cluster.

### 3.2. Why GPU–NIC affinity is not a cabling detail

Inside a DGX H100 node, each GPU links through dedicated PCIe Gen5 switches to a specific ConnectX-7 NIC:
- GPU 0 $\to$ NIC 0 (NUMA Node 0)
- GPU 7 $\to$ NIC 7 (NUMA Node 1)
If cabling or Kubernetes CNI device plugins mistakenly map GPU 0 to NIC 7, packets cross the high-latency CPU UPI bridge. Even with $400\text{ Gb/s}$ links, latency triples and bisection bandwidth collapses.

<!-- restored-svg:en_full:24 -->
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

### 3.3. Rail alignment and bisection bandwidth

Preventing network traffic from choking cluster spine switches requires strict **Rail Optimization**:

1. **Four Dedicated Physical Networks:**
   Each DGX node terminates onto four isolated physical networks:
   - **Compute Fabric:** 8x $400\text{ Gb/s}$ InfiniBand / RoCE ports dedicated strictly to GPU collectives, isolated from storage and management traffic.
   - **Storage Fabric:** 2x dual-port $400\text{ Gb/s}$ NICs ($800\text{ Gb/s}$ aggregate) handling dataset ingestion and checkpoint bursts.
   - **In-Band Management:** $100\text{ GbE}$ Ethernet carrying OS, Kubernetes API, scheduler, and telemetry signals.
   - **Out-of-Band (OOB) BMC:** $1\text{ GbE}$ dedicated IPMI network for remote power cycling and console triage even if the host OS crashes.
2. **1:1 Non-Blocking Clos:**
   The compute fabric maintains a 1:1 non-blocking oversubscription ratio between Leaf and Spine tiers, ensuring any two GPUs in a 1,024-node cluster communicate at line rate ($50\text{ GB/s}$).

### 3.4. Why network headroom is not sized from averages

In cloud web services, networks are sized for average throughput. In distributed AI, that approach guarantees failure.
Training traffic is inherently **bursty, synchronized, and lockstep**:
- The instant backpropagation completes, all 1,024 GPUs burst onto the fabric simultaneously.
- Sizing for average throughput causes switch buffer exhaustion, packet drops, PFC pause storms, and minutes of exposed synchronization stalls. Fabrics must be provisioned for **peak burst line rate**.

---

## 4. Physical limits beyond the GPU: Storage, power, and failures

In production data centers, cluster scalability is bounded not by GPU availability, but by physical facility constraints.

### 4.1. Storage performs two different jobs

As proven in Part 4, storage architecture must decouple into two distinct operational tiers:
1. **Dataset Ingestion:** Demands high IOPS, low latency, and zero POSIX metadata lock contention (GPFS, Lustre, or NVMe-oF caches).
2. **Checkpoint Dumps:** Demands massive sustained aggregate write throughput ($> 100 - 250\text{ GB/s}$) to drain in-memory snapshots rapidly.

### 4.2. Power and cooling must be checked before GPU count

A single DGX H100 server draws up to **$10.2\text{ kW}$**!
- A standard server rack housing 4 DGX H100 nodes draws:
  $$\text{Rack Power} = 4 \times 10.2\text{ kW} = 40.8\text{ kW}$$
- Traditional enterprise data centers are engineered for $5 - 10\text{ kW}$ per rack. Dissipating $40\text{ kW}$ with air cooling is physically untenable; fans spinning at maximum RPM still permit thermal throttling.
- **Liquid Cooling Mandate:** Modern AI facilities must deploy Direct-to-Chip Liquid Cooling (DLC) or immersion cooling.
- **PUE Reality:** At a PUE of 1.35, delivering $1\text{ MW}$ of compute requires provisioning $1.35\text{ MW}$ from the electrical grid. Transformer capacity must be validated before ordering hardware.

### 4.3. Connect failure domains to parallelism groups

Physical failure domains (blast radii) follow a clear hierarchy:
`GPU -> PCIe Riser -> Server PSU -> PDU -> Rack -> Switch -> Cooling Loop`

Parallel topologies must map defensively onto these domains:
- Avoid chaining an entire Pipeline Parallelism model replica to a single rack or PDU.
- Distribute Data Parallel replicas across distinct power circuits and racks to prevent a single breaker trip from invalidating an entire distributed run.

### 4.4. Spare capacity is not automatically waste

In a 1,024-GPU cluster with an MTBF of 10–20 hours, hardware faults are frequent.
Without spare nodes, a single broken chassis halts a 1,024-GPU job for days while awaiting vendor RMA.
Reserving **5% to 10% of cluster capacity (10–15 DGX nodes per 1,024 GPUs)** as hot spares is not waste; it is the single most effective investment to maintain 95%+ cluster availability.

---

## 5. A hardware reference design for a 70B model

Let us translate these principles into an enterprise production blueprint: **An End-to-End Platform for a 70-Billion Parameter Model**.

### 5.1. Write the assumptions first

- **Target Model:** Llama 3 70B ($L=80\text{ layers}, H=8192, A=64\text{ heads}, K=8\text{ KV heads (GQA)}$).
- **Precision:** BF16 ($2\text{ bytes/parameter}$).
- **Context Windows:** $8,192$ tokens (Standard) up to $32,768$ tokens (Extended).

### 5.2. Serving node

Objective: Low latency (TTFT $< 300\text{ ms}$, TPOT $< 25\text{ ms/token}$) and high concurrency.
- **Hardware:** 1 server with 4x NVIDIA H100 SXM5 ($320\text{ GB}$ HBM3).
- **Parallelism:** $TP = 4$ across local NVSwitch.
- **Memory Allocation:**
  - Model Weights (BF16): $140\text{ GB}$ ($35\text{ GB/GPU}$).
  - Runtime and Activation Buffers: $20\text{ GB}$.
  - Usable Dynamic KV Cache: $320 - 140 - 20 = \mathbf{160\text{ GB}}$!
- **Concurrency Budget:** Under GQA ($320\text{ KB/token}$), $160\text{ GB}$ accommodates **$500,000$ active tokens**, comfortably serving over 120 concurrent streams at $4\text{K}$ context without memory exhaustion.

### 5.3. Fine-tuning node

Objective: Weekly LoRA runs and periodic Full Fine-Tuning.
- **Hardware:** 2x DGX H100 servers ($16 \times \text{H100 } 80\text{ GB} = 1,280\text{ GB}$ HBM).
- **Parallelism:** $TP = 8$ intra-node, $DP = 2$ inter-node via PyTorch FSDP / ZeRO-3.
- **Full Fine-Tuning Feasibility:** The 70B model's $1,120\text{ GB}$ static state shards across 16 GPUs ($70\text{ GB/GPU}$). With FlashAttention and activation recomputation, it fits within $1,280\text{ GB}$ HBM, running at high MFU without CPU offloading.

### 5.4. Pretraining pool

Objective: Domain-specific 70B pre-training from scratch.
- **Hardware:** 64x DGX H100 nodes (512 GPUs, 8 Scalable Units).
- **Network Fabric:** 8-rail InfiniBand Quantum-2 ($400\text{ Gb/s}$ non-blocking fat-tree).
- **Parallelism:** $TP = 8, PP = 4, DP = 16 \implies 8 \times 4 \times 16 = 512\text{ GPUs}$.
- **Storage:** Lustre NVMe filesystem sustaining $\ge 120\text{ GB/s}$ write throughput to keep checkpoint overhead below 5%.

### 5.5. Why not merge all three pools automatically?

Why not pool all GPUs into a single dynamic Kubernetes cluster?
1. **SLA and Predictability Clashes:** Serving demands strict tail-latency guarantees (P-99 latency); training demands monolithic gang scheduling for weeks.
2. **Network Overprovisioning:** Serving nodes require only local NVLink; running inference on an InfiniBand compute fabric wastes expensive non-blocking fabric ports.
3. **Security and Isolation:** Exposing public serving APIs on the same physical fabric used for proprietary foundation model training violates zero-trust enterprise security boundaries.

### 5.6. How do decisions connect in the 70B design?

End-to-end architectural mapping:
- *Workload:* Llama 3 70B Serving $\to$ *Bottleneck:* Decode phase HBM bandwidth $\to$ *Design:* $TP=4$ H100 SXM5 ($13.4\text{ TB/s}$ aggregate bandwidth).
- *Workload:* 70B Full Fine-Tuning $\to$ *Bottleneck:* AdamW state memory ($1.12\text{ TB}$) $\to$ *Design:* 2x DGX (16 GPUs) + FSDP ZeRO-3.
- *Workload:* 70B Pre-training $\to$ *Bottleneck:* Collective communication & MTBF $\to$ *Design:* 512-GPU rail-optimized InfiniBand + Young-Daly asynchronous checkpointing.

---

## 6. The validation ladder before procurement

Never rely on vendor marketing datasheets during procurement; enforce a rigid **6-Tier Validation Ladder**:

<!-- restored-svg:en_full:29 -->
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

### 6.1. Six validation layers

The multi-tier acceptance hierarchy from physical cabling to workload SLOs:

1. **Layer 1 — Physical and Optical Tiers:** Scan optical transceiver signal power, cable attenuation, and switch port error counters (`SymbolError`, `LinkDown`). Zero errors tolerated.
2. **Layer 2 — Host Data Paths and NUMA:** Audit `nvidia-smi topo -m` and `lspci -vvv` to verify PCIe Gen5 $x16$ ($32\text{ GT/s}$) negotiation and proper GPU-to-NIC NUMA alignment across all sockets.
3. **Layer 3 — Point-to-Point RDMA Benchmarks:** Run `ibv_rc_pingpong` to verify unidirectional latency $< 1.5\ \mu\text{s}$ and `ibv_rc_bw` to ensure sustained single-rail throughput $\ge 47.5\text{ GB/s}$ ($380\text{ Gb/s}$).
4. **Layer 4 — NCCL Collectives:** Run `nccl-tests` across 512 GPUs. Confirm AllReduce bus bandwidth reaches $\ge 360\text{ GB/s}$ on NVSwitch and $\ge 45\text{ GB/s}$ over InfiniBand rails.
5. **Layer 5 — Model Step and Overlap:** Execute 100 steps of synthetic Llama 3 training. Use PyTorch Profiler to verify that gradient AllReduces overlap cleanly behind backward compute.
6. **Layer 6 — Workload SLO and FinOps:** Run a 48-hour continuous soak test. Validate tokens/second stability, MFU consistency, energy efficiency (tokens per joule), and cost amortization.

### 6.2. What evidence belongs in the decision table?

Procurement scorecards must demand concrete benchmark telemetry rather than vendor promises:
- A 1,024-GPU `all_reduce_perf` histogram documenting P-50, P-99, and minimum bus bandwidth.
- Continuous 24-hour soak run telemetry charting DCGM Xid counts and thermal clock stability.
- Storage throughput logs validating sustained multi-terabyte burst write speeds under 16,384 simulated worker threads.

### 6.3. The most common scope error

The most frequent budgetary disaster: **"Allocating 90% of budget to GPUs, leaving network and facility infrastructure underfunded."**
A healthy AI supercomputer budget adheres to this distribution:
- GPU Compute Servers: 55–60%
- High-Speed Network Fabric (Leaf, Spine, Optics, Cables): 20–25%
- Parallel Storage Infrastructure: 10–15%
- Facility, Power Distribution, and Liquid Cooling: 10%

Skimping on fabric to buy more GPUs is like mounting a Formula 1 engine onto a tractor chassis; the machine will not go faster, it will tear itself apart.

### 6.4. What makes an acceptance test fail?

A cluster acceptance test fails if any of these conditions occur:
1. **Flapping Link:** Any InfiniBand or RoCE link dropping link speed or emitting frame errors during test runs.
2. **Straggler Outliers:** Any node in an NCCL collective deviating by more than 5% from cluster mean (P-100 latency exceeding P-50 by $>10\%$).
3. **Thermal Throttling:** Any GPU reducing clock frequencies below nominal base rates during a 1-hour cuBLAS stress test (`SW Thermal Throttling`).

---

## 7. Testing what we learned

### 7.1. How much memory do 70B BF16 weights require?
70 billion parameters $\times 2\text{ bytes/parameter} = 140\times 10^9\text{ bytes} \approx \mathbf{140\text{ GB}}$ ($\approx 130.4\text{ GiB}$ in binary base).

### 7.2. Why do four 80 GB GPUs not imply “180 GB free for KV cache”?
Total HBM across 4 GPUs is $320\text{ GB}$. Subtracting $140\text{ GB}$ of weights leaves $180\text{ GB}$ on paper.
However:
- CUDA context and runtime overhead: $\approx 1.5 - 2\text{ GB}$ per GPU ($8\text{ GB}$ total).
- PyTorch / vLLM activation workspaces: $\approx 10 - 15\text{ GB}$.
- Memory fragmentation and safety headroom: $\approx 15 - 20\text{ GB}$.
Net usable KV cache is bounded to **$135 - 145\text{ GB}$**. Sizing for 180 GB will cause runtime out-of-memory crashes.

### 7.3. Why does full fine-tuning require much more HBM than serving?
Serving executes forward passes only, keeping model weights static ($2\text{ B/param}$) without backpropagation.
Full fine-tuning requires:
- Gradients ($2\text{ B/param}$),
- FP32 AdamW master weights, momentum, and variance ($12\text{ B/param}$),
- Intermediate activation tensors retained for backpropagation ($M_{\mathrm{act}}$).
Total state scales to $16\text{ bytes per parameter}$ plus activations—demanding **8 to 10 times more memory** than inference!

### 7.4. NCCL tests look good but training is slow: Is that a contradiction?
Not at all!
`nccl-tests` is an isolated synthetic microbenchmark streaming pre-warmed buffers from HBM.
In production training:
1. Dataloader workers reading disk may be stalled on filesystem metadata locks (Input Stalls).
2. Sub-optimal matrix dimensions may lead to poor Tensor Core tiling efficiency.
3. Compute and communication streams may fail to overlap cleanly.
A clean NCCL benchmark proves the fabric works; it does not prove the application runs efficiently.

---

### What questions can the reader now answer independently?

- Why must 70B serving, fine-tuning, and pre-training clusters be architected with fundamentally different network topologies?
- How do you calculate the impact of a DGX H100's $10.2\text{ kW}$ power draw on data center cooling and PUE planning?
- How do you apply the $16\text{ bytes per parameter}$ rule to calculate minimum GPU counts for distributed training?
- Which 6-tier validation ladder should be executed to certify cluster health prior to commercial sign-off?
- Why does expanding batch size purely to boost MFU risk harming FinOps and model convergence?

---

## The whole story in six lines

- The right AI hardware is selected not by chasing the fastest GPU, but by mapping tensor lifecycles across physical data paths.
- Pre-training is fabric-bound, full fine-tuning is memory-bound, and inference decode is HBM-bandwidth-bound.
- Sizing a 70B model requires 4 GPUs for serving, 16 GPUs for full fine-tuning, and hundreds of rail-optimized GPUs for pre-training.
- GPU utilization merely measures activity; genuine engineering ROI is measured in MFU and tokens per Joule.
- Densities of $40\text{ kW}$ per rack make liquid cooling an immutable physical requirement rather than an operational luxury.
- Sustainable platform procurement requires multi-tier validation from physical optics up to application SLOs.

---

## Glossary

- **Hardware Signature:** The characteristic demand profile an AI workload exerts across compute, memory, network, and storage.
- **Full Fine-Tuning:** Updating all model parameters with backpropagation and optimizer states, requiring $16\text{ B/param}$.
- **PEFT / LoRA:** Adapting models by freezing base weights and training small low-rank adapter matrices to save memory.
- **HBM Bandwidth:** The rate at which weights stream from memory into Tensor Cores ($3.35\text{ TB/s}$), bounding decode token speed.
- **MIG (Multi-Instance GPU):** Hardware partitioning of a GPU into isolated instances with dedicated compute and memory paths.
- **Rail Alignment:** Network architecture connecting matching NIC indices across nodes to dedicated leaf switches for single-hop RDMA.
- **DLC (Direct-to-Chip Liquid Cooling):** Circulating cooling fluid directly over server silicon to dissipate heat in high-density racks ($>40\text{ kW}$).
- **PUE (Power Usage Effectiveness):** The ratio of total data center power consumption to the power delivered to IT compute gear.
- **Validation Ladder:** A rigorous 6-tier acceptance testing methodology verifying hardware health from optics up to workload SLOs.

---

## References and verification links

- Previous article: [AI Infrastructure (4): Diagnosing Cluster Bottlenecks — Storage, Failures, and MFU](post.html?slug=from-gpus-to-ai-supercomputers)
- First article: [AI Infrastructure (1): GPU Architecture and LLM Scaling — SM, Tensor Cores, and the Memory Wall](post.html?slug=understanding-gpus-and-llm-scaling)
- **[K1] DGX H100 System Architecture:** NVIDIA, [*DGX H100 Architecture Whitepaper*](https://resources.nvidia.com/en-us-dgx-systems/dgx-h100-system-architecture).
- **[K2] Llama 3 Infrastructure Design:** Meta AI, [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783).
- **[K3] Data Center Power and Cooling:** Uptime Institute, [*Data Center High-Density Cooling and Power Trends*](https://uptimeinstitute.com).
- **[K4] vLLM and PagedAttention:** Woosuk Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180).
- **[K5] Megatron-LM and 3D Parallelism:** NVIDIA, [*Efficient Large-Scale Language Model Training on GPU Clusters*](https://arxiv.org/abs/2104.04473).
