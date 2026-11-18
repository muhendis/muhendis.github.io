A GPU datasheet might proudly boast hundreds of TFLOP/s or even PFLOP/s; yet your large language model might run at a fraction of that expected speed. This massive gap between paper peak compute and sustained production throughput is the defining reality of modern AI systems engineering: GPU performance is not governed merely by the arithmetic speed of execution units, but by how rapidly and uninterruptedly data moves across registers, shared memory, caches, HBM, NVLink, PCIe, and the scale-out cluster fabric.

Imagine standing in front of an eight-GPU NVIDIA H100 server inside a data centre. You run `nvidia-smi`; GPU utilization reads 100%. Yet when you benchmark token generation speed, you discover that the model extracts barely 2% of the hardware's theoretical compute capacity. The GPU appears busy; the fans scream at maximum RPM, and power supplies pull kilowatts of power. Where, then, is the compute being lost? The answer is not simply that "the GPU is idling"; rather, the execution pipelines inside the GPU silicon are burning clock cycles stalled, waiting hundreds of cycles for numbers to trickle in from memory.

**The core idea of this article:** Large-scale artificial intelligence is not merely a compute problem. It is fundamentally a problem of routing data to the right place, at the right time, across the right memory and interconnect hierarchy, with minimal latency. The most direct path to understanding a GPU and an AI supercomputer is to trace a tensor's physical journey from storage into the GPU, through the Streaming Multiprocessor (SM), and across to peer GPUs.

> **Our place in the series:** This article is the founding first part of our five-part *AI Infrastructure* series. Here, we establish the physical hardware map at the silicon level (SM microarchitecture, memory hierarchy, the Roofline model, and intra-node interconnects). Subsequent parts will apply this foundation to workloads (training/prefill/decode), cluster fabrics (InfiniBand/parallelism), diagnostics (MFU/storage walls), and enterprise platform design.

> **By the end of this article, you will be able to:**
>
> - Explain the hierarchical relationship between threads, warps, SMs, and Tensor Cores at the silicon level.
> - Distinguish memory capacity from memory bandwidth, and analyze the impact of coalesced memory access.
> - Apply the Roofline model to prove mathematically whether a workload is compute-bound or memory-bound.
> - Demonstrate why pre-training/prefill and autoregressive decode place the GPU into fundamentally opposing execution regimes.
> - Defend the division of responsibilities between PCIe Gen5 and NVLink 4, as well as NUMA topology inside an eight-GPU node, with concrete numerical arguments.

**In this article**

- [1. How does a GPU organize computation?](#1-how-does-a-gpu-organize-computation)
- [2. Why does the memory hierarchy determine performance?](#2-why-does-the-memory-hierarchy-determine-performance)
- [3. The Roofline model: What is the GPU actually waiting for?](#3-the-roofline-model-what-is-the-gpu-actually-waiting-for)
- [4. How are GPUs connected inside a single node?](#4-how-are-gpus-connected-inside-a-single-node)
- [5. From node to cluster: Our roadmap across the series](#5-from-node-to-cluster-our-roadmap-across-the-series)
- [6. Five essential system design questions](#6-five-essential-system-design-questions)
- [What questions can the reader now answer independently?](#what-questions-can-the-reader-now-answer-independently)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. How does a GPU organize computation?

At the core of modern AI infrastructure sits GPU silicon engineered to execute massive matrix operations in parallel. A CPU core is designed to maximize the single-threaded speed of sequential instruction streams that feature frequent branching, hard-to-predict memory access, and high sensitivity to latency. Consequently, the lion's share of CPU silicon is allocated to massive cache hierarchies (L2/L3), complex branch predictors, and out-of-order execution logic.

The GPU is built upon an entirely different philosophy: the overwhelming majority of silicon area is devoted to execution pipelines (ALUs and Tensor Cores). By executing thousands of concurrent, similar operations, the GPU maximizes total computational throughput. However, to harness this staggering compute density efficiently, threads must be organized within a strict hardware hierarchy.

> [!TIP]
> **Architecture Map: The Big Picture of an AI Supercomputer**
> To explore the full five-stage end-to-end architecture map (A: Hardware Hierarchy from SM to SuperPOD, B: End-to-End Data Path of a Training Step, C: Memory, Fabric, and I/O Walls, D: Bandwidth and Latency Ladder Across Interconnects, E: 16k-GPU LLaMA 3 405B Production System) in high resolution:
> [👉 **Click Here to Open the Full Architecture Poster in a New Tab (Interactive SVG) ↗**](/assets/diagrams/ai-supercomputer-poster-en.html)

The table below summarizes the key tiers of the big picture map and the hardware parameters explored across this article:

| Tier | Physical Scope | Key Architectural Metric | Coverage in This Article |
| :--- | :--- | :--- | :--- |
| **A · Hardware Hierarchy** | SM → H100 Die → DGX Node → SuperPOD | 990 TFLOP/s BF16 per GPU; 3.35 TB/s HBM3 | Section 1 and Section 4 |
| **B · Data Flow Pipeline** | HBM → SRAM → Registers → NVLink → Network | 3.35 TB/s (HBM) → 450 GB/s (NVLink) → 50 GB/s (Fabric) | Section 2 and Section 4 |
| **C · Hardware Boundaries** | Compute vs. Memory Transfer Balance | Critical Arithmetic Intensity $\alpha_{\text{crit}} \approx 295.4$ FLOP/byte | Section 3 |
| **D · Transfer Ladder** | Time required to transfer 140 GB weights | 42 ms in HBM vs. 2.2 s over PCIe vs. 2.8 s over Fabric | Section 3 |
| **E · Production Scale** | 16,384-GPU LLaMA 3 405B training system | 38–41% MFU, 2 TB/s storage stream, 466 failures in 54 days | Section 5 and Series Roadmap |

### 1.1. The thread → warp → SM → GPU hierarchy

To develop an intuitive mental model of GPU architecture, let us return to the restaurant kitchen analogy. A modern data centre GPU (such as an NVIDIA Hopper H100 or Blackwell B200) is not a single giant head chef; it is a brigade of 132 to 148 independent line cooks operating side by side in the same kitchen. Each of these line cooks is called a **Streaming Multiprocessor (SM)**.

Each line cook (SM) possesses their own prep counter (the Register File and Shared Memory) and specialized cutlery (Tensor Cores and CUDA Cores). Cooks do not handle tickets individually; orders arrive grouped into teams of 32 cooks. This 32-thread hardware unit is called a **warp**. All cooks pull raw ingredients from the kitchen's shared pantry: the **High-Bandwidth Memory (HBM)**.

The architecture diagram below details the silicon die layout of a modern GPU and the internal anatomy of an individual SM:

<svg viewBox="0 0 560 440" role="img" aria-label="NVIDIA GPU Streaming Multiprocessor architecture, 4 subpartitions, Tensor Core, Warp Scheduler, and memory hierarchy diagram." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sm-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="sm-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">GPU SM Architecture and Memory Hierarchy</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hopper (H100) &amp; Blackwell (B200)</text>
<rect x="16" y="48" width="140" height="240" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU Silicon Die</text>
<text x="86" y="80" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">H100: 132 SMs</text>
<text x="86" y="94" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">B200: 148 SMs</text>
<rect x="26" y="104" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="43" y="117" text-anchor="middle" style="fill:var(--c-accent);font-size:9px;font-family:var(--font-mono)">SM0</text>
<rect x="66" y="104" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM1</text>
<rect x="106" y="104" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM2</text>
<rect x="26" y="128" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="43" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM3</text>
<rect x="66" y="128" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM4</text>
<rect x="106" y="128" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM5</text>
<rect x="26" y="152" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="43" y="165" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM6</text>
<rect x="66" y="152" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="165" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM7</text>
<rect x="106" y="152" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="165" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM8</text>
<rect x="26" y="176" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="43" y="189" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM9</text>
<rect x="66" y="176" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="189" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM10</text>
<rect x="106" y="176" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="189" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM11</text>
<text x="86" y="218" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Independent SM Units</text>
<text x="86" y="234" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">H100: 990 TFLOP/s BF16</text>
<text x="86" y="250" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">B200: 2250 TFLOP/s BF16</text>
<text x="86" y="268" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px;font-weight:600">GPU: 132+ SMs (Flexible)</text>
<line x1="156" y1="112" x2="174" y2="112" marker-end="url(#sm-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="176" y="48" width="368" height="240" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Streaming Multiprocessor (SM) Anatomy</text>
<text x="360" y="80" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 Identical Subpartitions · 256 KiB Register File</text>
<rect x="184" y="88" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="270.0" y="103" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Subpartition 0</text>
<rect x="190" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="228.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="274" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="312.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="190" y="131" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="228.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="274" y="131" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="312.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Registers</text>
<rect x="364" y="88" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="450.0" y="103" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Subpartition 1</text>
<rect x="370" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="408.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="454" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="492.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="370" y="131" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="408.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="454" y="131" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="492.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Registers</text>
<rect x="184" y="158" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="270.0" y="173" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Subpartition 2</text>
<rect x="190" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="228.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="274" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="312.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="190" y="201" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="228.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="274" y="201" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="312.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Registers</text>
<rect x="364" y="158" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="450.0" y="173" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Subpartition 3</text>
<rect x="370" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="408.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="454" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="492.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="370" y="201" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="408.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="454" y="201" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="492.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Registers</text>
<rect x="184" y="230" width="172" height="48" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="270" y="248" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">SMEM / L1 Cache (256 KiB)</text>
<text x="270" y="266" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Fast shared on-chip SRAM</text>
<rect x="364" y="230" width="172" height="48" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="450" y="248" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">TMEM (256 KiB Blackwell)</text>
<text x="450" y="266" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Tensor Memory accumulator cache</text>
<line x1="270" y1="280" x2="270" y2="304" marker-end="url(#sm-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="450" y1="280" x2="450" y2="304" marker-end="url(#sm-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="306" width="528" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280" y="326" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Shared L2 Cache (~50 MB Hopper / 126 MB Blackwell)</text>
<text x="280" y="344" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Shared across all SMs · ~5.5 TB/s bidirectional bandwidth</text>
<line x1="180" y1="358" x2="180" y2="380" marker-end="url(#sm-arr-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="380" y1="358" x2="380" y2="380" marker-end="url(#sm-arr-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<rect x="16" y="382" width="528" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280" y="402" text-anchor="middle" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">High Bandwidth Memory (HBM3 / HBM3e)</text>
<text x="280" y="418" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80 GB @ 3.35 TB/s (H100) · 141 GB @ 4.8 TB/s (H200) · 192 GB @ 8.0 TB/s (B200)</text>
</svg>

*Figure 1.1: NVIDIA GPU Streaming Multiprocessor architecture. Left: silicon die layout featuring 132 SMs; right: internal anatomy of an individual SM with 4 subpartitions (SMSP), a 256 KiB register file, 4 Tensor Cores, and shared L1/SMEM memory.*

In physical hardware, every SM is divided into four identical quadrants called **SM subpartitions (SMSP 0–3)**:
1. **Four Tensor Cores (TC):** One dedicated hardware matrix multiply-accumulate engine per subpartition. Running at ~1.83 GHz under Tensor Core workload on an H100 SXM5, each Tensor Core executes 1,024 dense BF16 floating-point operations (FLOPs) per cycle ($8 \times 8 \times 8$ matrix multiply). Under FP8, this doubles to 2,048 FLOPs per cycle. On the Blackwell B200, each Tensor Core delivers 2,048 BF16 FLOPs per cycle, reaching 2,250 TFLOP/s per die.
2. **Four Warp Schedulers and Dispatch Units:** Each scheduler manages up to 16 warps assigned to its subpartition, issuing one instruction per clock cycle from a ready warp. Across the four subpartitions, a single SM can host up to **64 resident warps** ($64 \times 32 = 2,048$ threads) concurrently.
3. **128 FP32 CUDA Cores:** Each subpartition contains 32 FP32 ALUs, 16 INT32, and 16 FP64 units. These general-purpose pipelines execute activation functions (GELU, SiLU), normalizations (RMSNorm), and element-wise tensor additions. An H100 with 132 SMs packs 16,896 FP32 CUDA cores.
4. **256 KiB Register File:** 64 KiB per subpartition (256 KiB per SM, totaling ~33 MB across the die) of single-cycle register storage. Thread variables live here, and all execution units read their operands directly from this file.

The four subpartitions share a unified 256 KiB SRAM block at the base of the SM, split between L1 data cache and **Shared Memory (SMEM)**. Up to 228 KiB can be configured as programmer-managed SMEM. The **Tensor Memory Accelerator (TMA)** acts as a hardware copy engine, transferring tensor tiles asynchronously between L2/HBM and SMEM via direct memory access (DMA) without stalling SM compute cores.

Let us derive the headline peak compute formula for an H100 GPU:

$$C_{\text{peak}} = N_{\text{SM}} \times N_{\text{TC/SM}} \times \text{FLOP/cycle} \times f$$

Where $N_{\text{SM}}$ is the active SM count (132), $N_{\text{TC/SM}}$ is Tensor Cores per SM (4), $\text{FLOP/cycle}$ is operations per Tensor Core per cycle (1,024 for dense BF16), and $f$ is the clock speed under Tensor Core load ($1.83 \times 10^9 \text{ Hz}$):

```text
528 TC × 1,024 FLOP × 1.83 GHz = 989,489,152,000 FLOP/s ≈ 989.5 TFLOP/s ≈ 990 TFLOP/s
```

This arithmetic provides the silicon-level origin of the published "990 TFLOP/s BF16 non-sparse" vendor specification.

### 1.2. Why is a Tensor Core different? The silicon logic of matrix engines

Conventional CPU ALUs and GPU CUDA cores operate on scalar instructions: they multiply two numbers and add a third in a single cycle ($a \times b + c$, Multiply-Accumulate / MAC). When a 32-thread warp executes this, 32 scalar values are produced.

A Tensor Core is not a scalar pipeline; it is a specialized, hardware-level systolic array. With a single instruction ($mma.sync$ or Hopper's asynchronous $wgmma$), it multiplies and accumulates entire matrix tiles:

$$D = A \times B + C$$

Where $A$ ($M \times K$), $B$ ($K \times N$), and $C, D$ ($M \times N$) are tensor sub-matrices. On fourth-generation Hopper Tensor Cores, the fundamental hardware micro-step computes an $8 \times 8 \times 8$ matrix multiplication. An $8 \times 8 \times 8$ multiply requires $8 \times 8 \times 8 = 512$ multiplications and 512 additions—generating **1,024 FLOPs in a single instruction**!

The second pillar of Tensor Core throughput is reduced precision. As bit-width decreases, the silicon area required for digital multiplier trees shrinks quadratically. More arithmetic units can be packed into the same die space:

| Numeric Format | Total Bits | Sign | Exponent | Mantissa | Dynamic Range | Role in LLM Systems | H100 SXM Peak Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 1 | 8 | 23 | $\pm 10^{-38} \dots 10^{38}$ | Master weights, optimizer states, loss | 60 TFLOP/s (Vector) |
| **BF16** | 16 | 1 | 8 | 7 | $\pm 10^{-38} \dots 10^{38}$ | Foundation model pre-training | 990 TFLOP/s (TC) |
| **FP16** | 16 | 1 | 5 | 10 | $\pm 6 \times 10^{-5} \dots 65,504$ | Traditional inference and fine-tuning | 990 TFLOP/s (TC) |
| **FP8 (E4M3)** | 8 | 1 | 4 | 3 | $\pm 448$ | Forward pass activations and weights | 1,980 TFLOP/s (TC) |
| **FP8 (E5M2)** | 8 | 1 | 5 | 2 | $\pm 57,344$ | Backward pass gradient propagation | 1,980 TFLOP/s (TC) |
| **NVFP4 (E2M1)**| 4 | 1 | 2 | 1 | Block-scaled | Blackwell ultra-dense GEMM inference | 4,500 TFLOP/s (B200) |

Hopper introduces two distinct FP8 representations: **E4M3** allocates 3 mantissa bits for higher numerical precision, tailored for forward pass weights and activations where dynamic range is well-bounded. **E5M2** retains the 5-bit exponent of FP16, providing the wide dynamic range essential for backward pass gradients prone to underflow.

### 1.3. Latency hiding and Little's Law in hardware

A GPU's defining strength is not that it eliminates latency using massive caches like a CPU, but that it **hides latency** by rapidly switching between concurrent warps.

Consider physical hardware memory latencies:
- Register access: $\sim 1$ cycle ($<0.5 \text{ ns}$)
- Shared memory (SMEM): $\sim 29$ cycles ($\sim 15 \text{ ns}$)
- L1 data cache: $\sim 41$ cycles ($\sim 22 \text{ ns}$)
- L2 cache: $\sim 263$ cycles ($\sim 140 \text{ ns}$)
- HBM3 global memory: $\sim 479$ cycles ($\sim 260 \text{ ns}$)

When a warp issues an instruction to load a tensor from HBM, waiting for that data takes roughly 480 clock cycles. If the SM had only that single warp, its execution units would sit stalled 99% of the time.

The Warp Scheduler circumvents this stall: the instant a warp stalls waiting on memory, the scheduler deschedules it and issues an arithmetic instruction from another ready warp on the very next cycle. This context switch incurs **zero overhead** because all register state is physically allocated in hardware registers (avoiding the CPU's register-dumping context switch penalty).

The minimum warp count needed to keep pipelines saturated is governed by **Little's Law**:

$$W_{\text{required}} \ge 1 + \frac{L_{\text{latency}}}{T_{\text{math}}}$$

Where $L_{\text{latency}}$ is memory stall latency in cycles, and $T_{\text{math}}$ is instruction execution duration.

Consider a simple numerical step: if an instruction computes for 2 cycles and waits 6 cycles on a cache line:

```text
Warp 0: [Compute: 2 cycles] -> [Stall: 6 cycles...................]
Warp 1:                     -> [Compute: 2 cycles] -> [Stall: 6 cycles]
Warp 2:                                             -> [Compute: 2 cycles]
Warp 3:                                                                 -> [Compute: 2 cycles]
Total warps required = 1 + (6 / 2) = 4 warps.
```

If the SM only hosted 2 warps, execution units would idle half the time. With 4 warps, the pipeline achieves 100% saturation. To hide a 480-cycle HBM stall, an SM must maintain dozens of concurrent warps in flight. The SM's 64-warp capacity is built specifically to satisfy this latency-hiding imperative.

> [!NOTE]
> **Learning Check:** *"Does launching more threads always increase performance?"* No. Spawning excessive threads dilutes per-thread register and shared memory allocations. If hardware limits are breached, occupancy collapses or register spilling begins. The engineering goal is not maximizing thread count blindly, but reaching the critical occupancy threshold needed to mask memory stalls.

### 1.4. Warp divergence, occupancy, and register pressure

The GPU execution model is **SIMT (Single Instruction, Multiple Threads)**. All 32 threads in a warp execute the same instruction in lockstep, each operating on its own private data registers.

This architecture creates a severe penalty when code encounters branch divergence:

```cuda
if (threadIdx.x < 16) {
    path_A(); // 16 threads
} else {
    path_B(); // 16 threads
}
```

Because execution pipelines cannot execute two distinct program counters simultaneously within one warp, the hardware serializes the branches:

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

*Figure 1.2: SIMT execution and Warp Divergence. Top: a 128-thread block divided into 4 warps; bottom: conditional divergence serializes execution paths, masking idle lanes and cutting arithmetic throughput in half.*

Hardware constructs an **execution mask**: threads 0–15 are activated while lanes 16–31 are disabled, and `path_A()` executes. Then the mask inverts: lanes 0–15 stall while lanes 16–31 execute `path_B()`. Wall-clock execution time doubles, cutting sustained arithmetic throughput for that warp by **50%**.

The second fundamental trade-off is **register pressure** versus **occupancy**. An SM contains 65,536 32-bit registers (256 KiB). To host the theoretical limit of 64 resident warps (2,048 threads), the register budget per thread is strictly capped:

$$\text{Max Registers/Thread} = \frac{65,536 \text{ registers}}{2,048 \text{ threads}} = 32 \text{ registers/thread}$$

As kernels grow complex and compilers unroll loops, register demand escalates. As thread register consumption rises, the resident warp capacity of the SM declines:

| Registers per Thread | Active Warps per SM | Theoretical Occupancy | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **$\le 32$ registers** | 64 warps (2,048 threads) | **100%** | Optimal latency hiding capability |
| **64 registers** | 32 warps (1,024 threads) | **50%** | Latency hiding margin narrows |
| **128 registers** | 16 warps (512 threads) | **25%** | Vulnerable to memory stall pipeline bubbles |
| **255 registers (hardware cap)** | 8 warps (256 threads) | **12.5%** | Critical threshold; risk of pipeline starvation |

If a kernel exceeds the 255-register limit, **register spilling** occurs: excess variables spill into **local memory**. Do not be misled by the word "local": local memory physically resides in high-latency HBM! Register spilling inflicts devastating performance penalties.

> [!NOTE]
> **Thought Experiment: Is Low Occupancy Always Bad?**
> Traditional CUDA optimization treated 100% occupancy as mandatory. Modern Hopper kernels such as **FlashAttention-3** upended this rule by achieving peak performance with only 15–25% occupancy (8–16 warps). The mechanism: **Warp Specialization**. One or two producer warps continuously issue asynchronous TMA transfers, while remaining consumer warps execute Tensor Core $wgmma$ instructions without interruption. Because data arrives in SMEM ahead of time, compute warps never stall. If there are no stalls, you do not need dozens of resident warps to hide latency!

---

## 2. Why does the memory hierarchy determine performance?

A common pitfall when reasoning about large language models is viewing memory purely in terms of gigabytes. In production systems, performance is dictated not by where data can fit, but by how fast that data can be read into execution units.

### 2.1. Capacity and bandwidth are fundamentally different

Consider a library: the shelves may store 10 million volumes (immense capacity). But if the library aisles admit only one visitor at a time and the checkout counter has a single staff member (low bandwidth), the library can serve only 5 books per minute.

An H100 SXM5 GPU features 80 GB of HBM3. A 70-billion-parameter model in BF16 requires 140 GB; it cannot fit onto a single GPU. Yet even if quantized to 4-bit (35 GB) to fit easily into 80 GB, the model does not automatically generate tokens at lightning speed. To generate each single token, all 35 GB of weights must still be transferred from HBM into the Tensor Cores!

The GPU memory hierarchy is engineered as a steep pyramid to balance capacity against bandwidth:

| Memory Tier | Physical Location | H100 SXM Capacity | Bandwidth | Typical Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Register File** | Inside SM core | 256 KiB / SM (~33 MB die-wide) | **$\sim 30+$ TB/s** (aggregate) | $\sim 1$ cycle ($<0.5 \text{ ns}$) |
| **Shared Memory / L1**| Inside SM core | 256 KiB / SM (~33 MB die-wide) | **$\sim 15+$ TB/s** (aggregate) | $\sim 29$ cycles ($\sim 15 \text{ ns}$) |
| **L2 Cache** | Die centre | 50 MB (two partitions) | **$\sim 5.5$ TB/s** | $\sim 263$ cycles ($\sim 140 \text{ ns}$) |
| **HBM3 Memory** | On-package (interposer)| 80 GB (5 active stacks) | **3.35 TB/s** | $\sim 479$ cycles ($\sim 260 \text{ ns}$) |
| **Host RAM (DDR5)** | Motherboard sockets | 2 TB per node | 64 GB/s (PCIe Gen5 x16) | $\sim 1,000+$ cycles ($>500 \text{ ns}$)|

As data moves down from the peak to the base of the pyramid, capacity expands 2,400-fold (from 33 MB to 80 GB); however, bandwidth drops by 10x and access latency increases 500-fold. Achieving high MFU requires keeping active working sets at the top of the pyramid.

### 2.2. Coalesced access and the math of cache lines

HBM memory controllers do not fetch data byte by byte; transactions occur in 32-byte or 128-byte **cache lines**.

When a 32-thread warp reads 4-byte floating-point values from global memory, two scenarios unfold:

**Scenario A: Coalesced Access:**
All 32 threads request contiguous memory addresses ($A[0], A[1], \dots, A[31]$).
- Total useful data requested: $32 \times 4 = 128 \text{ bytes}$.
- Memory controllers service this in a **single 128-byte bus transaction**.
- Bus efficiency: $128 / 128 = \mathbf{100\%}$!

**Scenario B: Strided (Non-Coalesced) Access:**
When traversing matrix columns with a stride of 32 ($A[0], A[32], A[64], \dots$):
- Each thread lands in a separate 128-byte cache line.
- The memory controller must issue 32 independent 128-byte memory transactions.
- Total bytes transferred across the bus: $32 \times 128 = 4,096 \text{ bytes}$.
- Useful data consumed: 128 bytes.
- Bus efficiency: $128 / 4,096 = \mathbf{3.1\%}$!

Over 96% of memory bus bandwidth is squandered transporting unused bytes. A kernel with uncoalesced memory access will saturate the HBM bus regardless of Tensor Core compute power.

### 2.3. Tiling and shared memory: Why attention requires SRAM

Consider standard matrix multiplication ($C = A \times B$, each $N \times N$). A naive three-nested loop reads row and column vectors repeatedly from global memory, resulting in $O(N^3)$ global memory transfers. For $N=4,096$, this demands trillions of byte transfers, instantly bottlenecking HBM.

**Tiling** resolves this: matrices are partitioned into small sub-blocks of size $B_k \times B_k$. A block is loaded **once** from HBM into the SM's ultra-fast shared memory (SMEM / SRAM). The Tensor Cores multiply and accumulate these tiles in SRAM, reducing HBM memory traffic by a factor of $B_k$.

The most famous application of this principle in modern LLM architecture is **FlashAttention**. Standard self-attention materializes an $N \times N$ attention score matrix in HBM ($O(N^2)$ memory traffic). FlashAttention applies tiling combined with online softmax scaling to compute attention entirely within SMEM/SRAM without ever writing intermediate $N \times N$ matrices to HBM, cutting memory I/O from $O(N^2)$ to $O(N)$ and delivering 3–5x end-to-end speedups.

---

## 3. The Roofline model: What is the GPU actually waiting for?

The definitive analytical tool for diagnosing whether a model's execution speed is bottlenecked by arithmetic throughput or memory bandwidth is the **Roofline Model** introduced by **Williams et al. (2009)**.

### 3.1. Critical arithmetic intensity and the ridge point

The Roofline model formulates attainable performance ($P$, in FLOP/s) as a function of arithmetic intensity ($I$, in FLOP/byte):

$$P = \min(P_{\text{peak}}, \, I \times B_{\text{mem}})$$

Where $P_{\text{peak}}$ is theoretical peak compute (FLOP/s), $B_{\text{mem}}$ is peak memory bandwidth (bytes/s), and $I$ is operational intensity:

$$I = \frac{\text{Total FLOPs Executed}}{\text{Total Bytes Transferred from/to HBM}}$$

The inflection point separating the memory-bound regime from the compute-bound regime is the **critical arithmetic intensity (ridge point / knee point, $\alpha_{\text{crit}}$)**:

$$\alpha_{\text{crit}} = \frac{P_{\text{peak}}}{B_{\text{mem}}}$$

For an NVIDIA H100 SXM5 running BF16 Tensor Core arithmetic:

$$\alpha_{\text{crit}} = \frac{989.5 \times 10^{12} \text{ FLOP/s}}{3.35 \times 10^{12} \text{ B/s} \approx 3,350 \times 10^9 \text{ B/s}} \approx \mathbf{295.4 \text{ FLOP/byte}}$$

This number is the central compass of AI systems engineering:
- If your workload's arithmetic intensity is **$I < 295.4$ FLOP/byte**: the system operates in the **Memory-Bound** regime. Tensor Cores are starved; performance scales strictly with HBM bandwidth.
- If your workload's arithmetic intensity is **$I \ge 295.4$ FLOP/byte**: the system operates in the **Compute-Bound** regime. Memory bandwidth is sufficient to saturate arithmetic pipelines; performance approaches theoretical peak FLOP/s.

### 3.2. Why training/prefill and decode live in opposite regimes

Consider matrix multiplication ($X \times W$) where $X \in \mathbb{R}^{B \times D}$ is the activation matrix and $W \in \mathbb{R}^{D \times F}$ is the weight matrix (16-bit format, 2 bytes per parameter):
- **Total FLOPs:** $2 B D F$
- **Total Bytes Transferred:** $2(BD + DF + BF) \text{ bytes}$

Operational intensity is expressed as:

$$I(B) = \frac{2 B D F}{2(BD + DF + BF)} = \frac{B D F}{B(D + F) + D F}$$

Applying the dimensions of the LLaMA 3 70B MLP layer ($D = 8,192, F = 28,672$):
$D F = 234,881,024$ parameters ($469.8 \text{ MB}$).

**Case 1: Pre-training and Prefill ($B = 512$ tokens):**
Under large batch sizes and extensive prompt processing:
- Weight matrix: $469.8 \text{ MB}$
- Activations: $2 \times 512 \times (8,192 + 28,672) = 37.7 \text{ MB}$
- Arithmetic intensity:

$$I(512) = \frac{512 \times 8,192 \times 28,672}{512 \times 36,864 + 234,881,024} = \frac{120,259,084,288}{253,755,392} \approx \mathbf{473.9 \text{ FLOP/byte}}$$

Because $473.9 > 295.4$, execution is firmly **Compute-Bound**! Tensor Cores saturate, and sustained performance easily reaches 600–800 TFLOP/s (high MFU).

**Case 2: Autoregressive Decode ($B = 1$ token):**
During token-by-token generation:
- With $B=1$, weight matrix volume dominates memory traffic ($D F \gg B(D+F)$).
- Arithmetic intensity collapses to:

$$I(1) \approx \frac{1 \times D F}{D F} = \mathbf{1 \text{ FLOP/byte}}$$

Against a critical threshold of 295.4, the workload provides only 1 FLOP/byte! What throughput can the H100 deliver here?

$$P_{\text{decode}} = 1 \text{ FLOP/byte} \times 3.35 \text{ TB/s} = \mathbf{3.35 \text{ TFLOP/s}}$$

A 990 TFLOP/s supercomputing chip delivers barely **3.35 TFLOP/s**—a dismal **0.34% MFU**! Tensor Cores spend 99.66% of their cycles idle, waiting for weights to stream across the HBM bus.

This mathematical reality explains why inference serving systems (such as vLLM) rely on continuous batching, PagedAttention, and Speculative Decoding to increase batch sizes and amortize weight-loading costs across multiple user requests.

### 3.3. The 140 GB transfer ladder: Physical interconnect boundaries

When routing model weights or KV cache tensors across an infrastructure, understanding physical transfer latencies across hardware tiers is indispensable. Consider the duration required to transfer the 140 GB BF16 weights of a 70B parameter model:

| Interconnect / Fabric | Bandwidth (Unidirectional) | 140 GB Transfer Duration | Physical Reality |
| :--- | :--- | :--- | :--- |
| **HBM3 (On-Die)** | 3,350 GB/s (3.35 TB/s) | **41.8 milliseconds** | On-interposer silicon traces; baseline speed ceiling |
| **NVLink 4 (Intra-Chassis)**| 450 GB/s (per GPU) | **311 milliseconds** | 8-GPU DGX chassis non-blocking crossbar |
| **PCIe Gen5 x16** | 32 GB/s (uni) / 64 GB/s | **2.19 – 4.38 seconds** | Host CPU, NVMe, and generic adapter bus |
| **InfiniBand NDR 400G** | 50 GB/s (per GPU) | **2.80 seconds** | Scale-out inter-node cluster fabric |
| **100 GbE Enterprise Net** | 12.5 GB/s | **11.2 seconds** | Standard data centre management and egress fabric |
| **10 GbE Legacy Link** | 1.25 GB/s | **112 seconds (~1.9 min)**| Shared legacy network or remote NFS storage |

This hierarchy establishes an iron law of distributed systems:
Extracting a tensor from HBM is **7x slower**, and pushing it out of the server chassis across the network is **70x slower**!

### 3.4. A three-step Roofline recipe for systems engineers

To optimize any deep learning kernel or serving pipeline, follow this 3-step diagnostic loop:
1. **Measure Arithmetic Intensity:** Run NVIDIA Nsight Compute (`ncu`) to capture `dram__bytes.sum` and `sm__ops_path_tensor_src_fp16.sum`. Divide them to compute $I$.
2. **Compare Against $\alpha_{\text{crit}}$:** If $I < 295$, your workload is memory-bound; if $I \ge 295$, it is compute-bound.
3. **Select the Right Optimization Lever:**
   - *If Memory-Bound:* Buying faster compute will not help! Focus on kernel fusion (eliminating intermediate HBM round-trips), weight quantization (FP8/INT4 to reduce memory footprint), and increasing batch size.
   - *If Compute-Bound:* Focus on lower-precision Tensor Core intrinsics ($wgmma$, FP8), tensor layout alignments, and overlapping communication with compute.

---

## 4. How are GPUs connected inside a single node?

A single H100 die delivers 80 GB of memory and 990 TFLOP/s. However, frontier foundation models require hundreds of gigabytes, or even terabytes, of memory. The industry-standard foundational building block is therefore an 8-GPU server chassis (such as NVIDIA DGX H100 or HGX H100).

### 4.1. Dividing responsibilities: PCIe Gen5 vs. NVLink 4

Inside an 8-GPU server, two distinct interconnect fabrics coexist with strictly divided roles:

**PCIe Gen5 (General-Purpose I/O):**
- Connects CPUs, system host RAM, NVMe SSDs, and Network Interface Cards (NICs).
- A PCIe Gen5 x16 slot provides 64 GB/s full duplex (32 GB/s unidirectional) theoretical bandwidth.
- Highly versatile, but far too slow for frequent intra-layer tensor synchronizations.

**NVLink 4 (Dedicated GPU-to-GPU Fabric):**
- A proprietary, hardware cache-coherent mesh designed exclusively for inter-GPU communication.
- Each H100 GPU features 18 NVLink 4 connections. Each link delivers 25 GB/s unidirectional (50 GB/s bidirectional).
- Aggregate bandwidth per GPU: $18 \times 25 = \mathbf{450 \text{ GB/s}}$ unidirectional (**900 GB/s** bidirectional).
- Exactly **14x faster than PCIe Gen5**!

### 4.2. DGX H100 NUMA topology and GPU–NIC affinity

A DGX H100 node houses two Intel Xeon Platinum CPUs (CPU 0 and CPU 1), linked by an inter-socket **UPI (Ultra Path Interconnect, ~32–40 GB/s)** bridge. This forms a dual-socket **NUMA (Non-Uniform Memory Access)** topology:

```text
[ CPU 0 ] <====== UPI (~36 GB/s) ======> [ CPU 1 ]
    |                                        |
  PCIe Switch                              PCIe Switch
  +-- GPU 0, 1, 2, 3                       +-- GPU 4, 5, 6, 7
  +-- NIC 0, 1, 2, 3 (Compute)             +-- NIC 4, 5, 6, 7 (Compute)
  +-- Storage NIC (GDS)                    +-- Storage NIC (GDS)
  +-- 4x NVMe SSD                          +-- 4x NVMe SSD
```

The architectural consequence is stark:
- GPU 0 communicates with its local NIC 0 directly across the local PCIe switch at full 50 GB/s wire speed.
- If misconfigured such that GPU 0 attempts to transmit via NIC 4 on CPU 1, traffic must cross the narrow UPI socket link. Bandwidth collapses to the UPI bottleneck, and latency surges. Proper CPU/GPU process affinity pinning is therefore mandatory.

### 4.3. GPUDirect RDMA and the elimination of bounce buffers

In traditional Linux networking stacks, inbound network data took a tortuous path:
1. Packet arrives at the NIC.
2. NIC DMAs packet into host OS kernel memory (page cache).
3. Application copies buffer to user-space memory (bounce buffer).
4. CUDA driver transfers data over PCIe into GPU HBM.

This double-copy chain saturates CPU cores and introduces dozens of microseconds of latency.

**GPUDirect RDMA (Remote Direct Memory Access)** bypasses host memory entirely. The NIC initiates direct DMA transfers over the PCIe switch into or out of GPU HBM. The host CPU and RAM are completely bypassed, slashing network latency from tens of microseconds to **1–2 microseconds**.

### 4.4. The intra-node NVSwitch crossbar architecture

Directly point-to-point cabling 8 GPUs ($8 \times 7 / 2 = 28$ physical links) would create severe trace routing congestion. NVIDIA solves this by embedding 4 dedicated **NVSwitch** ASICs into the baseboard.

Each GPU's 18 NVLink connections are distributed across these 4 NVSwitches ($5 + 4 + 4 + 5 = 18$). The NVSwitches function as a non-blocking crossbar:
- Any GPU can read or write to any of the other 7 GPUs' memory spaces at full 450 GB/s unidirectional speed.
- The total intra-node bisection bandwidth reaches **3.6 TB/s** ($4 \text{ GPUs} \times 450 \text{ GB/s} \times 2$).

This 3.6 TB/s fabric enables **Tensor Parallelism (TP=8)** to execute within the chassis with near-zero communication overhead.

---

## 5. From node to cluster: Our roadmap across the series

An 8-GPU DGX node is a computational powerhouse: 640 GB of HBM3 memory and ~7.9 PFLOP/s of BF16 compute. Yet its boundaries are quickly reached.

### 5.1. Single-node boundaries and the scaling imperative

Consider training a frontier foundation model: **LLaMA 3 405B**.
- Model parameters (BF16): 810 GB.
- Gradients (BF16): 810 GB.
- AdamW Optimizer states (FP32 master weights, momentum, variance): $405 \times 12 = 4,860 \text{ GB}$.
- Total minimum pre-training memory state: **$\approx 6,480 \text{ GB}$ (6.48 Terabytes)**!

The 640 GB memory of an entire 8-GPU node holds barely **one-tenth** of this training state. Training on a single server is physically impossible; we must scale out across thousands of GPUs.

Yet crossing the server chassis changes the physical rules:
- Bandwidth drops from NVLink's 450 GB/s to InfiniBand's **50 GB/s** (a 9x drop).
- Latency increases from sub-microsecond crossbar hops to **multiple microseconds** over multi-tier optical fabrics.

This physical cliff gives rise to the foundational principle of distributed AI design:
> **"Tensors that communicate frequently must stay inside the chassis (on NVLink); tensors that communicate infrequently may cross the network (over InfiniBand)."**

### 5.2. Bridging to the next parts: From workloads to enterprise platform

The hardware anatomy established in this article forms the bedrock for our five-part journey:

1. **AI Infrastructure (Part 2): Inside LLM Workloads — Training, Prefill, and Decode**  
   Why the same GPU operates compute-bound during prefill but memory-bandwidth bound during decode; KV cache dynamics, PagedAttention, prefix caching, and the inner mechanics of vLLM serving.  
   *[Read Part 2](post.html?slug=gpu-workloads-and-production-inference)*

2. **AI Infrastructure (Part 3): Multi-GPU Fabrics and Distributed Training — NVLink, InfiniBand, and 4D Parallelism**  
   Inter-node network fabrics (Quantum-2 InfiniBand, rail-optimized fat-tree topologies, SHARP in-network reduction) and 4D model parallel partitioning (TP, PP, DP/FSDP, CP, EP, and DeepSeek DualPipe).  
   *[Read Part 3](post.html?slug=multi-gpu-networks-and-distributed-training)*

3. **AI Infrastructure (Part 4): Diagnosing AI Supercomputers — Storage, Failures, and MFU**  
   Why a 16,384-GPU supercomputer stalls at 38% MFU; GPUDirect Storage (GDS), checkpoint pauses, Young-Daly fault tolerance mathematics, and cluster telemetry.  
   *[Read Part 4](post.html?slug=from-gpus-to-ai-supercomputers)*

4. **AI Infrastructure (Part 5): Enterprise AI Platform Architecture — Power, FinOps, and Orchestration**  
   Megawatt-scale power budgets, liquid cooling architectures, Slurm/Kubernetes scheduling, GPU cost modeling (FinOps), and an end-to-end 70B reference design.  
   *[Read Part 5](post.html?slug=enterprise-ai-platform-architecture)*

---

## 6. Five essential system design questions

### 6.1. Why is a GPU a thousand times faster at GEMM but terrible at scalar CPU tasks?

A CPU features 8–64 sophisticated cores clocked at 4–5 GHz with large branch predictors, deep out-of-order execution windows, and massive caches, optimized to execute complex scalar code with minimal single-thread latency. However, its peak arithmetic operations per clock cycle are fundamentally bounded.

A GPU operates at lower clock frequencies (~1.8 GHz) but deploys thousands of parallel arithmetic units (16,896 CUDA cores and 528 Tensor Cores). Matrix multiplication (GEMM) is embarrassingly parallel. Tensor Cores execute hundreds of multiply-accumulate operations in a single cycle using systolic array structures, delivering arithmetic densities that CPU silicon cannot physically match.

### 6.2. Does 100% GPU utilization in nvidia-smi imply optimal model throughput?

No. The "GPU Utilization" metric reported by `nvidia-smi` only reflects the percentage of time that one or more SMs had at least one active thread scheduled. If an SM spends 480 clock cycles stalled waiting for weights to stream from HBM (memory stall), utilization still reports 100%!

The true performance metric is **Model FLOPs Utilization (MFU)**, which calculates the ratio of theoretical peak FLOPs against the useful floating-point operations executed by the model. During autoregressive decode, GPU utilization may read 100% while MFU hovers below 1%.

### 6.3. Why is NVLink not used everywhere instead of PCIe and Ethernet?

NVLink operates at extremely high frequencies with dense trace routing, suffering rapid signal attenuation over distance. On PCB substrates, it can only traverse distances of 20–30 centimetres.

A standard PCIe bus is necessary to maintain modularity, backward compatibility, and interoperability across third-party CPUs, SSDs, and network adapters. For rack-to-rack distances (tens of meters), serialized scale-out protocols such as InfiniBand or Ethernet over optical or active copper cables are physically required.

### 6.4. If weights fit entirely in HBM, why can execution still hit a memory wall?

Fitting model weights into HBM resolves the **capacity** constraint, but does not eliminate the **bandwidth** bottleneck.

During autoregressive single-token decode ($B=1$), every model parameter must be fetched from HBM into registers to compute just 2 FLOPs per parameter (arithmetic intensity $I = 1 \text{ FLOP/byte}$). Because HBM bandwidth (3.35 TB/s) bounds data streaming speed, Tensor Cores spend over 99% of their cycles idle. Even if the model fits entirely in memory, execution remains bottlenecked by the memory bandwidth wall.

### 6.5. Why does increasing threads per block sometimes degrade kernel execution speed?

Increasing thread count per block (e.g., from 128 to 1,024) multiplies the block's aggregate demand for registers and shared memory. The SM has hard limits of 256 KiB register file and 228 KiB SMEM.

If a single thread block consumes the majority of SM resources, the SM can only schedule one block at a time, lowering resident warps and hurting latency-hiding occupancy. Worse, if register constraints are exceeded, the compiler forces register spilling to high-latency HBM, causing catastrophic slowdowns.

---

### What questions can the reader now answer independently?

1. What physical memory tiers does a tensor traverse on its path from HBM into an SM, and how does latency scale at each boundary?
2. How do you mathematically prove why an H100 GPU with 990 TFLOP/s peak capacity drops to 3.35 TFLOP/s during single-token decoding?
3. How does the hardware execution mask operate during warp divergence, and why does execution time double?
4. Why is Tensor Parallelism (TP) optimal within an 8-GPU DGX chassis, but disastrous across server nodes?
5. When developing high-performance CUDA kernels, why should you prioritize warp specialization over theoretical occupancy?

---

## The whole story in six lines

- A GPU's theoretical FLOP/s specification is meaningless without sufficient memory and interconnect bandwidth to feed its execution pipelines.
- The Streaming Multiprocessor (SM) hides hundred-cycle HBM latencies by executing zero-overhead context switches across up to 64 resident warps.
- Tensor Cores are hardware systolic matrix engines; migrating from FP32 to FP8 and NVFP4 doubles and quadruples arithmetic density per unit of silicon.
- The critical arithmetic intensity ridge point for an H100 is 295.4 FLOP/byte; prefill and training live above it (compute-bound), while decode lives far below it (memory-bound).
- Inside an 8-GPU node, the NVSwitch crossbar delivers 3.6 TB/s bisection bandwidth—14x faster than PCIe—acting as a single virtual GPU.
- Moving 140 GB of model weights takes 42 milliseconds within HBM, 2.2 seconds over PCIe, and 2.8 seconds across the network; the slowest link dictates system performance.

---

## Glossary

- **Streaming Multiprocessor (SM):** The fundamental independent compute unit on a GPU (132 on H100, 148 on B200), housing schedulers, registers, Tensor Cores, and shared memory.
- **Warp:** A group of 32 threads scheduled and executed in lockstep by hardware using a single SIMT instruction.
- **Warp Divergence:** The performance penalty incurred when threads in a warp take differing branches, forcing hardware to serialize execution via execution masks.
- **Latency Hiding:** The mechanism whereby the warp scheduler switches to another ready warp when one stalls on memory, keeping arithmetic pipelines saturated with zero overhead.
- **Occupancy:** The ratio of active resident warps on an SM to the hardware's theoretical maximum resident warp capacity (64 on H100).
- **Roofline Model:** An analytical performance model charting attainable throughput against operational intensity (FLOP/byte) to identify memory vs. compute bounds.
- **Critical Arithmetic Intensity ($\alpha_{\text{crit}}$):** The inflection threshold (peak FLOP/s divided by peak memory bandwidth) above which a kernel becomes compute-bound (~295.4 FLOP/byte for H100 BF16).
- **GPUDirect RDMA:** Direct memory access technology enabling NICs to read and write directly to GPU HBM over PCIe, bypassing host CPU and system RAM.
- **NVSwitch:** An on-node crossbar switching ASIC that connects multiple GPUs via NVLink in a non-blocking all-to-all topology.

---

## Going deeper

- [K1] NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Official microarchitecture guide: GPCs, TPCs, SM subpartitions, HBM3 memory controllers, and TMA engine.
- [K2] NVIDIA Corporation. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html). Hardware limits: 64 warps per SM, 65,536 registers, 255 registers per thread max, 228 KB max shared memory.
- [K3] Weile Luo et al. (2024). [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499). arXiv:2402.13499. Measured Hopper load latencies: SMEM ~29, L1 ~41, L2 ~263, and HBM ~479 cycles.
- [K4] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao (2024). [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608). arXiv:2407.08608. Hopper warp specialization, asynchronous TMA transfers, and wgmma.
- [K5] Samuel Williams, Andrew Waterman, David Patterson (2009). [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785). Communications of the ACM, 52(4), 65–76. Formulation of critical arithmetic intensity and ridge points.
- [K6] Jacob Austin, Swapnil Patil, Adam Paszke, Reiner Pope (2025). [How to Scale Your Model: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). JAX-ML Scaling Book, Chapter 12. Hardware hierarchy and bandwidth differences.
- [K7] NVIDIA Corporation. [Introduction to NVIDIA DGX H100/H200 Systems](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). Node hardware: NUMA topology, PCIe switches, dual-socket Xeon architecture, and NVMe pathways.
- [K8] NVIDIA Corporation. [Fabric Manager for NVIDIA NVSwitch Systems](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). NVSwitch crossbar architecture, 18-link 5+4+4+5 distribution, and 3.6 TB/s bisection bandwidth.
- [K9] On this blog: [AI Infrastructure (Part 2): Inside LLM Workloads — Training, Prefill, and Decode](post.html?slug=gpu-workloads-and-production-inference).
- [K10] On this blog: [AI Infrastructure (Part 3): Multi-GPU Fabrics and Distributed Training — NVLink, InfiniBand, and 4D Parallelism](post.html?slug=multi-gpu-networks-and-distributed-training).
