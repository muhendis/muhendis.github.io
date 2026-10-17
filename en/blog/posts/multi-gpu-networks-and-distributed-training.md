When one GPU lacks memory or compute, we split the model across GPUs. The second GPU adds more than capacity: it also adds synchronization, message latency, and physical topology.

**The central idea:** Distributed-training performance depends on matching the mathematical partition of the model to physical tensor paths.

> **What we carry forward:** [The second article](post.html?slug=gpu-workloads-and-production-inference) separated training, prefill, and decode resource profiles. We now follow what crosses NVLink, PCIe, and the scale-out fabric after one GPU is no longer enough.

> **By the end of this article**
>
> - You will calculate ring-AllReduce traffic volume and latency steps analytically.
> - You will explain the differences, tensor transforms, and memory savings between TP, PP, DP/FSDP, CP, and EP.
> - You will place parallel groups onto physical topology (NVLink, Leaf, Spine) following strict communication hierarchy.
> - You will follow compute, collective communication, dataloader, and checkpoint flows through one training step with critical path analysis.
> - You will evaluate scaling efficiency, straggler impact, and fabric bottlenecks with quantitative rigor.

**In this article**

- [1. Which physical paths connect GPUs?](#1-which-physical-paths-connect-gpus)
- [2. How much data do collectives move?](#2-how-much-data-do-collectives-move)
- [3. Along which dimensions can a model be partitioned?](#3-along-which-dimensions-can-a-model-be-partitioned)
- [4. How is parallelism mapped to topology?](#4-how-is-parallelism-mapped-to-topology)
- [5. Where does data move during one training step?](#5-where-does-data-move-during-one-training-step)
- [6. Scheduling, sharing, and validation](#6-scheduling-sharing-and-validation)
- [7. A 16-GPU design example](#7-a-16-gpu-design-example)
- [8. Seven core system design questions](#8-seven-core-system-design-questions)
- [What questions can the reader now answer?](#what-questions-can-the-reader-now-answer)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [References and verification links](#references-and-verification-links)

---

## 1. Which physical paths connect GPUs?

In a distributed AI cluster, there is no abstract "GPU count"; there is a physical topology graph dictating the bandwidth, latency, and switching tiers connecting those GPUs. No matter how fast matrix math completes on Tensor Cores, if the pipes carrying tensors become congested or lag behind, Tensor Cores stall in idle cycles.

| Path | Scope | Bidirectional Bandwidth | Typical Latency ($\alpha$) | Most Suitable Traffic |
|---|---|---|---|---|
| **HBM3 / HBM3e** | Intra-GPU | $3.35 - 8.0 \text{ TB/s}$ | $\approx 30 - 50 \text{ ns}$ | Weights, activations, local gradient matrices |
| **NVLink 4 / NVSwitch** | Intra-node (8 GPUs) | $900 \text{ GB/s}$ / GPU | $\approx 0.5 - 0.8\ \mu\text{s}$ | Tensor Parallelism (TP), high-frequency all-reduce |
| **PCIe Gen5** | GPU ↔ Host / NIC | $128 \text{ GB/s}$ ($x16$) | $\approx 1.0 - 1.5\ \mu\text{s}$ | Dataloader token streams, CPU offload, local NVMe I/O |
| **InfiniBand (NDR 400G)** | Inter-node fabric | $100 \text{ GB/s}$ ($800\text{Gb/s}$ bidi) | $\approx 1.2 - 2.0\ \mu\text{s}$ | Data Parallelism (DP/FSDP), Pipeline Parallelism (PP) |
| **RoCE v2 (400GbE)** | Inter-node Ethernet | $100 \text{ GB/s}$ per port | $\approx 2.5 - 5.0\ \mu\text{s}$ | Scale-out RDMA, lossy fabric management |

### 1.1. GPU–NIC affinity

Inside a DGX H100 node, we find two x86 CPUs, eight H100 SXM5 GPUs, four NVSwitch 3 ASICs, and eight OSFP ConnectX-7 NICs (Network Interface Cards). These components are not wired haphazardly; they adhere to a rigid Non-Uniform Memory Access (NUMA) and PCIe tree hierarchy.

A 4-step mental model for optimizing the GPU-to-NIC physical path:

1. **NUMA Node Partitioning:** GPUs 0–3 and NICs 0–3 map to CPU socket NUMA Node 0; GPUs 4–7 and NICs 4–7 map to NUMA Node 1. The two CPU sockets communicate across UPI (Ultra Path Interconnect, $\approx 64 \text{ GB/s}$) links.
2. **GPUDirect RDMA (GDR) Path:** In an optimal distributed setup, when GPU 0 transmits a tensor to another node in the cluster, data flows straight through the local PCIe switch to NIC 0 (`GPU HBM -> PCIe Switch -> NIC -> Fabric`). Host RAM and the OS kernel are completely bypassed (zero-copy RDMA).
3. **Cross-NUMA Penalty (The UPI Bottleneck):** If software misconfiguration causes GPU 0 to route traffic through NIC 4, packets must traverse PCIe Switch 0 $\to$ CPU 0 $\to$ UPI link $\to$ CPU 1 $\to$ PCIe Switch 1 $\to$ NIC 4. The UPI link chokes; latency jumps from $1.2\ \mu\text{s}$ to $3.8\ \mu\text{s}$ and bandwidth drops by over 50%.
4. **NVSwitch vs. PCIe Disconnect:** The eight GPUs share an all-to-all $900 \text{ GB/s}$ coherent domain via NVSwitch. But any byte leaving the chassis must traverse PCIe Gen5 lanes to reach an InfiniBand or RoCE NIC.



<!-- restored-svg:en_full:20 -->
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

### 1.2. Rail topology

In a supercomputer with thousands of GPUs (e.g., a DGX SuperPOD), arbitrary inter-node wiring creates massive switch queues and congestion. To prevent this, clusters deploy a multi-tier, non-blocking **Fat-Tree Clos** network architecture, customized for AI workloads as a **Rail-Optimized Network**.

Principles of rail-optimized network design:

1. **Rail Isolation:** Each DGX node contains 8 compute NICs numbered 0 through 7. Across the entire cluster, all NIC 0 interfaces wire to an independent Leaf switch group (Rail 0), all NIC 1 interfaces wire to Rail 1, ..., and all NIC 7 interfaces wire to Rail 7.
2. **Scalable Unit (SU):** A standard DGX H100 SuperPOD SU houses 32 nodes (256 GPUs). Inside one SU, each rail has one 64-port Quantum-2 InfiniBand switch (8 leaf switches total). All 32 nodes connect their $r$-th NIC directly to Leaf switch $r$ (32 downlinks). The remaining 32 ports cable up to the Spine tier (32 uplinks).
3. **Traffic Localization:** When running Data Parallelism or Tensor Parallelism, GPU $r$ consistently communicates with GPU $r$ on other nodes. Rail optimization guarantees this traffic remains entirely inside a single leaf switch; packets never climb to the spine tier, achieving single-hop latency.
4. **Non-blocking Spine Bisection:** To bridge disparate SUs, leaf uplinks connect to 16 Spine switches with a 1:1 oversubscription ratio (perfectly symmetric bandwidth). Across a 1,024-GPU cluster, any GPU pair maintains full $400 \text{ Gb/s}$ ($50 \text{ GB/s}$) non-blocking bisection bandwidth.



<!-- restored-svg:en_full:22 -->
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

<!-- restored-svg:en_full:23 -->
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

<!-- restored-svg:en_full:21 -->
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

### 1.3. Which messages care about latency and bandwidth?

To quantify the time required to move tensors across physical interconnects, we use the foundational **Hockney ($\alpha$-$\beta$) Model** from parallel computing. The transfer time $T$ for a message of size $S$ is expressed in terms of network latency $\alpha$ and effective bandwidth $\beta$:

$$
T(S) = \alpha + \frac{S}{\beta}
$$

A critical tipping point in this equation is the **Characteristic Message Size ($S_{1/2}$)**, also known as the half-power point, where half the transfer duration is spent on latency ($\alpha$) and half on bandwidth ($S/\beta$):

$$
S_{1/2} = \alpha \times \beta
$$

Analyzing physical fabric interconnects through this lens:

- **NVLink 4:** $\alpha \approx 0.5\ \mu\text{s}$, $\beta = 450 \text{ GB/s}$ (unidirectional). $S_{1/2} \approx 225 \text{ KB}$.
- **InfiniBand NDR (400 Gb/s):** $\alpha \approx 1.5\ \mu\text{s}$, $\beta = 50 \text{ GB/s}$. $S_{1/2} \approx 75 \text{ KB}$.
- **Standard Ethernet (TCP/IP 100 Gb/s):** $\alpha \approx 25\ \mu\text{s}$, $\beta = 12.5 \text{ GB/s}$. $S_{1/2} \approx 312 \text{ KB}$.

This math dictates two operational regimes:

1. **Latency-Dominated Regime ($S \ll S_{1/2}$):** Pipeline Parallelism activation handoffs, distributed barriers, MoE routing decisions, and scalar synchronizations fall here. A $400 \text{ Gb/s}$ link provides zero speedup if the payload is small; the clock is dictated strictly by physical switching latency ($\alpha$).
2. **Bandwidth-Dominated Regime ($S \gg S_{1/2}$):** Large Data Parallelism gradient reductions ($S \ge 100 \text{ MB}$), FSDP model parameter AllGathers, and Megatron TP layer activations fall here. Here, $\alpha$ is negligible; link throughput saturation ($\beta$) governs completion time.

---

## 2. How much data do collectives move?

Distributed deep learning does not rely on point-to-point socket messaging; it relies on synchronous collective communication primitives executed concurrently across ranks. The NVIDIA Collective Communications Library (NCCL) is the de facto standard engine optimizing these primitives for hardware topologies.

### 2.1. Core collectives

All distributed training frameworks build upon four fundamental collective operations:

1. **AllReduce:** Every rank contributes a local tensor of size $S$. A reduction operation (typically SUM) is applied element-wise across all ranks, and the final reduced tensor ($S$) is replicated across every participating rank.
   - *Use Cases:* Tensor Parallelism layer outputs, Data Parallelism gradient synchronization.
2. **ReduceScatter:** Every rank contributes a tensor of size $S$. The reduction is computed, but instead of replicating the result, each rank receives a disjoint $S/N$ slice (shard) of the final tensor.
   - *Use Cases:* ZeRO-2 and ZeRO-3 gradient aggregation, Sequence Parallelism.
3. **AllGather:** Every rank contributes its local shard ($S/N$). Communication concatenates these shards in order, resulting in every rank holding the complete tensor ($S$). It is the exact inverse of ReduceScatter.
   - *Use Cases:* FSDP (ZeRO-3) forward and backward layer weight gathering.
4. **AllToAll:** Every rank prepares a personalized, distinct slice of data for every other rank in the group. Similar to a matrix transpose ($N \times N$), every rank transmits a custom chunk and receives custom chunks from all peers.
   - *Use Cases:* Mixture of Experts (MoE) token dispatch/combine and Context Parallelism ring passes.



<!-- restored-svg:en_full:25 -->
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

### 2.2. Ring-AllReduce calculation

Executing an AllReduce naively (such as every GPU transmitting directly to every other GPU, or routing through a parameter server) incurs $O(N)$ communication complexity, saturating fabric links as node counts grow. The **Ring AllReduce Algorithm**, popularized by Baidu and adopted by NCCL, decouples the data volume transferred per GPU from cluster size.

A 4-step walkthrough of Ring AllReduce:

1. **Logical Ring and Sharding:** $N$ GPUs are arranged in a logical ring ($0 \to 1 \to 2 \to \dots \to N-1 \to 0$). The tensor of size $S$ bytes is partitioned into $N$ equal chunks of size $S/N$.
2. **Reduce-Scatter Phase ($N-1$ Steps):**
   - At each step, every GPU sends one chunk to its clockwise neighbor while receiving a chunk from its counter-clockwise neighbor, accumulating received values into its local chunk.
   - After $N-1$ transfer steps, each GPU holds the fully accumulated sum for exactly one chunk ($S/N$ bytes).
   - Data sent per GPU during this phase:
     $$V_{\mathrm{RS}} = (N-1) \times \frac{S}{N} = \frac{N-1}{N} S$$
3. **All-Gather Phase ($N-1$ Steps):**
   - The goal is now to circulate the fully reduced chunks so every GPU possesses the full tensor.
   - Across another $N-1$ steps, each GPU sends its fully accumulated chunk around the ring, overwriting local buffers until every rank has received all chunks.
   - Data sent per GPU during this phase:
     $$V_{\mathrm{AG}} = (N-1) \times \frac{S}{N} = \frac{N-1}{N} S$$
4. **Total Data Volume and Duration:**
   The total data transmitted per GPU is the sum of both phases:
   $$V_{\mathrm{ring}} = V_{\mathrm{RS}} + V_{\mathrm{AG}} = 2 \left(\frac{N-1}{N}\right) S$$
   Under the Hockney model, total Ring AllReduce duration is:
   $$T_{\mathrm{ring}} = 2(N-1)\alpha + 2\left(\frac{N-1}{N}\right)\frac{S}{\beta}$$

**Numerical Walk:**
Consider synchronizing an $S = 8\text{ GB}$ gradient tensor across an 8-GPU node ($N=8$):
- Data transmitted per GPU:
  $$V_{\mathrm{ring}} = 2 \times \frac{8-1}{8} \times 8\text{ GB} = 2 \times \frac{7}{8} \times 8 = 14\text{ GB}$$
- In the limit as $N \to \infty$, $\frac{N-1}{N} \to 1$, meaning transferred data per GPU asymptotically approaches $2S = 16\text{ GB}$ and never exceeds it! Even scaling from 8 to 1,024 GPUs, the per-GPU data load remains capped.
- However, latency overhead $2(N-1)\alpha$ scales linearly. At $N=1,024$, 2,046 latency hops accumulate, which is why large clusters employ hierarchical rings (intra-node + inter-node) or tree-based algorithms.



<!-- restored-svg:en_full:26 -->
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

### 2.3. Ring or tree?

NCCL dynamically selects between Ring and Tree topologies based on message size and cluster geometry:

1. **Tree-Based AllReduce (Binomial Tree / Double Binary Tree):**
   - Reductions propagate up a binary tree to the root and broadcast back down to the leaves.
   - Latency scales with $2 \log_2 N$ hops instead of $2(N-1)$. For 1,024 GPUs, a ring requires 2,046 hops, whereas a binary tree finishes in just $2 \times 10 = 20$ hops!
   - However, tree algorithms cannot achieve 100% link utilization across all intermediate links simultaneously. Thus, Tree wins for latency-bound small messages, while Ring wins for bandwidth-bound massive tensors.
2. **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol):**
   - NVIDIA Quantum-2 InfiniBand switches feature embedded hardware arithmetic logic units (ALUs), enabling **In-Network Computing**.
   - Traditionally, GPUs exchange payloads across the network while GPU CUDA/Tensor cores perform floating-point additions, causing data to traverse links multiple times.
   - With SHARP, GPUs stream gradient vectors directly to the switch ASIC. The switch performs FP16/BF16/FP32 reductions on-chip at line rate and multicasts the reduced tensor back down to the GPUs in a single broadcast pass.
   - Physical cable traffic drops by 50% ($S$ upstream, $S$ downstream), cutting large-scale AllReduce time almost in half.



<!-- restored-svg:en_full:27 -->
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

### 2.4. What does overlap provide?

The primary killer of distributed scaling efficiency is idle waiting: GPUs idling while the network transfers data, or network links sitting empty while GPUs compute. Modern frameworks (PyTorch DDP, Megatron-LM) leverage **Compute–Communication Overlap**.

This mechanism operates via concurrent CUDA Streams and **Gradient Bucketing**:

1. **Reverse Execution Order:** Backward passes traverse layers in reverse, from Layer 80 down to Layer 1.
2. **Bucketing:** PyTorch DDP groups layer gradients into contiguous memory buffers (e.g., $25 \text{ MB}$ buckets).
3. **Concurrent Execution:** The instant Layer 80's bucket fills, an asynchronous AllReduce is launched on a dedicated NCCL communication stream. Simultaneously, the GPU compute stream continues crunching backward GEMM kernels for Layers 79, 78, and 77.
4. **Hidden Time:** If communication latency ($T_{\mathrm{comm}}$) is less than or equal to computation duration ($T_{\mathrm{comp}}$), the network transfer is completely hidden behind the math ($T_{\mathrm{comm}} \le T_{\mathrm{comp}}$). A profiler shows virtually zero exposed network stalls.



<!-- restored-svg:en_full:28 -->
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

### 2.5. Why is collective time more than bytes divided by bandwidth?

When running an AllReduce across 1,024 GPUs, even if Hockney's equation predicts a 5 ms transfer, your profiler might log 18 ms. This discrepancy occurs because collective operations represent strict **distributed synchronization barriers**:

$$
T_{\mathrm{collective}} = \max_{i \in [1, N]} (T_{\mathrm{compute}, i}) + \max_{i \in [1, N]} (T_{\mathrm{comm}, i}) + T_{\mathrm{skew}}
$$

Three physical causes drive this tail latency:

1. **Straggler Effect:** If a single GPU out of 1,024 throttles clock speeds due to thermal limits or pauses to correct DRAM ECC bit errors, it finishes backward compute 10 ms late. The remaining 1,023 GPUs sit frozen at the AllReduce barrier.
2. **Queuing Delay and Packet Drops:** On RoCE v2 lossy Ethernet fabrics, when Priority Flow Control (PFC) emits pause frames, buffer congestion can cascade backward (head-of-line blocking), halting transmission across an entire rail.
3. **P-99 Tail Distribution:** As rank counts expand, the probability that at least one GPU experiences a P-99 or P-99.9 latency blip approaches 1.0. A collective's speed is pegged not to the median (P-50), but to the slowest outlier (P-100).

---

## 3. Along which dimensions can a model be partitioned?

Scaling a multi-billion parameter model across hundreds of GPUs cannot rely on a single technique. Different tensor dimensions (batch size, layer depth, hidden dimension, sequence length, and expert count) are partitioned across distinct parallel axes, creating **Multi-Dimensional (3D / 4D) Parallelism**.

| Parallelism Axis | Partitioned Dimension | Collective Type | Frequency | Minimum Hardware Requirement |
|---|---|---|---|---|
| **Data Parallelism (DP)** | Batch Size ($B$) | AllReduce | Once per step (end of backward) | InfiniBand / RoCE |
| **ZeRO-3 / FSDP** | Weights, Grads, Opt State | AllGather + ReduceScatter | Per layer (forward & backward) | Low-latency Fabric |
| **Tensor Parallelism (TP)** | Hidden Dim ($H$), FFN ($4H$) | AllReduce | Twice per Transformer layer | Intra-node NVLink ($900\text{ GB/s}$) |
| **Pipeline Parallelism (PP)** | Layer Depth ($L$) | P2P Send/Recv | Per microbatch at stage boundary | Inter-node Fabric |
| **Context Parallelism (CP)** | Sequence Length ($S$) | AllToAll / P2P Ring | Per Attention layer | Intra-node NVLink / Low-latency Leaf |
| **Expert Parallelism (EP)** | MoE Experts ($E$) | AllToAll (Dispatch & Combine) | Twice per MoE layer | Rail-optimized Fabric (Low oversubscription) |

### 3.1. Data Parallelism

In classical Distributed Data Parallelism (DDP), every GPU replicates the entire model weights, optimizer states, and gradients in its local HBM.

1. **Mechanism:** The global batch $B$ is partitioned across $N_{\mathrm{DP}}$ GPUs. Each GPU processes its independent microbatch slice ($B / N_{\mathrm{DP}}$) through forward and backward passes.
2. **Synchronization:** At the end of backward, each GPU possesses local gradients. An AllReduce averages gradients across all replicas, after which each GPU updates its local optimizer weights identically.
3. **Capacity Ceiling:** As models scale (e.g., a 70B parameter LLM), static model parameters and optimizer states exceed single-GPU HBM ($80 \text{ GB}$), breaking standard DDP and necessitating FSDP.



<!-- restored-svg:en_full:30 -->
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

### 3.2. FSDP and ZeRO

Introduced by DeepSpeed and standardized in PyTorch as **Fully Sharded Data Parallel (FSDP)**, the **ZeRO (Zero Redundancy Optimizer)** family eliminates redundant state storage across data-parallel ranks.

Memory accounting for FP16/BF16 training under AdamW:
- Model Parameters (FP16/BF16): $2\text{ bytes}$
- Gradients (FP16/BF16): $2\text{ bytes}$
- Optimizer States (FP32 master weights, momentum, variance): $4 + 4 + 4 = 12\text{ bytes}$
- **Total Static Memory:** Exactly **$16\text{ bytes}$** per parameter!

For a 70-billion parameter model, static memory demands:
$$M_{\mathrm{static}} = 70 \times 10^9 \times 16\text{ bytes} \approx 1,120\text{ GB} = 1.12\text{ TB}$$
Fitting this onto an 80 GB GPU is impossible. ZeRO shards this state across three progressive tiers:

1. **ZeRO-1 (Optimizer State Partitioning):**
   - The $12\text{ bytes}$ of optimizer state are split across $N$ GPUs ($12 / N$).
   - Memory per GPU: $2 + 2 + \frac{12}{N}$ bytes. For 4 GPUs, parameter memory drops from $16$ to $7\text{ bytes}$.
   - *Communication Overhead:* Zero! The standard AllReduce is swapped for a ReduceScatter; each rank updates only its local slice of optimizer states.
2. **ZeRO-2 (Gradient Partitioning):**
   - Gradients are also sharded across $N$ GPUs ($2 / N$).
   - Memory per GPU: $2 + \frac{2}{N} + \frac{12}{N}$ bytes. For 4 GPUs, memory drops to $5.5\text{ bytes}$ per parameter.
   - *Communication Overhead:* Zero!
3. **ZeRO-3 / FSDP (Parameter Partitioning):**
   - Model parameters themselves are sharded across $N$ GPUs ($2 / N$).
   - Memory per GPU: $\frac{2 + 2 + 12}{N} = \frac{16}{N}$ bytes. Memory scales down linearly; for 4 GPUs, it drops to $4\text{ bytes}$ per parameter.
   - *Communication Cost:* Memory drops, but communication increases! Before computing a layer's forward pass, an `AllGather` fetches weights from peers, discarding them immediately afterward. In backward, weights are gathered again via `AllGather`, backward runs, weights are discarded, and gradients are sharded via `ReduceScatter`.
   - FSDP increases total communication volume by 50% relative to DDP ($2S \to 3S$), in exchange for massive memory savings.



<!-- restored-svg:en_full:33 -->
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

### 3.3. Tensor Parallelism

Even with FSDP, extreme models can have activations or individual matrix multiplications that exceed a single GPU's memory or cause intolerable single-device compute latency. Megatron-LM introduced **Tensor Parallelism (TP)**, splitting weight matrices within each Transformer layer across row or column dimensions.

Megatron-LM MLP 4-step parallel execution:

1. **Input Broadcast:** Input tensor $X$ is replicated identically across all TP ranks.
2. **Column Parallel Linear (First Projection - $W_1$):**
   - The weight matrix of the first linear layer is split along its columns across $N_{\mathrm{TP}}$ ranks: $W_1 = [W_{1,1} \mid W_{1,2} \mid \dots \mid W_{1,k}]$.
   - Each GPU multiplies independently: $Y_i = \text{GELU}(X \cdot W_{1,i})$.
   - Zero communication required! Intermediate activation size is $4H / N_{\mathrm{TP}}$.
3. **Row Parallel Linear (Second Projection - $W_2$):**
   - The weight matrix of the second linear layer is split along its rows:
     $$W_2 = \begin{bmatrix} W_{2,1} \\ W_{2,2} \\ \vdots \\ W_{2,k} \end{bmatrix}$$
   - Each GPU multiplies its local activation slice with its row slice: $Z_i = Y_i \cdot W_{2,i}$.
   - Mathematically, the true output is the sum of these local products:
     $$Z = X W_1 W_2 = \sum_{i=1}^{k} Z_i$$
4. **AllReduce Barrier:**
   - The local $Z_i$ matrices are partial results. An **AllReduce (SUM)** is executed at the layer output, writing the complete $Z$ tensor to all GPUs.

**The Golden Rule: Why TP Never Leaves the Node**
A standard Transformer block requires an AllReduce after Self-Attention and another after the MLP. In an 80-layer model, a single forward pass triggers $80 \times 2 = 160$ AllReduces, totaling **320 AllReduces** when including backward! Because each AllReduce sits on the critical execution path, this operation requires sub-microsecond latency and $900 \text{ GB/s}$ bandwidth available exclusively on NVSwitch. Crossing InfiniBand or RoCE would cause the model to spend over 80% of its execution time waiting on network queues.



<!-- restored-svg:en_full:31 -->
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

### 3.4. Pipeline Parallelism

When a model is so large that its layers cannot fit within a single 8-GPU node (e.g., a 405B model spanning 128 layers), layers are grouped into sequential stages and distributed across nodes via **Pipeline Parallelism (PP)**.

For instance, across 4 pipeline stages ($p=4$):
- Stage 0: Layers 1–20
- Stage 1: Layers 21–40
- Stage 2: Layers 41–60
- Stage 3: Layers 61–80

#### Pipeline Bubbles and the 1F1B Schedule

In a naive execution, Stage 1 sits idle while Stage 0 executes. This wasted execution capacity is the **Pipeline Bubble**. To shrink the bubble, the global batch is split into numerous smaller **microbatches** ($m$), and the **1F1B (One Forward, One Backward)** schedule is applied.

Under 1F1B, each GPU alternates between executing one forward microbatch and one backward microbatch, maintaining steady pipeline flow and bounding activation memory.

The mathematical formula for ideal 1F1B bubble fraction is:

$$
F_{\mathrm{bubble}} = \frac{p - 1}{m + p - 1}
$$

Where $p$ is the number of pipeline stages and $m$ is the number of microbatches.

**Numerical Walk:**
Consider a 16-stage pipeline ($p=16$) as microbatch count varies:
- $m = 16 \implies F_{\mathrm{bubble}} = \frac{15}{16 + 15} = \frac{15}{31} \approx 48.4\%$ (Nearly half of compute capacity is wasted!)
- $m = 32 \implies F_{\mathrm{bubble}} = \frac{15}{32 + 15} = \frac{15}{47} \approx 31.9\%$
- $m = 64 \implies F_{\mathrm{bubble}} = \frac{15}{64 + 15} = \frac{15}{79} \approx 19.0\%$
- $m = 128 \implies F_{\mathrm{bubble}} = \frac{15}{128 + 15} = \frac{15}{143} \approx 10.5\%$

To push the bubble fraction below 10%, microbatch depth must reach $m \ge 8p$. However, higher $m$ increases stored activation memory. Modern systems bypass this trade-off using **Interleaved 1F1B** (assigning multiple non-contiguous virtual stages per GPU) or **DeepSeek-V3 DualPipe** (bidirectional pipelining overlapping forward and backward execution).



<!-- restored-svg:en_full:32 -->
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

### 3.5. Context and Expert Parallelism

Emerging AI workloads introduce two additional tensor dimensions that require specialized parallelization: ultra-long context windows and sparse Mixture of Experts (MoE).

#### Context Parallelism (CP / RingAttention)
When context length scales from $32\text{K}$ to $128\text{K}$ or $1\text{M}$ tokens, attention memory scales quadratically ($O(S^2)$), quickly exhausting HBM.
- **Mechanism:** The input sequence is split across the token dimension into $N_{\mathrm{CP}}$ chunks. Each GPU receives an $S / N_{\mathrm{CP}}$ slice.
- **RingAttention:** Query ($Q$) blocks stay pinned on the local GPU, while Key ($K$) and Value ($V$) blocks circulate around a logical ring via P2P transfers. Each GPU computes partial attention against rotating $K, V$ blocks, updating softmax statistics online.

#### Expert Parallelism (EP / MoE)
In Mixture of Experts architectures (such as Mixtral or DeepSeek), the dense MLP is replaced by dozens of sparse expert sub-networks. A router network assigns each incoming token to its Top-K experts.
- **Mechanism:** Experts are distributed across GPUs (e.g., 64 experts across 16 GPUs, 4 experts per GPU).
- **AllToAll Dispatch:** Tokens destined for remote experts are routed across the fabric using an `AllToAll` collective.
- **Execution and AllToAll Combine:** Expert GPUs execute their local feedforward math, and output vectors are returned to their originating GPUs via a second `AllToAll`.
- **Failure Mode:** If routing decisions become skewed, specific expert GPUs experience hot-spots, stalling the entire cluster at the AllToAll barrier.



<!-- restored-svg:en_full:34 -->
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

### 3.6. In what order should a parallel plan be selected?

When architecting distributed training, engineering decisions follow a strict hierarchical sequence:

1. **Step 1: Set TP (Intra-Node Boundary):** If layer activations or weights exceed single-GPU capacity, scale $TP$. Keep $TP \le 8$ (never exceed one DGX chassis or the NVSwitch domain).
2. **Step 2: Enable CP (Long Context):** If sequence length $S \ge 32\text{K}$, enable $CP$ ($CP = 2, 4, 8$) to avoid attention memory blowup, pairing it with TP in low-latency domains.
3. **Step 3: Add PP (Model Still Too Large):** If the model still cannot fit in memory after TP, partition layers across sequential nodes using $PP$ ($PP = 4, 8, 16$).
4. **Step 4: Fill Remaining Scale with DP / FSDP:** All remaining cluster nodes are assigned to Data Parallelism ($DP$) to maximize global batch size and training throughput.
5. **Step 5: Apply EP if MoE:** For MoE models, map experts across the DP replica pool ($DP = EP \times DP_{\mathrm{replicate}}$).

---

## 4. How is parallelism mapped to topology?

Designing a parallel strategy is only half the battle; mapping those abstract groups to physical cables, switches, and rails dictates real-world efficiency.

### 4.1. Placement priority

The golden rule of cluster topology placement: **The parallel dimension with the highest communication intensity and frequency must map to the fastest, lowest-latency physical interconnect.**

```
Fast Interconnect  <--------------------------------------------------> Slow Interconnect
NVLink (900 GB/s)      SU-Local Leaf (50 GB/s)    Spine Switch (Clos)      Storage Fabric
[      TP      ]  ->  [     CP / EP     ]   ->   [     PP / DP     ]   ->  [  Checkpoint  ]
```

Violating this hierarchy degrades performance instantly:
- Spreading TP across scale-out network fabrics cuts bandwidth from $900 \text{ GB/s}$ to $50 \text{ GB/s}$, crippling training throughput.
- DP gradient synchronizations, which run once per step and overlap with backward math, comfortably traverse multi-tier Clos spine fabrics.



<!-- restored-svg:en_full:35 -->
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

### 4.2. Product of parallel dimensions

Total cluster GPU count ($N_{\mathrm{total}}$) must strictly equal the mathematical product of all active parallel axes:

$$
N_{\mathrm{total}} = TP \times CP \times EP \times PP \times DP
$$

#### Case Study: Meta Llama 3 405B

Meta trained the 405-billion parameter Llama 3 model on **16,384 NVIDIA H100 GPUs**. They deployed two distinct 4D parallelism topologies:

1. **Standard Context (8K Tokens):**
   - $TP = 8$ (Each 8-GPU group occupies one DGX H100 chassis, keeping all TP traffic on NVLink).
   - $CP = 1$ (8K tokens fit in local memory; CP disabled).
   - $PP = 16$ (The 128-layer model is divided into 16 pipeline stages of 8 layers each. 16 chassis form a 128-GPU model replica).
   - $DP = 128$ (128 independent model replicas run across the cluster).
   - Verification: $8 \times 1 \times 16 \times 128 = 16,384\text{ GPUs}$.
   - **Performance:** Achieved **41% BF16 Model FLOPs Utilization (MFU)**.
2. **Long Context (128K Tokens):**
   - At 128K context, activation memory ballooned, requiring Context Parallelism:
   - $TP = 8, CP = 16, PP = 16, DP = 8$
   - Verification: $8 \times 16 \times 16 \times 8 = 16,384\text{ GPUs}$.
   - **Performance:** Maintained **38% BF16 MFU** despite substantial CP communication overhead.



<!-- restored-svg:en_full:36 -->
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

### 4.3. Strong and weak scaling

Scaling performance in distributed training is evaluated across two regimes:

1. **Weak Scaling:** Workload per GPU (microbatch size) is held constant as total GPU count grows, expanding global problem size. Deep learning workloads typically excel under weak scaling because the compute-to-communication ratio remains stable. However, if global batch size exceeds the gradient noise scale, optimization dynamics degrade and training fails to converge.
2. **Strong Scaling:** Total problem size (fixed global batch) is held constant while GPU count increases. As GPUs double, compute time per GPU halves ($T_{\mathrm{comp}} / 2$). However, collective communication data volume stays flat or latency hops increase ($2(N-1)\alpha$). Eventually, communication dominates compute, and Amdahl's Law caps speedup: adding more GPUs yields diminishing returns or degrades throughput.

Scaling efficiency formula:

$$
\eta = \frac{T_1}{k \times T_k} \times 100\%
$$

Where $T_1$ is execution time on the baseline cluster and $T_k$ is time on a cluster $k$ times larger. While ideal efficiency is $\eta = 100\%$, production AI clusters achieving $\ge 80 - 85\%$ are considered highly optimized.

---

## 5. Where does data move during one training step?

Profiling distributed training requires understanding the physical lifecycle of tensors across hardware during a single step.



<!-- restored-svg:en_full:38 -->
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

### 5.1. Input

Token batches read from distributed parallel storage (Lustre, GPFS, or NVMe-oF). CPU dataloader worker processes batch tokens, write to pinned host memory, and launch asynchronous transfers via `cudaMemcpyAsync` over PCIe Gen5 into GPU HBM. Any I/O stall or host CPU starvation causes immediate GPU starvation (Input Stall).

### 5.2. Forward

Once data reaches HBM:
1. Embedding layers convert token IDs to embedding vectors.
2. The TP group begins Transformer blocks: Column-parallel GEMM executes, intermediate activations generate, Row-parallel GEMM executes, and an AllReduce across NVLink aggregates layer output.
3. FlashAttention kernels compute $Q, K, V$ attention matrices.
4. At pipeline stage boundaries, activation tensors are transmitted via InfiniBand P2P Send to the subsequent node.
5. Intermediate activations are cached in HBM for backward pass (or discarded if using selective activation recomputation).

### 5.3. Backward

At the final pipeline stage, the loss function evaluates, triggering backpropagation:
1. Adjoint gradient tensors compute in reverse order.
2. Across stage boundaries, gradients transmit via P2P to preceding stages.
3. As layer parameter gradients resolve, PyTorch DDP / FSDP copies them into communication buckets. Once full, an asynchronous AllReduce / ReduceScatter launches on a background CUDA stream.
4. Computation and communication race; ideally, all gradient synchronization completes the instant backward computation finishes.

### 5.4. Optimizer and checkpoint

Once gradients are unified across ranks:
1. Optimizer kernels (e.g., FP32 AdamW) execute, updating master weights, momentum, and variance tensors.
2. Master weights are cast back to FP16/BF16 and written to model parameter buffers.
3. At scheduled intervals (e.g., every 1,000 steps), the cluster saves state to parallel storage via synchronous checkpointing or background non-blocking DMA.

### 5.5. Step-time budget

Total step duration is broken down into core components:

$$
T_{\mathrm{step}} = T_{\mathrm{compute}} + T_{\mathrm{unhidden\ comm}} + T_{\mathrm{input\ stall}} + T_{\mathrm{checkpoint\ amortized}}
$$

Where $T_{\mathrm{unhidden\ comm}}$ represents exposed communication that fails to overlap with compute, directly stalling Tensor Cores.

### 5.6. Building one step’s critical path with numbers

Let us model a concrete profiler trace:
- Forward pass: $14\text{ ms}$
- Backward pass: $28\text{ ms}$
- Optimizer step: $3\text{ ms}$
- Total raw computation time: $T_{\mathrm{compute}} = 14 + 28 + 3 = 45\text{ ms}$
- Total collective network time: $22\text{ ms}$ (Gradient AllReduce)
- Pipeline activation transfer: $4\text{ ms}$ (Exposed latency)
- Dataloader batch preparation: $10\text{ ms}$

Adding these numbers naively yields $45 + 22 + 4 + 10 = 81\text{ ms}$. This interpretation is **completely incorrect**!

Critical path decomposition:
1. The dataloader pre-fetches data asynchronously during the preceding step, contributing **$0\text{ ms}$** to the critical path.
2. Out of $22\text{ ms}$ of gradient AllReduce, $18\text{ ms}$ overlaps cleanly with backward computation ($28\text{ ms}$). Exposed communication: $22 - 18 =$ **$4\text{ ms}$**.
3. Pipeline P2P transfer delays the start of the next stage, lying directly on the critical path: **$4\text{ ms}$**.

True step time:
$$T_{\mathrm{step}} = 45\text{ ms (compute)} + 4\text{ ms (unhidden comm)} + 4\text{ ms (pipeline bubble)} = 53\text{ ms}$$

In this cluster, upgrading network cards to 800G yields minimal improvement; optimization must focus on eliminating the $4\text{ ms}$ pipeline bubble!

---

## 6. Scheduling, sharing, and validation

Setting up cluster hardware is meaningless without proper job scheduling and fault isolation.

### 6.1. Gang scheduling

In cloud microservices, if a pod crashes or starts late, the system survives. In distributed AI training, it is catastrophic. In a 32-node job, if 31 nodes boot while 1 lags behind, the 248 active GPUs lock at the first AllReduce barrier, idling indefinitely.

**Gang Scheduling (All-or-Nothing Scheduling)** via Slurm or Kubernetes Kueue / Volcano is mandatory:
- Unless 100% of requested resources (e.g., 32 nodes, 256 GPUs) can be allocated simultaneously, the scheduler **never starts the job**.
- If any single worker fails, the entire job terminates immediately or restarts cleanly from the last valid checkpoint.

### 6.2. Topology-aware scheduling

Modern schedulers do not treat GPUs as fungible numbers; they parse the physical data center fabric tree:
- An 8-GPU job is never split across two nodes; it is pinned to a single DGX chassis within one NVLink domain.
- A 64-GPU job is pinned to nodes within the same Scalable Unit (SU) under a shared Leaf switch, preventing traffic from traversing spine switches.

### 6.3. MIG, MPS, and time-slicing

When partitioning a single GPU among multiple tasks, three options exist:

| Technology | Isolation Level | Memory Protection | Fault Isolation | Distributed Training Suitability |
|---|---|---|---|---|
| **Multi-Instance GPU (MIG)** | Hardware (Dedicated SMs, memory paths) | Complete (Hardware partition) | Excellent (Instance crash is isolated) | Unsuitable (Single inference / dev only) |
| **Multi-Process Service (MPS)** | Logical (Shared CUDA context on hardware) | Shared HBM | Weak (Memory violation drops context) | Ideal for small models / inference batching |
| **Time-Slicing** | Temporal (Round-robin GPU scheduling) | Shared HBM | None (OOM crashes processes) | Terrible (Severe context-switch penalties) |

For large-scale synchronous training, none of these virtualization layers are used; every GPU is allocated to a single bare-metal rank.

### 6.4. Acceptance testing

A newly deployed or serviced cluster undergoes a strict **Acceptance Testing Pyramid**:

```
          [ Level 5: End-to-End LLM Soak Run (24-48h) ]
        [ Level 4: Full Cluster NCCL AllReduce Sweep (Bisection) ]
      [ Level 3: Dual-Node RDMA Ping-Pong & NCCL Bandwidth ]
    [ Level 2: Single-Node cuBLAS GEMM & NVLink BandwidthTest ]
  [ Level 1: Hardware & DCGM Diagnostics (ECC, Thermal, PCIe Eye) ]
```

1. **Level 1 (DCGM):** NVIDIA Data Center GPU Manager scans for DRAM ECC errors, PCIe link downgrades, power throttling, and thermal anomalies.
2. **Level 2 (Single Node):** `bandwidthTest` validates GPU-HBM ($3.35\text{ TB/s}$) and P2P NVLink ($900\text{ GB/s}$) throughput. `cublasLt-bench` validates raw GEMM TFLOP/s.
3. **Level 3 (Dual-Node RDMA):** `ibv_rc_pingpong` verifies RDMA latency ($< 1.5\ \mu\text{s}$) and `nccl-tests` validates per-rail bandwidth ($> 45\text{ GB/s}$).
4. **Level 4 (Cluster NCCL Sweep):** Concurrent AllReduce and AllToAll sweeps benchmark bisection throughput, flagging flapping links and congested leaf-spine uplinks.
5. **Level 5 (Soak Test):** A 48-hour continuous synthetic LLM training run monitors MFU stability and loss convergence.

---

## 7. A 16-GPU design example

Let us synthesize these principles in a concrete engineering design challenge.

**Scenario:** You have two DGX H100 servers ($N = 16\text{ GPUs}$). You need to fine-tune or pre-train an initial slice of Llama 3 70B. How should you design the parallelism topology?

### 7.1. Physical placement

A common initial instinct might be: 4 pipeline stages with 2 GPUs per stage for TP ($TP=2, PP=4, DP=2$).
This design is **fundamentally flawed!**

Why?
- $TP=2$ wastes the $900 \text{ GB/s}$ NVSwitch intra-node bandwidth across the 8-GPU chassis.
- $PP=4$ creates 4 pipeline boundaries between the two servers, streaming activations and gradients over InfiniBand and incurring an $F_{\mathrm{bubble}} = \frac{3}{m + 3}$ bubble penalty.

**Optimal Design:**
- $TP = 8$: Model layers are partitioned across all 8 GPUs inside each DGX. Intensive layer AllReduces remain confined entirely to local $900 \text{ GB/s}$ NVLink!
- $PP = 1$: Pipelining is disabled, eliminating bubble overhead ($F_{\mathrm{bubble}} = 0$) and stage handoff latency!
- $DP = 2$: The two chassis are paired via Data Parallelism (FSDP / ZeRO-3).

### 7.2. Memory question

Static state for a 70B model requires $1,120\text{ GB}$.
- Under $TP=8$, weights divide across 8 GPUs per node.
- Sharding across both nodes with FSDP (ZeRO-3) distributes state across all 16 GPUs:
  $$M_{\mathrm{per\_gpu}} = \frac{1,120\text{ GB}}{16} = 70\text{ GB}$$
- Each H100 provides $80\text{ GB}$ HBM. The remaining $10\text{ GB}$ combined with FlashAttention and selective activation recomputation allows training to fit comfortably without CPU offloading!

### 7.3. Performance question

The only traffic crossing the scale-out fabric consists of layer-by-layer FSDP `AllGather` and `ReduceScatter` collectives during forward and backward passes.
- Each DGX node provides 8x $400 \text{ Gb/s}$ ConnectX-7 NICs, totaling:
  $$8 \times 50\text{ GB/s} = 400\text{ GB/s}$$
- This massive bandwidth hides FSDP communication almost entirely behind backward GEMM math ($T_{\mathrm{unhidden\ comm}} \approx 0$).

### 7.4. How do you build the communication map?

Document the design with a comprehensive communication matrix:

| Collective Type | Parallel Dimension | Hardware Interconnect | Expected Bandwidth | Profiler Expectation |
|---|---|---|---|---|
| Intra-Layer AllReduce | Tensor Parallel ($TP=8$) | Intra-Node NVSwitch 3 | $900\text{ GB/s}$ per GPU | Distinct bursts between compute GEMMs |
| Weight Fetch (AllGather) | FSDP ($DP=2$) | ConnectX-7 InfiniBand | $400\text{ GB/s}$ aggregated | Overlaps cleanly with forward compute |
| Gradient Shard (ReduceScatter) | FSDP ($DP=2$) | ConnectX-7 InfiniBand | $400\text{ GB/s}$ aggregated | Overlaps cleanly with backward compute |
| Token Input | Dataloader | PCIe Gen5 x16 | $32 - 64\text{ GB/s}$ | Off critical path (asynchronous prefetch) |

### 7.5. How do you interpret a scaling result?

Running on a single DGX (8 GPUs) with $TP=8$ yields a step time of $1.20\text{ seconds}$. Scaling to two DGXs (16 GPUs) at the same batch size yields $0.68\text{ seconds}$.
- Speedup: $\frac{1.20}{0.68} = 1.76\times$
- Scaling Efficiency:
  $$\eta = \frac{1.76}{2.0} \times 100\% = 88.2\%$$
This represents stellar scaling efficiency! The 11.8% loss is cleanly accounted for by inter-node FSDP synchronization barriers and fabric P-99 jitter.

### 7.6. How do you complete the design answer?

In a production system architecture review, deliver a complete design by addressing three operational safeguards:
1. **Failure Domain (Blast Radius):** If one node faults, training halts; mitigate this by saving asynchronous checkpoints to local NVMe storage every 30 minutes.
2. **NUMA Alignment:** Verify that every GPU maps directly to its local ConnectX-7 NIC (audited via `NCCL_TOPO_DUMP`).
3. **PFC / Flow Control:** For RoCE fabrics, ensure lossless Priority Flow Control (PFC) and ECN congestion marking are correctly provisioned on all switch ports.

---

## 8. Seven core system design questions

### 8.1. DP versus FSDP?

In standard Distributed Data Parallelism (DDP), every GPU stores a complete replica of model parameters, gradients, and optimizer states; communication is restricted to an AllReduce of gradients at the end of backward. Memory complexity scales as $O(\text{Model Size})$, causing GPUs to run out of memory on large models.

FSDP (Fully Sharded Data Parallel) adopts ZeRO-3 principles, sharding parameters, gradients, and optimizer states equally across data-parallel ranks. Memory complexity drops to $O(\text{Model Size} / N)$. The trade-off is communication: each layer requires an `AllGather` to reconstruct weights before forward and backward passes, and a `ReduceScatter` to synchronize gradients, increasing network traffic by 50% relative to DDP.

### 8.2. Why keep TP inside a node?

Tensor Parallelism splits weight matrices within individual Transformer layers along rows and columns. Consequently, both Multi-Head Attention and MLP blocks require an AllReduce collective at their respective layer outputs.

In an 80-layer model, a single step triggers 320 AllReduce operations. These collectives sit directly on the critical execution path; downstream layers cannot begin until preceding AllReduces complete. This ultra-high message frequency and sub-microsecond latency requirement can only be satisfied by NVSwitch's $900 \text{ GB/s}$ bandwidth and $0.5\ \mu\text{s}$ latency. Routing TP across scale-out networks causes severe throughput collapse.

### 8.3. How do you reduce the PP bubble?

In Pipeline Parallelism, sequential dependencies between stages generate idle bubble time defined by $F_{\mathrm{bubble}} = \frac{p-1}{m+p-1}$. Cluster architects employ three primary mechanisms to mitigate this:

1. **Increase Microbatch Count ($m \gg p$):** Scaling microbatches to 4–8 times the stage count drops the bubble fraction to 10–15%, though it increases activation memory retention.
2. **Interleaved 1F1B:** Rather than assigning a single contiguous layer chunk, each physical GPU manages multiple virtual stages ($v$). The bubble formula scales down to $F_{\mathrm{bubble}} \approx \frac{p-1}{v \cdot m}$.
3. **DualPipe (DeepSeek-V3 Approach):** Bidirectional scheduling drives forward and backward microbatches concurrently in opposing directions, overlapping communication directly into compute bubbles.

### 8.4. Why can AllReduce be slow?

Ring or Tree AllReduce performance falling below theoretical line rate typically traces to 5 root causes:
1. **Straggler GPU:** A single GPU throttling due to thermal limits or memory ECC corrections stalls all ranks at the collective barrier.
2. **NUMA / UPI Crossing:** A GPU routing traffic through an off-socket NIC traverses high-latency CPU UPI links.
3. **Fabric Congestion and Packet Drops:** In RoCE fabrics, PFC pause storms or switch buffer exhaustion lead to head-of-line blocking.
4. **Oversubscription:** Fabric designs with 2:1 or 3:1 oversubscription choke bisection bandwidth between distinct racks.
5. **Small Message Payload:** Messages smaller than the characteristic threshold $S_{1/2}$ become latency-dominated ($\alpha$), failing to saturate link bandwidth.

### 8.5. Why is GPU scaling not linear?

Adding GPUs fails to produce linear speedups due to three immutable physical principles:
1. **Amdahl’s Law:** Serial, unparallelized execution phases (dataloader setup, optimizer updates, barrier latency) consume an increasing proportion of total step time as GPU counts rise.
2. **Interconnect Latency Steps:** In Ring AllReduce, latency hops grow as $2(N-1)\alpha$. In large clusters, network diameter and switch traversals multiply.
3. **Strong Scaling Limits:** When scaling at a fixed global batch size, computation per GPU ($T_{\mathrm{comp}}$) diminishes while communication volume ($T_{\mathrm{comm}}$) stays flat or grows, eroding the compute-to-communication ratio.

### 8.6. Why does MoE create AllToAll?

In Mixture of Experts architectures, a router network dynamically assigns tokens to specialized expert sub-networks. Because experts are distributed across GPUs (Expert Parallelism), token representations must be physically transported to their target expert devices.

This operation represents a personalized, all-to-all shuffle: every GPU transmits distinct token slices to all peer GPUs (`AllToAll Dispatch`). Following expert computation, outputs must be routed back to originating devices via a second shuffle (`AllToAll Combine`). This creates intensive fabric bisection traffic, stressing cross-rack uplinks.

### 8.7. Why must a distributed job start as a gang?

Collective communication libraries like NCCL operate as tightly coupled, synchronous distributed state machines. If a single rank fails to participate, collectives cannot progress.

If a cluster scheduler boots jobs incrementally (e.g., launching 60 out of 64 requested GPUs), the active 60 GPUs stall at the initialization barrier, consuming power and memory while starving other queue workloads. Gang Scheduling ensures that jobs launch only when 100% of required ranks are physically provisioned and ready.

---

### What questions can the reader now answer?

- How do you use the Hockney model to determine whether a collective is bound by network latency or bandwidth?
- Why does the Ring AllReduce algorithm cap per-GPU data volume at $2\frac{N-1}{N}S$ regardless of cluster size?
- In what sequence should you configure TP, PP, CP, EP, and DP for a 70B or 405B parameter model?
- What is the quantitative performance penalty of cross-NUMA GPU-to-NIC routing inside a DGX H100?
- How do you decompose a profiler trace to separate exposed network stalls from compute-hidden communications?

---

## The whole story in six lines

- Distributed training performance depends not on abstract GPU counts, but on matching tensor traffic to physical interconnect topology.
- Ring AllReduce bounds per-GPU data volume to $2S$ asymptotically, but latency hops scale with cluster diameter.
- DP partitions the batch, TP partitions layer matrices, PP partitions depth, CP partitions sequence length, and EP partitions experts.
- High-frequency TP must remain pinned to intra-node NVLink, while DP and PP span scale-out leaf-spine fabrics.
- Step-time efficiency hinges on overlapping collective gradient communication completely behind backward compute GEMMs.
- Production stability requires gang scheduling, rail-optimized network fabrics, and rigorous multi-tier acceptance testing.

---

## Glossary

- **Rank:** An independent CUDA process with a unique global identifier participating in a distributed training run.
- **Collective:** A synchronous communication operation involving all ranks in a group simultaneously (AllReduce, AllGather, etc.).
- **NVSwitch:** An on-node crossbar ASIC providing $900 \text{ GB/s}$ bidirectional all-to-all bandwidth across 8 GPUs.
- **Rail-Optimized:** A network topology wiring identical NIC indices across nodes to dedicated leaf switches, keeping intra-rail traffic single-hop.
- **ZeRO / FSDP:** Memory-efficient architectures sharding model parameters, gradients, and optimizer states across data-parallel ranks.
- **Pipeline Bubble:** Idle hardware duration caused by sequential dependencies between upstream and downstream pipeline stages.
- **Straggler:** A degraded rank that runs slower due to thermal throttling or hardware errors, blocking all ranks at collective barriers.
- **In-Network Computing (SHARP):** Offloading floating-point reduction and aggregation directly onto InfiniBand switch ASICs.
- **Gang Scheduling:** A scheduling policy that starts a job only when 100% of its requested ranks are physically available.

---

## References and verification links

- Previous article: [AI Infrastructure (2): How LLMs Use the GPU — Training, Prefill, and Decode](post.html?slug=gpu-workloads-and-production-inference)
- Next article: [AI Infrastructure (4): Diagnosing Cluster Bottlenecks — Storage, Failures, and MFU](post.html?slug=from-gpus-to-ai-supercomputers)
- **[K1] Scale-out topology:** NVIDIA, [*DGX SuperPOD Network Fabrics*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html).
- **[K2] NCCL:** NVIDIA, [*NCCL User Guide*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html).
- **[K3] Parallelism strategies:** NVIDIA, [*Megatron Core Parallelism Guide*](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) and PyTorch, [*FSDP2*](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).
- **[K4] Gang scheduling:** Kubernetes SIGs, [*Kueue All-or-Nothing Scheduling*](https://github.com/kubernetes-sigs/kueue/blob/main/site/content/en/v0.20/docs/concepts/all_or_nothing.md).
- **[K5] Communication benchmarking:** NVIDIA, [*nccl-tests*](https://github.com/NVIDIA/nccl-tests) and [*DCGM Diagnostics*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/dcgm-diagnostics.html).
