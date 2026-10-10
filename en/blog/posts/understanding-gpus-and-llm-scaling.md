A cluster of 1,024 NVIDIA H100 GPUs looks unstoppable on paper: over 1 ExaFLOP of 8-bit tensor compute, 80 Terabytes of high-bandwidth memory, and multi-terabit interconnect fabrics. Yet the moment you launch pre-training for a 70-billion-parameter foundation model, your telemetry dashboard confronts you with a sobering reality: Model FLOPs Utilization (MFU) languishes under 38%, your expensive Streaming Multiprocessors spend 40% of their clock cycles stalled on high-bandwidth memory queues, scale-out InfiniBand links are inundated with gradient synchronization bubbles, and synchronous checkpoint writes freeze the entire cluster for five minutes every few hours.

This bottleneck is not a random glitch or a software bug. It is the physical consequence of treating a modern GPU supercomputer as a uniform, monolithic computing pool. Unlike Google TPUs, which are wired into uniform 2D or 3D tori with static, synchronous communication scheduling, modern AI supercomputers are deeply hierarchical, multi-tiered systems governed by three distinct physical pillars: **Compute**, **Network**, and **Storage**. Inside each silicon die, over a hundred independent Streaming Multiprocessors compete for capacity-limited L2 caches. Inside each 8-GPU chassis, proprietary NVSwitch crossbars sustain an astonishing 900 GB/s bidirectional throughput per chip. But the moment a collective operation steps across the node boundary into the InfiniBand scale-out fabric, available bandwidth drops by nearly an order of magnitude—and if your I/O storage subsystem cannot stream tens of terabytes of checkpoints and trillions of tokens without choking host CPUs, your entire multi-million-dollar fabric grinds to a halt.

This article dissects the entire foundation model scaling stack across its three foundational pillars. We trace how individual warp schedulers dispatch arithmetic instructions on silicon, explore intra-node and inter-node network topologies from NVLink to rail-optimized InfiniBand fat-trees, analyze hardware-level collective acceleration like Mellanox SHARP, examine exabyte-scale I/O pipelines from GPUDirect Storage to the Young-Daly checkpoint formula, and quantify every trade-off with rigorous mathematical formulations and empirical benchmarks.

**In this article**

- [1. The modern AI triad: Compute, network, and storage](#1-the-modern-ai-triad-compute-network-and-storage)
- [2. Pillar I — Compute: Streaming Multiprocessors, microarchitecture, and execution dynamics](#2-pillar-i-compute-streaming-multiprocessors-microarchitecture-and-execution-dynamics)
- [3. Pillar II — Network: Interconnect fabrics, collectives, and multi-dimensional parallelism](#3-pillar-ii-network-interconnect-fabrics-collectives-and-multi-dimensional-parallelism)
- [4. Pillar III — Storage: Foundation model datasets, checkpointing, and high-throughput I/O](#4-pillar-iii-storage-foundation-model-datasets-checkpointing-and-high-throughput-io)
- [5. Synthesis: Training vs inference and operational trade-offs](#5-synthesis-training-vs-inference-and-operational-trade-offs)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. The modern AI triad: Compute, network, and storage

Distributed foundation model training is an exercise in extreme physical co-design. When engineers attempt to scale a large language model across thousands of accelerators, they inevitably crash into three physical walls:

1. **The Memory Wall (Compute):** While raw floating-point arithmetic throughput has exploded by more than $1,000\times$ over the past decade, high-bandwidth memory (HBM) speeds have grown by less than $30\times$. Inside each GPU, arithmetic units starve unless operational matrix dimensions provide enough data reuse to keep computational intensity above the hardware roofline knee.
2. **The Fabric Wall (Network):** Communication bandwidth does not scale linearly with cluster size. Moving activations and weight gradients across an 8-GPU chassis over NVLink is an order of magnitude faster than routing packets across optical InfiniBand or RoCE v2 fabrics spanning hundreds of server racks. Failing to align model dimensions with this interconnect hierarchy results in crippling communication stalls.
3. **The I/O Wall (Storage):** Foundation models are trained on 15+ trillion tokens of unstructured text and multimodal data, while their parameter states and optimizer buffers consume 7.5+ Terabytes of memory for a 405-billion parameter checkpoint. If storage access relies on legacy operating system page caches and CPU memory copies, host memory buses saturate, causing cluster-wide GPU compute lockouts during every checkpoint interval.

To eliminate idle hardware cycles and maximize Model FLOPs Utilization (MFU), an AI infrastructure engineer must master all three pillars simultaneously:

<svg viewBox="0 0 840 8611" role="img" aria-label="A five-part poster of AI infrastructure. Part A zooms out in four levels. Level 1, the Streaming Multiprocessor: four identical subpartitions, each with a Tensor Core doing 1,024 BF16 FLOPs per cycle, a warp scheduler for 32-thread warps, 32 FP32 ALUs and 16,384 registers, plus 256 KiB of SMEM or L1; like a line cook and their counter. 132 SMs make one H100 SXM5 GPU: an 814 square millimetre die with 80 billion transistors, a 50 MB L2, and 80 GB of HBM3 at 3.35 TB/s in 5 of 6 stack sites, 990 TFLOP/s BF16 at 700 W; like a kitchen whose pantry is HBM. 8 GPUs make one DGX H100 node: 4 NVSwitches, 900 GB/s NVLink per GPU, 3.6 TB/s bisection, 2 Xeon CPUs with 2 TB RAM on PCIe Gen5 at 64 GB/s, one 400 Gb/s compute NIC per GPU, 2 dual-port NICs for storage and management, and 8 NVMe drives. 32 nodes make a scalable unit and 4 make a DGX SuperPOD of 127 nodes, 1,016 GPUs: 32 leaf and 16 spine switches in a rail-optimised fat-tree where NIC i of every node goes to leaf i, a separate storage fabric with at least 40 GB/s per node, and in-band and out-of-band Ethernet management. Part B follows one training step of a Llama 3 405B layout, TP 8 by PP 16 by DP 128: 1 read 16 million tokens from storage into HBM; 2 forward pass from HBM into the Tensor Cores; 3 tensor-parallel AllReduce over NVLink, twice per layer; 4 send activations to the next pipeline stage over the network; 5 backward pass along the same links; 6 data-parallel gradient AllReduce through rail switches to the other 127 replicas; 7 AdamW optimizer step in HBM; 8 every 14 minutes, write a 7.5 TB checkpoint to storage. Part C shows three walls with computed curves. Memory wall: for a Llama 3 70B MLP on H100, achieved speed rises with batch size from 3.3 TFLOP/s at batch 1 to 420 at batch 128 and reaches the 990 ceiling near batch 310; fixes are bigger batches, kernel fusion and low-bit weights for decode. Network wall: bandwidth per GPU falls from 3.35 TB/s in HBM to 450 GB/s on NVLink and 50 GB/s off the node; tensor parallelism fits 13 GPUs on NVLink so 8, but only 1.45 over InfiniBand; a 16-stage pipeline idles 48 percent with 16 microbatches and 10.5 percent with 128; fixes are TP in the node, SHARP and overlapping communication. I/O wall: Young-Daly loss curves against checkpoint interval, with a minimum of 24 percent at 42.9 minutes for 25 GB/s and 7.5 percent at 13.6 minutes for 250 GB/s; fixes are GPUDirect Storage, asynchronous writes and the square-root interval. Part D times the same 140 GB load: 42 milliseconds from HBM, 0.31 s over NVLink, 2.2 s over PCIe, 2.8 s over InfiniBand per GPU, 5.6 s for legacy storage shared by the whole cluster; decode on 8 H100s is capped near 190 tokens per second. Part E lists Meta&#x27;s Llama 3 405B run: 16,384 H100s at 400 TFLOP/s and 41 percent MFU, a RoCE network with 3,072-GPU pods, Tectonic storage of 240 PB with 2 TB/s sustained and 7 TB/s peak, and 466 interruptions in 54 days, about one every 3 hours. The slowest link sets the pace of the whole cluster. After the cluster level, a card lists the SuperPOD&#x27;s building blocks and why each exists: 127 DGX H100 systems do the maths; 48 QM9700 switches form the compute fabric with a rail per GPU; 16 more form a separate storage fabric of at least 40 GB/s per node; 4 UFM appliances monitor the InfiniBand fabric; 8 SN4600C and 8 SN2201 switches carry in-band and out-of-band management; 5 management servers run the control software; 38 racks with 108 PDUs deliver about 1.3 MW. Software: Base Command Manager provisions nodes, Slurm schedules jobs, Magnum IO moves data with NCCL, GPUDirect and SHARP, NGC ships matched containers, Base Command Platform serves data scientists, and AI Enterprise adds supported tools. The node level redraws NVIDIA&#x27;s DGX H100 system topology: CPU 0 and CPU 1 joined by UPI; CPU 0 feeds GPUs 0 to 3 and NICs 0 to 3, CPU 1 feeds GPUs 4 to 7 and NICs 4 to 7, each GPU paired with its own network card; two upper PCIe switches carry the dual-port storage cards and four NVMe drives each; CPU 0 has two M.2 OS drives and CPU 1 a 100 GbE card; every GPU sends 18 NVLinks, split 5, 4, 4 and 5, to four NVSwitches, which a lower PCIe switch connects to CPU 0 for management. Six numbered circles on the node diagram match six questions below it: who feeds whom (two NUMA halves), how a GPU reaches the network (its own NIC with GPUDirect RDMA), how GPUs talk (18 NVLinks to four NVSwitches), where storage traffic goes (upper PCIe switches), what joins the halves (CPU communication, UPI) and who manages the NVSwitches (Fabric Manager over PCIe from CPU 0). The cluster level opens with the hierarchy H100 SXM5, DGX H100, scalable unit of 32 DGX, and DGX SuperPOD of 4 units. A closer-look card counts the ports: each 64-port leaf sends 32 ports down to nodes and 32 up to spines, 1,024 leaf-spine cables in all, so the fabric is non-blocking; same-rail traffic needs one leaf, cross-rail traffic crosses a spine unless NCCL&#x27;s PXN first moves it over NVLink; all GPUs can send 50.8 TB/s against 51.2 TB/s of spine capacity; from 16 units a core tier is added, and 2,048 DGX use 512 leaf, 512 spine and 256 core switches. A note explains why the SuperPOD has 127 DGX rather than 128: a pair of UFM appliances takes one DGX slot, and smaller builds start from a single unit, between 31 and 127 DGX. The cluster drawing follows NVIDIA&#x27;s compute-fabric diagram: Spine-01 to Spine-16 above; SU 1 with Leaf-01-01 to 08 and nodes H100-001 to 032; SU 2 with Leaf-02-01 to 08 and nodes H100-033 to 064; SU 3, elided, repeats the pattern; SU 4 with Leaf-04-01 to 08, nodes H100-097 to 127 and the UFM pair C-UFM 1 and 2; spine-leaf links of 2 × 400 Gb/s, DGX-leaf links of 400 Gb/s and UFM-leaf links of 200 Gb/s. Under the hierarchy, a what-does-what table explains each level: the H100 does the maths and holds its share of the model in HBM; the DGX joins 8 GPUs into one NVLink domain where tensor parallelism lives, while its CPUs run the OS and data loader; the SU is the repeating unit of 32 DGX and 8 leaf switches where same-numbered GPUs meet in one hop; the SuperPOD joins the SUs through the spine, adds storage, management and software, and spreads data parallelism over the whole cluster. A definitions box explains leaf and spine: a leaf is the lower-tier switch that servers plug into, 8 per SU, one per rail, with 32 ports down and 32 up; a spine is the upper-tier switch that joins leaves and has no servers, 16 in all, each linked to all 32 leaves; since every leaf reaches every spine, any two servers meet in at most leaf, spine, leaf." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bp-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bp-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="bp-wrn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="bp-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<text x="16" y="27" text-anchor="start" style="fill:var(--c-accent);font-size:17px;font-weight:700">The Big Picture: The 3 Pillars of AI Infrastructure</text>
<text x="16" y="48" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">From inside one H100 to a 16,384-H100 cluster: where the parts are, how data flows, where it stalls</text>
<rect x="16" y="60" width="808" height="197" rx="10" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="40" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="108.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">A · Parts</text>
<text x="108.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SM → cluster</text>
<line x1="178.0" y1="96.0" x2="194.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="196" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="264.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">B · One step</text>
<text x="264.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">data&#x27;s path</text>
<line x1="334.0" y1="96.0" x2="350.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="352" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="420.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">C · Walls</text>
<text x="420.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">why it stalls</text>
<line x1="490.0" y1="96.0" x2="506.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="508" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="576.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">D · Speed</text>
<text x="576.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">intuition</text>
<line x1="646.0" y1="96.0" x2="662.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="664" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="732.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">E · Llama 3</text>
<text x="732.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">a real cluster</text>
<text x="40" y="144" text-anchor="start" style="fill:var(--c-text);font-size:12px">First we meet the parts, then we watch them work together in one training step. After that we see where they stall, build a</text>
<text x="40" y="161" text-anchor="start" style="fill:var(--c-text);font-size:12px">feel for the speeds, and check it all against a real cluster.</text>
<rect x="28.0" y="175.0" width="784.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Reference system: H100 SXM5 → DGX H100 → DGX SuperPOD (InfiniBand). Every main number belongs to it; other GPUs</text>
<text x="40" y="211" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">and other clusters appear only in parentheses.</text>
<rect x="40.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="56" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">compute</text>
<rect x="116.1" y="236" width="11" height="11" rx="2" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="132.1" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">network</text>
<rect x="192.2" y="236" width="11" height="11" rx="2" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="208.2" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">storage</text>
<rect x="268.3" y="236" width="11" height="11" rx="2" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="284.3" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">wall / bottleneck</text>
<rect x="398.9" y="236" width="11" height="11" rx="2" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="414.9" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">fix</text>
<rect x="16" y="269" width="808" height="5562" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="293" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="297.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">A</text>
<text x="56" y="298" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Zoom out: from one SM to the whole cluster</text>
<text x="28" y="319" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Each level is ×N copies of the one before; the dark highlight marks that previous level.</text>
<rect x="28" y="331" width="784" height="260" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="355" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">1 · SM: the GPU&#x27;s basic worker</text>
<rect x="40" y="367" width="228" height="152" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="48" y="383" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">SM</text>
<rect x="48" y="391" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="56" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">subpartition 0</text>
<rect x="56.0" y="413.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="100.0" y="426.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="156" y="391" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="164" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">subpartition 1</text>
<rect x="164.0" y="413.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="208.0" y="426.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48" y="441" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="56" y="456" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">subpartition 2</text>
<rect x="56.0" y="463.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="100.0" y="476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="156" y="441" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="164" y="456" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">subpartition 3</text>
<rect x="164.0" y="463.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="208.0" y="476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48.0" y="493.0" width="212.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.0" y="507.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SMEM / L1 · 256 KiB</text>
<rect x="288" y="367" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="389" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">4</text>
<text x="298" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">subpartitions</text>
<rect x="416" y="367" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="389" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1,024</text>
<text x="426" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FLOP/cycle (TC)</text>
<rect x="288" y="423" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="445" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">256 KiB</text>
<text x="298" y="462" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">registers / SM</text>
<rect x="416" y="423" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="445" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">64</text>
<text x="426" y="462" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">warps at once</text>
<text x="288" y="493" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">each subpartition: 1 Tensor Core,</text>
<text x="288" y="509" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1 warp scheduler, 32 FP32 ALUs,</text>
<text x="288" y="525" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16,384 registers · 256 KiB SMEM/L1</text>
<text x="288" y="541" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">warp = 32 threads running together</text>
<text x="552" y="381" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Why it matters</text>
<text x="552" y="399" text-anchor="start" style="fill:var(--c-text);font-size:12px">While one warp waits for memory, the</text>
<text x="552" y="416" text-anchor="start" style="fill:var(--c-text);font-size:12px">scheduler switches straight to another.</text>
<text x="552" y="433" text-anchor="start" style="fill:var(--c-text);font-size:12px">That is how SMs hide waiting time, but</text>
<text x="552" y="450" text-anchor="start" style="fill:var(--c-text);font-size:12px">if every warp is waiting, there is</text>
<text x="552" y="467" text-anchor="start" style="fill:var(--c-text);font-size:12px">nothing left to hide it with.</text>
<line x1="540.0" y1="367.0" x2="540.0" y2="519.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="559.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="574" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Kitchen analogy: an SM is a line cook; registers and SMEM are the cook&#x27;s counter.</text>
<line x1="420.0" y1="594.0" x2="420.0" y2="616.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="609" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 132 SMs = 1 GPU</text>
<rect x="28" y="619" width="784" height="276" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="643" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">2 · GPU: the H100 SXM5 chip</text>
<rect x="40" y="655" width="228" height="180" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="46.0" y="663.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="692.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="46.0" y="719.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="748.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="46.0" y="775.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="804.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="663.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="248.0" y="692.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="719.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="248.0" y="748.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="775.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.04;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="80" y="661" width="148" height="168" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="88" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.95"/>
<rect x="100" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88.0" y="735.0" width="129.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152.5" y="748.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2 · 50 MB</text>
<rect x="288" y="655" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="677" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">132</text>
<text x="298" y="694" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SMs</text>
<rect x="416" y="655" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="677" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">990</text>
<text x="426" y="694" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (BF16)</text>
<rect x="288" y="711" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="733" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">80 GB</text>
<text x="298" y="750" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM3 memory</text>
<rect x="416" y="711" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="733" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">3.35 TB/s</text>
<text x="426" y="750" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM speed</text>
<text x="288" y="781" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">814 mm² die · 80 billion transistors</text>
<text x="288" y="797" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">6 HBM sites, 5 active · 700 W</text>
<text x="288" y="813" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">shared L2: 50 MB (~5.5 TB/s)</text>
<text x="288" y="829" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FP8: 1,980 TFLOP/s (dense)</text>
<text x="288" y="845" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(B200: 148 SMs · 192 GB · 8 TB/s)</text>
<text x="552" y="669" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Why it matters</text>
<text x="552" y="687" text-anchor="start" style="fill:var(--c-text);font-size:12px">The chip can do 990 trillion operations</text>
<text x="552" y="704" text-anchor="start" style="fill:var(--c-text);font-size:12px">a second but read only 3.35 TB a second</text>
<text x="552" y="721" text-anchor="start" style="fill:var(--c-text);font-size:12px">from memory. Unless each byte read feeds</text>
<text x="552" y="738" text-anchor="start" style="fill:var(--c-text);font-size:12px">~295 operations, the Tensor Cores sit</text>
<text x="552" y="755" text-anchor="start" style="fill:var(--c-text);font-size:12px">idle.</text>
<line x1="540.0" y1="655.0" x2="540.0" y2="835.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="863.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="878" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">132 line cooks share one kitchen; HBM is the pantry, L2 the shared shelf.</text>
<line x1="420.0" y1="898.0" x2="420.0" y2="920.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="913" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 8 GPUs = 1 node</text>
<rect x="28" y="923" width="784" height="1611" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="947" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">3 · Node: inside the DGX H100</text>
<rect x="36" y="959" width="488" height="418" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="44.0" y="971.0" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="76.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="76.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 data</text>
<rect x="130.0" y="971.0" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="158.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="158.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="206.0" y="971.0" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="239.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="239.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dual-port</text>
<rect x="288.0" y="971.0" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="321.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="321.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dual-port</text>
<rect x="374.0" y="971.0" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="402.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="402.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="452.0" y="971.0" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="484.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="484.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 data</text>
<line x1="109.0" y1="989.0" x2="128.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="187.0" y1="989.0" x2="204.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="356.0" y1="989.0" x2="372.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="432.0" y1="989.0" x2="450.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="158.0" y1="1009.0" x2="158.0" y2="1025.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="402.0" y1="1009.0" x2="402.0" y2="1025.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="120.0" y="1027.0" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="158" y="1047" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 0</text>
<text x="158" y="1063" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="364.0" y="1027.0" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="402" y="1047" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="402" y="1063" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="210.0" y="1029.0" width="62.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="241.0" y="1043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 × M.2 OS</text>
<line x1="198.0" y1="1039.0" x2="208.0" y2="1039.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="288.0" y="1029.0" width="62.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="319.0" y="1043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<line x1="352.0" y1="1039.0" x2="362.0" y2="1039.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="199.0" y1="1061.0" x2="361.0" y2="1061.0" marker-start="url(#bp-acc)" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<path d="M196 1071 H280 V1293" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="40" y="1099" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="286" y="1099" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<path d="M132 1075 V1083 H69 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M148 1075 V1090 H127 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M164 1075 V1090 H185 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M180 1075 V1083 H243 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M376 1075 V1083 H317 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M392 1075 V1090 H375 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M408 1075 V1090 H433 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M424 1075 V1083 H491 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="46.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="69.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0</text>
<line x1="69.0" y1="1133.0" x2="69.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="44.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.34;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="69.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<rect x="104.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="127.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 1</text>
<line x1="127.0" y1="1133.0" x2="127.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="102.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="127.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<rect x="162.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="185.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 2</text>
<line x1="185.0" y1="1133.0" x2="185.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="160.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="185.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<rect x="220.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="243.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 3</text>
<line x1="243.0" y1="1133.0" x2="243.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="218.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="243.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<rect x="294.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="317.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 4</text>
<line x1="317.0" y1="1133.0" x2="317.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="292.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 4</text>
<rect x="352.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="375.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 5</text>
<line x1="375.0" y1="1133.0" x2="375.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="350.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="375.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 5</text>
<rect x="410.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="433.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 6</text>
<line x1="433.0" y1="1133.0" x2="433.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="408.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="433.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 6</text>
<rect x="468.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="491.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 7</text>
<line x1="491.0" y1="1133.0" x2="491.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="466.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="491.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 7</text>
<line x1="49.0" y1="1183.0" x2="49.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="54.0" y1="1183.0" x2="54.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="59.0" y1="1183.0" x2="59.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="64.0" y1="1183.0" x2="64.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="69.0" y1="1183.0" x2="69.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="74.0" y1="1183.0" x2="74.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="79.0" y1="1183.0" x2="79.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="84.0" y1="1183.0" x2="84.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="89.0" y1="1183.0" x2="89.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="107.0" y1="1183.0" x2="107.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="112.0" y1="1183.0" x2="112.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="117.0" y1="1183.0" x2="117.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="122.0" y1="1183.0" x2="122.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="127.0" y1="1183.0" x2="127.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="132.0" y1="1183.0" x2="132.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="137.0" y1="1183.0" x2="137.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="142.0" y1="1183.0" x2="142.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="147.0" y1="1183.0" x2="147.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="165.0" y1="1183.0" x2="165.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="170.0" y1="1183.0" x2="170.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="175.0" y1="1183.0" x2="175.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="180.0" y1="1183.0" x2="180.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="185.0" y1="1183.0" x2="185.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="190.0" y1="1183.0" x2="190.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="195.0" y1="1183.0" x2="195.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="200.0" y1="1183.0" x2="200.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="205.0" y1="1183.0" x2="205.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="223.0" y1="1183.0" x2="223.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="228.0" y1="1183.0" x2="228.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="233.0" y1="1183.0" x2="233.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="238.0" y1="1183.0" x2="238.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="243.0" y1="1183.0" x2="243.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="248.0" y1="1183.0" x2="248.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="253.0" y1="1183.0" x2="253.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="258.0" y1="1183.0" x2="258.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="263.0" y1="1183.0" x2="263.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="297.0" y1="1183.0" x2="297.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="302.0" y1="1183.0" x2="302.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="307.0" y1="1183.0" x2="307.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="312.0" y1="1183.0" x2="312.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="317.0" y1="1183.0" x2="317.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="322.0" y1="1183.0" x2="322.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="327.0" y1="1183.0" x2="327.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="332.0" y1="1183.0" x2="332.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="337.0" y1="1183.0" x2="337.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="355.0" y1="1183.0" x2="355.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="360.0" y1="1183.0" x2="360.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="365.0" y1="1183.0" x2="365.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="370.0" y1="1183.0" x2="370.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="375.0" y1="1183.0" x2="375.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="380.0" y1="1183.0" x2="380.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="385.0" y1="1183.0" x2="385.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="390.0" y1="1183.0" x2="390.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="395.0" y1="1183.0" x2="395.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="413.0" y1="1183.0" x2="413.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="418.0" y1="1183.0" x2="418.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="423.0" y1="1183.0" x2="423.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="428.0" y1="1183.0" x2="428.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="433.0" y1="1183.0" x2="433.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="438.0" y1="1183.0" x2="438.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="443.0" y1="1183.0" x2="443.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="448.0" y1="1183.0" x2="448.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="453.0" y1="1183.0" x2="453.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="471.0" y1="1183.0" x2="471.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="476.0" y1="1183.0" x2="476.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="481.0" y1="1183.0" x2="481.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="486.0" y1="1183.0" x2="486.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="491.0" y1="1183.0" x2="491.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="496.0" y1="1183.0" x2="496.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="501.0" y1="1183.0" x2="501.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="506.0" y1="1183.0" x2="506.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="511.0" y1="1183.0" x2="511.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="44.0" y1="1221.0" x2="516.0" y2="1221.0" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<line x1="78.0" y1="1221.0" x2="78.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="84.0" y1="1221.0" x2="84.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="90.0" y1="1221.0" x2="90.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="96.0" y1="1221.0" x2="96.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="102.0" y1="1221.0" x2="102.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="108.0" y1="1221.0" x2="108.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="114.0" y1="1221.0" x2="114.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="52.0" y="1243.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="1260.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 0</text>
<line x1="182.0" y1="1221.0" x2="182.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="188.0" y1="1221.0" x2="188.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="194.0" y1="1221.0" x2="194.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="200.0" y1="1221.0" x2="200.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="206.0" y1="1221.0" x2="206.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="212.0" y1="1221.0" x2="212.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="218.0" y1="1221.0" x2="218.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="156.0" y="1243.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="200.0" y="1260.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 1</text>
<line x1="342.0" y1="1221.0" x2="342.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="348.0" y1="1221.0" x2="348.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="354.0" y1="1221.0" x2="354.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="360.0" y1="1221.0" x2="360.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="366.0" y1="1221.0" x2="366.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="372.0" y1="1221.0" x2="372.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="378.0" y1="1221.0" x2="378.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="316.0" y="1243.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="360.0" y="1260.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 2</text>
<line x1="446.0" y1="1221.0" x2="446.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="452.0" y1="1221.0" x2="452.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="458.0" y1="1221.0" x2="458.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="464.0" y1="1221.0" x2="464.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="470.0" y1="1221.0" x2="470.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="476.0" y1="1221.0" x2="476.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="482.0" y1="1221.0" x2="482.0" y2="1243.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="420.0" y="1243.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="464.0" y="1260.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 3</text>
<rect x="250.0" y="1295.0" width="60.0" height="30" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="280" y="1308" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="280" y="1321" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<path d="M96 1269 V1317 H248" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M200 1269 V1304 H248" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M360 1269 V1304 H312" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M464 1269 V1317 H312" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="280" y="1341" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">PCIe: CPU 0 → NVSwitches (management)</text>
<line x1="48.0" y1="1361.0" x2="72.0" y2="1361.0" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="78" y="1365" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe</text>
<line x1="126.8" y1="1361.0" x2="150.8" y2="1361.0" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="156.8" y="1365" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink</text>
<line x1="218.0" y1="1361.0" x2="242.0" y2="1361.0" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="248.0" y="1365" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">CPU communication (UPI)</text>
<circle cx="102" cy="1051" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="102" y="1055" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<circle cx="52" cy="1087" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="52" y="1091" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="98" cy="1207" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="98" y="1211" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="239" cy="1018" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="239" y="1022" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="280" cy="1061" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="1065" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<circle cx="320" cy="1282" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="320" y="1286" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="540" y="959" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="981" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">8</text>
<text x="550" y="998" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs · 640 GB</text>
<rect x="674" y="959" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="981" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">450 GB/s</text>
<text x="684" y="998" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink, one way</text>
<rect x="540" y="1015" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="1037" text-anchor="start" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">64 GB/s</text>
<text x="550" y="1054" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe, CPU path</text>
<rect x="674" y="1015" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="1037" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">50 GB/s</text>
<text x="684" y="1054" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">network / GPU</text>
<text x="540" y="1085" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 NVSwitch · 3.6 TB/s bisection · 2 × Xeon</text>
<text x="540" y="1100" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8480C (112 cores), 2 TB RAM · 8 × 3.84 TB NVMe</text>
<text x="540" y="1115" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">data · power 10.2 kW</text>
<rect x="540.0" y="1133.0" width="260.0" height="194" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="1155" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Worked path: one gradient&#x27;s trip</text>
<text x="552" y="1172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">through the node</text>
<text x="552" y="1195" text-anchor="start" style="fill:var(--c-text);font-size:12px">A gradient from the neighbouring node</text>
<text x="552" y="1212" text-anchor="start" style="fill:var(--c-text);font-size:12px">enters NIC 5 and, via GPUDirect RDMA,</text>
<text x="552" y="1229" text-anchor="start" style="fill:var(--c-text);font-size:12px">is written straight into GPU 5&#x27;s</text>
<text x="552" y="1246" text-anchor="start" style="fill:var(--c-text);font-size:12px">memory without touching CPU 1. If GPU</text>
<text x="552" y="1263" text-anchor="start" style="fill:var(--c-text);font-size:12px">5 then shares it with GPU 2, it</text>
<text x="552" y="1280" text-anchor="start" style="fill:var(--c-text);font-size:12px">travels over NVLink through the</text>
<text x="552" y="1297" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitches, not over the network or</text>
<text x="552" y="1314" text-anchor="start" style="fill:var(--c-text);font-size:12px">through the CPUs.</text>
<rect x="36" y="1389" width="768" height="502" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="1411" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Legend: what is each part?</text>
<text x="48" y="1429" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Names come from NVIDIA&#x27;s diagram legend; colours are the poster&#x27;s pillar colours. Roles come from the DGX H100 user guide, and</text>
<text x="48" y="1444" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">those marked “likely” are inferences.</text>
<rect x="48.0" y="1462.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CX-7</text>
<text x="126" y="1476" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7” · top, 2 cards</text>
<text x="126" y="1492" text-anchor="start" style="fill:var(--c-text);font-size:11px">Dual-port network card; per the user guide, for</text>
<text x="126" y="1507" text-anchor="start" style="fill:var(--c-text);font-size:11px">the storage fabric and in-band management. Each</text>
<text x="126" y="1522" text-anchor="start" style="fill:var(--c-text);font-size:11px">sits on an upper PCIe switch.</text>
<rect x="48.0" y="1536.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="1550.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0–7</text>
<text x="126" y="1550" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7 Network Module” · 8</text>
<text x="126" y="1566" text-anchor="start" style="fill:var(--c-text);font-size:11px">Single-port 400 Gb/s compute-fabric cards, one per</text>
<text x="126" y="1581" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU; they leave the node through 4 OSFP ports. The</text>
<text x="126" y="1596" text-anchor="start" style="fill:var(--c-text);font-size:11px">grey frames are the modules that carry them</text>
<text x="126" y="1611" text-anchor="start" style="fill:var(--c-text);font-size:11px">(likely).</text>
<rect x="48.0" y="1625.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1639.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe</text>
<text x="126" y="1639" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · top, 4 + 4</text>
<text x="126" y="1655" text-anchor="start" style="fill:var(--c-text);font-size:11px">Data drives: 8 × 3.84 TB U.2 NVMe as a RAID 0</text>
<text x="126" y="1670" text-anchor="start" style="fill:var(--c-text);font-size:11px">cache, local space for reads and checkpoints.</text>
<rect x="48.0" y="1684.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1698.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">M.2 OS</text>
<text x="126" y="1698" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · next to CPU 0</text>
<text x="126" y="1714" text-anchor="start" style="fill:var(--c-text);font-size:11px">Likely the 2 × 1.92 TB M.2 operating-system drives</text>
<text x="126" y="1729" text-anchor="start" style="fill:var(--c-text);font-size:11px">(RAID 1); the diagram uses the same icon as the</text>
<text x="126" y="1744" text-anchor="start" style="fill:var(--c-text);font-size:11px">data drives.</text>
<rect x="48.0" y="1758.0" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="1772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe sw</text>
<text x="126" y="1772" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe Switches” · 3</text>
<text x="126" y="1788" text-anchor="start" style="fill:var(--c-text);font-size:11px">Fan one CPU link out to many devices: the upper</text>
<text x="126" y="1803" text-anchor="start" style="fill:var(--c-text);font-size:11px">two carry NVMe and CX-7, the lower one links the</text>
<text x="126" y="1818" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVSwitches to CPU 0.</text>
<rect x="48.0" y="1832.0" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="1846.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<text x="126" y="1846" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “100 GbE” · on CPU 1</text>
<text x="126" y="1862" text-anchor="start" style="fill:var(--c-text);font-size:11px">An extra Ethernet card; the diagram does not say</text>
<text x="126" y="1877" text-anchor="start" style="fill:var(--c-text);font-size:11px">what for (likely a service or management network).</text>
<rect x="432.0" y="1462.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="465.0" y="1476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVSwitch</text>
<text x="510" y="1476" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVSwitch” · 4</text>
<text x="510" y="1492" text-anchor="start" style="fill:var(--c-text);font-size:11px">The NVLink switch between GPUs; every GPU spreads</text>
<text x="510" y="1507" text-anchor="start" style="fill:var(--c-text);font-size:11px">its 18 links over these four as 5 + 4 + 4 + 5.</text>
<rect x="432.0" y="1521.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="1535.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CPU</text>
<text x="510" y="1535" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU 0 / CPU 1”</text>
<text x="510" y="1551" text-anchor="start" style="fill:var(--c-text);font-size:11px">Intel Xeon 8480C (56 cores each); each feeds four</text>
<text x="510" y="1566" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPUs and their NICs over PCIe.</text>
<line x1="432.0" y1="1590.0" x2="498.0" y2="1590.0" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="510" y="1594" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU communication”</text>
<text x="510" y="1610" text-anchor="start" style="fill:var(--c-text);font-size:11px">The link that joins the two sockets; on Intel Xeon</text>
<text x="510" y="1625" text-anchor="start" style="fill:var(--c-text);font-size:11px">this link is called UPI.</text>
<line x1="432.0" y1="1649.0" x2="498.0" y2="1649.0" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="510" y="1653" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe”</text>
<text x="510" y="1669" text-anchor="start" style="fill:var(--c-text);font-size:11px">The general road from the CPUs to NICs, GPUs,</text>
<text x="510" y="1684" text-anchor="start" style="fill:var(--c-text);font-size:11px">drives and NVSwitches (PCIe Gen5).</text>
<line x1="432.0" y1="1708.0" x2="498.0" y2="1708.0" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="510" y="1712" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Missing from NVIDIA&#x27;s legend: NVLink</text>
<text x="510" y="1728" text-anchor="start" style="fill:var(--c-text);font-size:11px">The coloured bundles below the GPUs are NVLinks:</text>
<text x="510" y="1743" text-anchor="start" style="fill:var(--c-text);font-size:11px">18 links per GPU, 900 GB/s both ways in total.</text>
<line x1="420.0" y1="1458.0" x2="420.0" y2="1877.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1917" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Reading the topology: six questions</text>
<text x="40" y="1936" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Follow the numbered circles on the diagram in order; each one answers a question.</text>
<circle cx="49" cy="1962" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="1966" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="1966" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Who feeds whom?</text>
<text x="66" y="1986" text-anchor="start" style="fill:var(--c-text);font-size:12px">The node is two mirror-image halves. CPU 0 feeds GPUs</text>
<text x="66" y="2003" text-anchor="start" style="fill:var(--c-text);font-size:12px">0–3 and NICs 0–3 over PCIe; CPU 1 feeds GPUs 4–7 and</text>
<text x="66" y="2020" text-anchor="start" style="fill:var(--c-text);font-size:12px">NICs 4–7. Each half has its own memory, which is called</text>
<text x="66" y="2037" text-anchor="start" style="fill:var(--c-text);font-size:12px">a NUMA domain.</text>
<text x="66" y="2056" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> the process that prepares data for GPU 5</text>
<text x="66" y="2073" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">should run on CPU 1 and use CPU 1&#x27;s memory; otherwise</text>
<text x="66" y="2090" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">every batch first crosses the link between the CPUs.</text>
<circle cx="49" cy="2116" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2120" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="2120" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">How does a GPU reach the network?</text>
<text x="66" y="2140" text-anchor="start" style="fill:var(--c-text);font-size:12px">Right above every GPU sits its own 400 Gb/s NIC (GPU i</text>
<text x="66" y="2157" text-anchor="start" style="fill:var(--c-text);font-size:12px">↔ NIC i). With GPUDirect RDMA, data goes from GPU</text>
<text x="66" y="2174" text-anchor="start" style="fill:var(--c-text);font-size:12px">memory straight to the NIC and never stops in CPU</text>
<text x="66" y="2191" text-anchor="start" style="fill:var(--c-text);font-size:12px">memory.</text>
<text x="66" y="2210" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> 8 × 50 = 400 GB/s leave the chassis. In</text>
<text x="66" y="2227" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">the cluster NIC i plugs into leaf switch i, which is</text>
<text x="66" y="2244" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">why same-numbered GPUs on different nodes meet in a</text>
<text x="66" y="2261" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">single hop.</text>
<circle cx="49" cy="2287" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2291" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="2291" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">How do GPUs talk to each other?</text>
<text x="66" y="2311" text-anchor="start" style="fill:var(--c-text);font-size:12px">Each GPU&#x27;s 18 NVLinks run down to the four NVSwitches:</text>
<text x="66" y="2328" text-anchor="start" style="fill:var(--c-text);font-size:12px">5 links to two of them and 4 to the other two. Any two</text>
<text x="66" y="2345" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPUs talk through a single switch hop at 450 GB/s each</text>
<text x="66" y="2362" text-anchor="start" style="fill:var(--c-text);font-size:12px">way.</text>
<text x="66" y="2381" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> this road is 9× wider than the network, so</text>
<text x="66" y="2398" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">tensor parallelism, which talks in every layer, stays</text>
<text x="66" y="2415" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">inside these eight GPUs.</text>
<circle cx="441" cy="1962" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="1966" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="458" y="1966" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Where does storage traffic go?</text>
<text x="458" y="1986" text-anchor="start" style="fill:var(--c-text);font-size:12px">The two upper PCIe switches carry the dual-port CX-7</text>
<text x="458" y="2003" text-anchor="start" style="fill:var(--c-text);font-size:12px">cards (storage fabric and in-band management) and four</text>
<text x="458" y="2020" text-anchor="start" style="fill:var(--c-text);font-size:12px">U.2 NVMe data drives each. Two M.2 operating-system</text>
<text x="458" y="2037" text-anchor="start" style="fill:var(--c-text);font-size:12px">drives also sit next to CPU 0.</text>
<text x="458" y="2056" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> reading data and writing checkpoints use</text>
<text x="458" y="2073" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">different cards and cables from gradient traffic, so a</text>
<text x="458" y="2090" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">big checkpoint does not slow down AllReduce.</text>
<circle cx="441" cy="2116" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="2120" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="2120" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">What joins the two halves?</text>
<text x="458" y="2140" text-anchor="start" style="fill:var(--c-text);font-size:12px">The two CPUs are linked by what NVIDIA calls “CPU</text>
<text x="458" y="2157" text-anchor="start" style="fill:var(--c-text);font-size:12px">communication” (UPI on Xeon). That link carries the</text>
<text x="458" y="2174" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPUs&#x27; own traffic: GPUs in different halves talk over</text>
<text x="458" y="2191" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVLink, not through the CPUs.</text>
<text x="458" y="2210" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> a CPU can reach the other half&#x27;s memory,</text>
<text x="458" y="2227" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">but more slowly; NUMA pinning exists to avoid that</text>
<text x="458" y="2244" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">penalty.</text>
<circle cx="441" cy="2270" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="2274" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="2274" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Who manages the NVSwitches?</text>
<text x="458" y="2294" text-anchor="start" style="fill:var(--c-text);font-size:12px">A PCIe line from CPU 0 reaches the four NVSwitches</text>
<text x="458" y="2311" text-anchor="start" style="fill:var(--c-text);font-size:12px">through the lower PCIe switch. It is a control path:</text>
<text x="458" y="2328" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA Fabric Manager sets up the switches&#x27; routing</text>
<text x="458" y="2345" text-anchor="start" style="fill:var(--c-text);font-size:12px">tables through it, and no GPU data flows on it.</text>
<text x="458" y="2364" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> if Fabric Manager is not running, the</text>
<text x="458" y="2381" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">NVLink fabric never comes up and multi-GPU jobs fail to</text>
<text x="458" y="2398" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">start, even though every GPU looks healthy on its own.</text>
<line x1="420.0" y1="1950.0" x2="420.0" y2="2415.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2449" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Why it matters</text>
<text x="40" y="2467" text-anchor="start" style="fill:var(--c-text);font-size:12px">Inside the chassis, GPUs reach each other&#x27;s memory directly over NVLink. The PCIe path to the CPU is 7× narrower and the</text>
<text x="40" y="2484" text-anchor="start" style="fill:var(--c-text);font-size:12px">network path out of the chassis 9× narrower, so the busiest talk stays inside.</text>
<rect x="36.0" y="2502.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="2517" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">A restaurant with 8 kitchens: NVLink is the corridor that joins them.</text>
<line x1="420.0" y1="2537.0" x2="420.0" y2="2559.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="2552" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 32 nodes = 1 SU · × 4 SUs = SuperPOD</text>
<rect x="28" y="2562" width="784" height="1655" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2586" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4 · Cluster: the DGX SuperPOD</text>
<text x="40" y="2608" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Remember the hierarchy: from part to whole</text>
<rect x="40.0" y="2618.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="126.0" y="2632" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="126.0" y="2646" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">one GPU module</text>
<line x1="214.0" y1="2638.0" x2="234.0" y2="2638.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="236.0" y="2618.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="322.0" y="2632" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="322.0" y="2646" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8-GPU server</text>
<line x1="410.0" y1="2638.0" x2="430.0" y2="2638.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="432.0" y="2618.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="518.0" y="2632" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="518.0" y="2646" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 DGX = 256 GPUs</text>
<line x1="606.0" y1="2638.0" x2="626.0" y2="2638.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="628.0" y="2618.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="714.0" y="2632" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="714.0" y="2646" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 SUs = 127 DGX</text>
<text x="40" y="2678" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">H100 SXM5 modules plug not into the motherboard but into the HGX baseboard that carries 8 GPUs and 4 NVSwitches; the DGX H100&#x27;s two</text>
<text x="40" y="2693" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">CPUs are Intel Xeon 8480C. A SuperPOD is built from 32-DGX units (SUs); the reference design has 4 SUs.</text>
<text x="40" y="2725" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">What are leaf and spine?</text>
<rect x="40" y="2737" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="48.0" y="2745.0" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="162.0" y="2760.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Leaf switch</text>
<text x="52" y="2787" text-anchor="start" style="fill:var(--c-text);font-size:12px">The lower tier, where servers plug</text>
<text x="52" y="2804" text-anchor="start" style="fill:var(--c-text);font-size:12px">in directly; the leaves are the</text>
<text x="52" y="2821" text-anchor="start" style="fill:var(--c-text);font-size:12px">tips of the tree, nearest to the</text>
<text x="52" y="2838" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPUs.</text>
<text x="52" y="2861" text-anchor="start" style="fill:var(--c-text);font-size:12px">In the SuperPOD: 8 per SU, one per</text>
<text x="52" y="2878" text-anchor="start" style="fill:var(--c-text);font-size:12px">rail. 32 of its 64 ports go down to</text>
<text x="52" y="2895" text-anchor="start" style="fill:var(--c-text);font-size:12px">DGX nodes and 32 go up to spines.</text>
<rect x="296" y="2737" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="304.0" y="2745.0" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="418.0" y="2760.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Spine switch</text>
<text x="308" y="2787" text-anchor="start" style="fill:var(--c-text);font-size:12px">The upper tier that joins the</text>
<text x="308" y="2804" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaves to each other; no server</text>
<text x="308" y="2821" text-anchor="start" style="fill:var(--c-text);font-size:12px">ever plugs into a spine.</text>
<text x="308" y="2844" text-anchor="start" style="fill:var(--c-text);font-size:12px">In the SuperPOD: 16 in all, each</text>
<text x="308" y="2861" text-anchor="start" style="fill:var(--c-text);font-size:12px">linked to all 32 leaves with two</text>
<text x="308" y="2878" text-anchor="start" style="fill:var(--c-text);font-size:12px">lines apiece, which fills its</text>
<text x="308" y="2895" text-anchor="start" style="fill:var(--c-text);font-size:12px">64 ports.</text>
<text x="552" y="2751" text-anchor="start" style="fill:var(--c-text);font-size:12px">The rule of the two tiers is simple:</text>
<text x="552" y="2768" text-anchor="start" style="fill:var(--c-text);font-size:12px">every leaf connects to every spine. Any</text>
<text x="552" y="2785" text-anchor="start" style="fill:var(--c-text);font-size:12px">two servers therefore meet in at most</text>
<text x="552" y="2802" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf → spine → leaf, and the path does</text>
<text x="552" y="2819" text-anchor="start" style="fill:var(--c-text);font-size:12px">not get longer as the cluster grows. To</text>
<text x="552" y="2836" text-anchor="start" style="fill:var(--c-text);font-size:12px">add servers you add leaves; to add</text>
<text x="552" y="2853" text-anchor="start" style="fill:var(--c-text);font-size:12px">bandwidth you add spines. The pattern</text>
<text x="552" y="2870" text-anchor="start" style="fill:var(--c-text);font-size:12px">comes from telephone exchanges (Clos,</text>
<text x="552" y="2887" text-anchor="start" style="fill:var(--c-text);font-size:12px">1953); when the links get thicker toward</text>
<text x="552" y="2904" text-anchor="start" style="fill:var(--c-text);font-size:12px">the top it is called a fat-tree</text>
<text x="552" y="2921" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Leiserson, 1985).</text>
<text x="552" y="2943" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Analogy: a leaf is the local exchange of a</text>
<text x="552" y="2958" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">neighbourhood, a spine the trunk line</text>
<text x="552" y="2973" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">between cities.</text>
<text x="40" y="3011" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">What does what?</text>
<text x="168" y="3037" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Job</text>
<text x="384" y="3037" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Holds</text>
<text x="600" y="3037" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Links and parallelism</text>
<line x1="40.0" y1="3046.0" x2="800.0" y2="3046.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="3059.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="3075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="168" y="3075" text-anchor="start" style="fill:var(--c-text);font-size:12px">does the model&#x27;s maths; the</text>
<text x="168" y="3092" text-anchor="start" style="fill:var(--c-text);font-size:12px">Tensor Cores in its 132 SMs run</text>
<text x="168" y="3109" text-anchor="start" style="fill:var(--c-text);font-size:12px">the matrix multiplications.</text>
<text x="384" y="3075" text-anchor="start" style="fill:var(--c-text);font-size:12px">this GPU&#x27;s share of the weights,</text>
<text x="384" y="3092" text-anchor="start" style="fill:var(--c-text);font-size:12px">activations and optimizer state,</text>
<text x="384" y="3109" text-anchor="start" style="fill:var(--c-text);font-size:12px">in 80 GB of HBM3.</text>
<text x="600" y="3075" text-anchor="start" style="fill:var(--c-text);font-size:12px">18 NVLinks reach the other</text>
<text x="600" y="3092" text-anchor="start" style="fill:var(--c-text);font-size:12px">7 GPUs in the chassis, PCIe</text>
<text x="600" y="3109" text-anchor="start" style="fill:var(--c-text);font-size:12px">reaches the CPU and its own NIC;</text>
<text x="600" y="3126" text-anchor="start" style="fill:var(--c-text);font-size:12px">every GPU has a TP, a PP and a</text>
<text x="600" y="3143" text-anchor="start" style="fill:var(--c-text);font-size:12px">DP coordinate.</text>
<line x1="40.0" y1="3153.0" x2="800.0" y2="3153.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3167.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="3183.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="168" y="3183" text-anchor="start" style="fill:var(--c-text);font-size:12px">joins 8 GPUs into one NVLink</text>
<text x="168" y="3200" text-anchor="start" style="fill:var(--c-text);font-size:12px">domain; two Xeon CPUs run the</text>
<text x="168" y="3217" text-anchor="start" style="fill:var(--c-text);font-size:12px">OS, the data loader and the</text>
<text x="168" y="3234" text-anchor="start" style="fill:var(--c-text);font-size:12px">processes that send work to the</text>
<text x="168" y="3251" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPUs.</text>
<text x="384" y="3183" text-anchor="start" style="fill:var(--c-text);font-size:12px">on the HGX baseboard 8 GPUs and</text>
<text x="384" y="3200" text-anchor="start" style="fill:var(--c-text);font-size:12px">4 NVSwitches, plus 2 TB of RAM,</text>
<text x="384" y="3217" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 × 3.84 TB of NVMe cache and 8</text>
<text x="384" y="3234" text-anchor="start" style="fill:var(--c-text);font-size:12px">+ 2 network cards.</text>
<text x="600" y="3183" text-anchor="start" style="fill:var(--c-text);font-size:12px">tensor parallelism (TP 8) stays</text>
<text x="600" y="3200" text-anchor="start" style="fill:var(--c-text);font-size:12px">inside this chassis; the chassis</text>
<text x="600" y="3217" text-anchor="start" style="fill:var(--c-text);font-size:12px">reaches the network over 8</text>
<text x="600" y="3234" text-anchor="start" style="fill:var(--c-text);font-size:12px">rails, one per GPU.</text>
<line x1="40.0" y1="3261.0" x2="800.0" y2="3261.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3275.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="3291.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="168" y="3291" text-anchor="start" style="fill:var(--c-text);font-size:12px">the repeating unit of the build</text>
<text x="168" y="3308" text-anchor="start" style="fill:var(--c-text);font-size:12px">and of the network: 32 DGX</text>
<text x="168" y="3325" text-anchor="start" style="fill:var(--c-text);font-size:12px">(8 racks × 4) and the 8 leaf</text>
<text x="168" y="3342" text-anchor="start" style="fill:var(--c-text);font-size:12px">switches that join them.</text>
<text x="384" y="3291" text-anchor="start" style="fill:var(--c-text);font-size:12px">256 GPUs; each rail is one leaf,</text>
<text x="384" y="3308" text-anchor="start" style="fill:var(--c-text);font-size:12px">so Leaf-xx-0i gathers NIC i of</text>
<text x="384" y="3325" text-anchor="start" style="fill:var(--c-text);font-size:12px">every node.</text>
<text x="600" y="3291" text-anchor="start" style="fill:var(--c-text);font-size:12px">same-numbered GPUs meet in one</text>
<text x="600" y="3308" text-anchor="start" style="fill:var(--c-text);font-size:12px">hop; pipeline and data-parallel</text>
<text x="600" y="3325" text-anchor="start" style="fill:var(--c-text);font-size:12px">traffic between nodes flows here</text>
<text x="600" y="3342" text-anchor="start" style="fill:var(--c-text);font-size:12px">first, rail by rail.</text>
<line x1="40.0" y1="3352.0" x2="800.0" y2="3352.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3366.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="3382.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="168" y="3382" text-anchor="start" style="fill:var(--c-text);font-size:12px">joins the SUs into one</text>
<text x="168" y="3399" text-anchor="start" style="fill:var(--c-text);font-size:12px">non-blocking network through the</text>
<text x="168" y="3416" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine layer and adds the</text>
<text x="168" y="3433" text-anchor="start" style="fill:var(--c-text);font-size:12px">separate storage and management</text>
<text x="168" y="3450" text-anchor="start" style="fill:var(--c-text);font-size:12px">networks, UFM, and software such</text>
<text x="168" y="3467" text-anchor="start" style="fill:var(--c-text);font-size:12px">as Base Command and Slurm.</text>
<text x="384" y="3382" text-anchor="start" style="fill:var(--c-text);font-size:12px">127 DGX = 1,016 GPUs; a single</text>
<text x="384" y="3399" text-anchor="start" style="fill:var(--c-text);font-size:12px">training job can span the whole</text>
<text x="384" y="3416" text-anchor="start" style="fill:var(--c-text);font-size:12px">cluster.</text>
<text x="600" y="3382" text-anchor="start" style="fill:var(--c-text);font-size:12px">data parallelism spans the</text>
<text x="600" y="3399" text-anchor="start" style="fill:var(--c-text);font-size:12px">cluster and cross-rail traffic</text>
<text x="600" y="3416" text-anchor="start" style="fill:var(--c-text);font-size:12px">turns through a spine; Slurm</text>
<text x="600" y="3433" text-anchor="start" style="fill:var(--c-text);font-size:12px">places the job on nodes and</text>
<text x="600" y="3450" text-anchor="start" style="fill:var(--c-text);font-size:12px">takes failed nodes out of</text>
<text x="600" y="3467" text-anchor="start" style="fill:var(--c-text);font-size:12px">service.</text>
<rect x="40.0" y="3485.0" width="760.0" height="67" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="52" y="3505" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Why 127, not 128?</tspan> On paper 4 SUs × 32 = 128 DGX. In NVIDIA&#x27;s deployment pattern, however, a pair of UFM appliances takes</text>
<text x="52" y="3522" text-anchor="start" style="fill:var(--c-text);font-size:12px">one DGX&#x27;s place (C-UFM 1 and 2 in SU 4 in NVIDIA&#x27;s diagram), so a full SuperPOD holds at most 127 DGX, i.e. 1,016 GPUs.</text>
<text x="52" y="3539" text-anchor="start" style="fill:var(--c-text);font-size:12px">The cable table shows it too: 127 × 8 + 4 UFM = 1,020 cables. Smaller builds start from a single SU (31 to 127 DGX).</text>
<rect x="36" y="3564" width="768" height="392" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="214.0" y="3578.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="252.0" y="3593.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-01</text>
<rect x="318.0" y="3578.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="3593.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-02</text>
<rect x="530.0" y="3578.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="568.0" y="3593.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-16</text>
<text x="462" y="3593" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-weight:700">⋯</text>
<text x="620" y="3593" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 × QM9700</text>
<rect x="46" y="3634" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="52.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="83.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-01</text>
<rect x="120.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="151.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-02</text>
<rect x="206.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-08</text>
<text x="194" y="3661" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="89.0" y1="3720.0" x2="83.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="3720.0" x2="151.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="3720.0" x2="237.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="54" y="3720" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="89" y="3736" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-001</text>
<text x="89" y="3751" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="175.0" y1="3720.0" x2="83.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="3720.0" x2="151.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="3720.0" x2="237.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="140" y="3720" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="175" y="3736" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-032</text>
<text x="175" y="3751" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="132" y="3743" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="56" y="3798" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 1</text>
<text x="98" y="3798" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-001 … 032 · 8 leaves</text>
<rect x="290" y="3634" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="296.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="327.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-01</text>
<rect x="364.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="395.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-02</text>
<rect x="450.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="481.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-08</text>
<text x="438" y="3661" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="333.0" y1="3720.0" x2="327.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="3720.0" x2="395.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="3720.0" x2="481.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="298" y="3720" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="333" y="3736" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-033</text>
<text x="333" y="3751" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="419.0" y1="3720.0" x2="327.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="3720.0" x2="395.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="3720.0" x2="481.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="384" y="3720" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="419" y="3736" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-064</text>
<text x="419" y="3751" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="376" y="3743" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="300" y="3798" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 2</text>
<text x="342" y="3798" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-033 … 064 · 8 leaves</text>
<rect x="574" y="3634" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="580.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="611.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-01</text>
<rect x="648.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="679.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-02</text>
<rect x="734.0" y="3646.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="765.0" y="3660.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-08</text>
<text x="722" y="3661" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="617.0" y1="3720.0" x2="611.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="3720.0" x2="679.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="3720.0" x2="765.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="582" y="3720" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="617" y="3736" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-097</text>
<text x="617" y="3751" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="703.0" y1="3720.0" x2="611.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="3720.0" x2="679.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="3720.0" x2="765.0" y2="3666.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="668" y="3720" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="703" y="3736" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-127</text>
<text x="703" y="3751" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="660" y="3743" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="584" y="3798" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 4</text>
<text x="626" y="3798" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-097 … 127 + C-UFM</text>
<text x="545" y="3726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">⋯</text>
<text x="545" y="3746" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SU 3</text>
<path d="M746 3735 H743 V3666" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="3726.0" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="3739.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 1</text>
<path d="M746 3759 H740 V3666" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="3750.0" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="3763.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 2</text>
<line x1="252.0" y1="3600.0" x2="83.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="151.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="237.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="327.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="395.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="481.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="611.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="679.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3600.0" x2="765.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="83.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="151.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="237.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="327.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="395.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="481.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="611.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="679.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3600.0" x2="765.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="83.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="151.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="237.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="327.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="395.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="481.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="611.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="679.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3600.0" x2="765.0" y2="3646.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="48.0" y1="3832.0" x2="70.0" y2="3832.0" style="stroke:var(--c-danger);stroke-width:2.4"/>
<text x="76" y="3836" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine–leaf: 2 × 400 Gb/s</text>
<line x1="300.0" y1="3832.0" x2="322.0" y2="3832.0" style="stroke:var(--c-warn);stroke-width:2.4"/>
<text x="328" y="3836" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX–leaf: 400 Gb/s</text>
<line x1="552.0" y1="3832.0" x2="574.0" y2="3832.0" style="stroke:var(--c-accent);stroke-width:2.4"/>
<text x="580" y="3836" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">UFM–leaf: 200 Gb/s</text>
<text x="48" y="3860" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Line colours follow NVIDIA&#x27;s diagram and mark the cable type here (not the poster&#x27;s pillar colours). Names follow it too:</text>
<text x="48" y="3875" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Leaf-SU-rail, so Leaf-04-08 is rail 8 of SU 4 and NIC 0 to 7 of every DGX go to Leaf-xx-01 to 08. SU 3 (H100-065 … 096) repeats the</text>
<text x="48" y="3890" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">same layout.</text>
<rect x="44.0" y="3904.0" width="368.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="54" y="3920" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">storage fabric (own InfiniBand)</text>
<text x="54" y="3936" text-anchor="start" style="fill:var(--c-text);font-size:11px">16 switches · ≥ 40 GB/s per node</text>
<rect x="428.0" y="3904.0" width="368.0" height="40" rx="5" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="438.0" y="3920" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">management (Ethernet)</text>
<text x="438.0" y="3936" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">in-band 100 GbE · OOB 1 GbE</text>
<rect x="40.0" y="3968" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="3990" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">127</text>
<text x="50.0" y="4007" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX nodes</text>
<rect x="234.0" y="3968" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="244.0" y="3990" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1,016</text>
<text x="244.0" y="4007" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs</text>
<rect x="428.0" y="3968" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="438.0" y="3990" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">32 + 16</text>
<text x="438.0" y="4007" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">leaf + spine</text>
<rect x="622.0" y="3968" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="632.0" y="3990" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">4</text>
<text x="632.0" y="4007" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">separate networks</text>
<text x="40" y="4038" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">127 DGX (UFM pair in SU 4)</text>
<text x="420" y="4038" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">within a rail: one hop to 31 nodes</text>
<text x="40" y="4054" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">compute: 32 leaf + 16 spine</text>
<text x="420" y="4054" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">every leaf to every spine, 2 × 400 Gb/s</text>
<text x="40" y="4070" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1:1 fat-tree · SHARPv3</text>
<text x="420" y="4070" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 networks: compute, storage, 2 mgmt</text>
<text x="40" y="4098" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Why it matters</text>
<text x="40" y="4116" text-anchor="start" style="fill:var(--c-text);font-size:12px">NIC 0 of every node plugs into Leaf-xx-01 and NIC 7 into Leaf-xx-08; these eight lanes are the rails, and same-numbered GPUs</text>
<text x="40" y="4133" text-anchor="start" style="fill:var(--c-text);font-size:12px">meet in a single hop. Every leaf in turn reaches every spine over two 400 Gb/s links, so cross-rail traffic always turns</text>
<text x="40" y="4150" text-anchor="start" style="fill:var(--c-text);font-size:12px">through a spine. Storage and management traffic run on their own networks so they never crowd the compute fabric. The same</text>
<text x="40" y="4167" text-anchor="start" style="fill:var(--c-text);font-size:12px">pattern grows to 64 SUs: 2,048 DGX, i.e. 16,384 H100s, the scale parts B and C use.</text>
<rect x="36.0" y="4185.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="4200" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">A restaurant chain: InfiniBand is the courier between branches, storage the warehouse.</text>
<line x1="420.0" y1="4220.0" x2="420.0" y2="4242.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="4235" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">a closer look at the SuperPOD&#x27;s network</text>
<rect x="28" y="4245" width="784" height="719" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="4269" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4b · A closer look at the SuperPOD</text>
<text x="40" y="4289" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Let&#x27;s read the same 127 DGX as a network: why is it non-blocking, how does traffic find its way, and how far does it grow?</text>
<text x="40" y="4319" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">1 · Why is it non-blocking (1:1)? Count the ports</text>
<rect x="40.0" y="4345.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="105.0" y="4359" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">32 DGX</text>
<text x="105.0" y="4373" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">NIC i of each</text>
<line x1="173.0" y1="4365.0" x2="212.0" y2="4365.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.8"/>
<text x="192" y="4357" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 ports</text>
<rect x="215.0" y="4345.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4359" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">leaf i · QM9700</text>
<text x="280.0" y="4373" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">64 × 400 Gb/s ports</text>
<line x1="348.0" y1="4365.0" x2="387.0" y2="4365.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.8"/>
<text x="367" y="4357" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 ports</text>
<rect x="390.0" y="4345.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="455.0" y="4359" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">16 spines</text>
<text x="455.0" y="4373" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">64 ports each</text>
<rect x="40.0" y="4401.0" width="480.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4416.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32 leaves × 32 uplinks = 1,024 cables = 16 spines × 64 ports</text>
<text x="40" y="4445" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Cable counts (NVIDIA&#x27;s table): 1,020 from nodes to leaves (127 × 8 + 4 UFM),</text>
<text x="40" y="4462" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">1,024 from leaves to spines.</text>
<text x="552" y="4357" text-anchor="start" style="fill:var(--c-text);font-size:12px">32 of each leaf&#x27;s 64 ports go down to</text>
<text x="552" y="4374" text-anchor="start" style="fill:var(--c-text);font-size:12px">the nodes and 32 go up to the spines.</text>
<text x="552" y="4391" text-anchor="start" style="fill:var(--c-text);font-size:12px">Whatever traffic enters from below can</text>
<text x="552" y="4408" text-anchor="start" style="fill:var(--c-text);font-size:12px">leave upward, so there is no</text>
<text x="552" y="4425" text-anchor="start" style="fill:var(--c-text);font-size:12px">oversubscription (1:1).</text>
<line x1="540.0" y1="4343.0" x2="540.0" y2="4462.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="4498" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">2 · How does traffic find its way?</text>
<text x="40" y="4522" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Same rail: node A&#x27;s GPU 3 → node B&#x27;s GPU 3</text>
<rect x="406.4" y="4510.0" width="113.6" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="463.2" y="4523.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">1 leaf, no spine</text>
<rect x="40.0" y="4536.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="4552.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="4548.0" x2="138.0" y2="4548.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="140.0" y="4536.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="180.0" y="4552.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="222.0" y1="4548.0" x2="238.0" y2="4548.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="4536.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="4552.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · B</text>
<text x="40" y="4588" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Different rail, direct: GPU 3 → GPU 5</text>
<rect x="412.5" y="4576.0" width="107.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="466.2" y="4589.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">through a spine</text>
<rect x="40.0" y="4602.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="4618.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="4614.0" x2="138.0" y2="4614.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="140.0" y="4602.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="180.0" y="4618.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="222.0" y1="4614.0" x2="238.0" y2="4614.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="4602.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4618.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="322.0" y1="4614.0" x2="338.0" y2="4614.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="340.0" y="4602.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="380.0" y="4618.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="422.0" y1="4614.0" x2="438.0" y2="4614.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="440.0" y="4602.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="480.0" y="4618.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<text x="40" y="4654" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Different rail with PXN: NVLink first, then one leaf</text>
<rect x="455.2" y="4642.0" width="64.8" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="487.6" y="4655.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">no spine</text>
<rect x="40.0" y="4668.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="4684.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="4680.0" x2="138.0" y2="4680.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<rect x="140.0" y="4668.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="180.0" y="4684.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · A</text>
<line x1="222.0" y1="4680.0" x2="238.0" y2="4680.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="4668.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4684.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="322.0" y1="4680.0" x2="338.0" y2="4680.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="340.0" y="4668.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="380.0" y="4684.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<text x="552" y="4522" text-anchor="start" style="fill:var(--c-text);font-size:12px">According to NVIDIA, traffic within a</text>
<text x="552" y="4539" text-anchor="start" style="fill:var(--c-text);font-size:12px">rail is always one hop from the other 31</text>
<text x="552" y="4556" text-anchor="start" style="fill:var(--c-text);font-size:12px">nodes of an SU, while traffic between</text>
<text x="552" y="4573" text-anchor="start" style="fill:var(--c-text);font-size:12px">rails crosses the spine layer. PXN in</text>
<text x="552" y="4590" text-anchor="start" style="fill:var(--c-text);font-size:12px">NCCL 2.12 first moves the data over</text>
<text x="552" y="4607" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVLink to the GPU on the right rail and</text>
<text x="552" y="4624" text-anchor="start" style="fill:var(--c-text);font-size:12px">skips the spine; NVIDIA reports this</text>
<text x="552" y="4641" text-anchor="start" style="fill:var(--c-text);font-size:12px">doubles all2all throughput.</text>
<line x1="540.0" y1="4510.0" x2="540.0" y2="4692.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="4728" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">3 · Scale and bandwidth</text>
<rect x="40.0" y="4740" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="4762" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">400 GB/s</text>
<text x="50.0" y="4779" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per node</text>
<rect x="228.0" y="4740" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="238.0" y="4762" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">12.8 TB/s</text>
<text x="238.0" y="4779" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per SU</text>
<rect x="40.0" y="4796" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="4818" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">50.8 TB/s</text>
<text x="50.0" y="4835" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">all GPUs</text>
<rect x="228.0" y="4796" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="238.0" y="4818" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">51.2 TB/s</text>
<text x="238.0" y="4835" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine capacity</text>
<text x="40" y="4874" text-anchor="start" style="fill:var(--c-text);font-size:12px">A node pushes 8 × 50 GB/s = 400 GB/s into the network, and a</text>
<text x="40" y="4891" text-anchor="start" style="fill:var(--c-text);font-size:12px">32-node SU 12.8 TB/s. What all 1,016 GPUs can send</text>
<text x="40" y="4908" text-anchor="start" style="fill:var(--c-text);font-size:12px">(50.8 TB/s) and what the 16 spines can carry</text>
<text x="40" y="4925" text-anchor="start" style="fill:var(--c-text);font-size:12px">(16 × 64 × 50 GB/s = 51.2 TB/s) are almost equal: that is</text>
<text x="40" y="4942" text-anchor="start" style="fill:var(--c-text);font-size:12px">what non-blocking means in numbers.</text>
<text x="432" y="4728" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">4 · Going bigger: a third tier</text>
<text x="432" y="4754" text-anchor="start" style="fill:var(--c-text);font-size:12px">In NVIDIA&#x27;s table, two tiers (leaf + spine) suffice up to</text>
<text x="432" y="4771" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 SUs; from 16 SUs a core tier is added on top. A 2,048-DGX</text>
<text x="432" y="4788" text-anchor="start" style="fill:var(--c-text);font-size:12px">(16,384-H100) build uses 512 leaf, 512 spine and 256 core</text>
<text x="432" y="4805" text-anchor="start" style="fill:var(--c-text);font-size:12px">switches; parts B and C think at this scale.</text>
<text x="432" y="4831" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> NCCL lays out its channels along the rails:</text>
<text x="432" y="4848" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">same-numbered GPUs talk through a single leaf, and</text>
<text x="432" y="4865" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">cross-rail traffic is first moved onto NVLink with PXN.</text>
<line x1="420.0" y1="4740.0" x2="420.0" y2="4942.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="4967.0" x2="420.0" y2="4989.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="4982" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">the building blocks that keep it running</text>
<rect x="28" y="4992" width="784" height="782" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5016" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">SuperPOD&#x27;s building blocks: why is each one there?</text>
<text x="40" y="5035" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVIDIA&#x27;s 4-SU reference design (127 DGX); counts and roles come from NVIDIA&#x27;s documentation. The stripe colour shows which pillar a</text>
<text x="40" y="5050" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">part serves; grey is management and infrastructure.</text>
<text x="40" y="5074" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Hardware</text>
<rect x="36" y="5084" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5084" width="4" height="114" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="48" y="5103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">DGX H100 system</text>
<rect x="237.8" y="5090.0" width="38.2" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="256.9" y="5103.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×127</text>
<text x="48" y="5120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 H100 · 640 GB HBM3 · 900 GB/s NVLink</text>
<text x="48" y="5138" text-anchor="start" style="fill:var(--c-text);font-size:11px">All of the model&#x27;s maths happens here;</text>
<text x="48" y="5153" text-anchor="start" style="fill:var(--c-text);font-size:11px">every other part of the cluster exists</text>
<text x="48" y="5168" text-anchor="start" style="fill:var(--c-text);font-size:11px">to feed these boxes.</text>
<rect x="296" y="5084" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5084" width="4" height="114" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="308" y="5103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Compute fabric · QM9700</text>
<rect x="503.9" y="5090.0" width="32.2" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="519.9" y="5103.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×48</text>
<text x="308" y="5120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">32 leaf + 16 spine · 8 × 400 Gb/s per</text>
<text x="308" y="5134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">node</text>
<text x="308" y="5152" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gradient and activation traffic must</text>
<text x="308" y="5167" text-anchor="start" style="fill:var(--c-text);font-size:11px">cross nodes without jamming, so every</text>
<text x="308" y="5182" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU gets its own 400 Gb/s rail.</text>
<rect x="556" y="5084" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5084" width="4" height="114" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="568" y="5103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Storage fabric · QM9700</text>
<rect x="763.9" y="5090.0" width="32.2" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="779.9" y="5103.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×16</text>
<text x="568" y="5120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">≥ 40 GB/s per node · ~4:3</text>
<text x="568" y="5138" text-anchor="start" style="fill:var(--c-text);font-size:11px">Data reads and checkpoint writes stay</text>
<text x="568" y="5153" text-anchor="start" style="fill:var(--c-text);font-size:11px">off the compute fabric&#x27;s AllReduce</text>
<text x="568" y="5168" text-anchor="start" style="fill:var(--c-text);font-size:11px">traffic; its speed is sized to the</text>
<text x="568" y="5183" text-anchor="start" style="fill:var(--c-text);font-size:11px">storage array&#x27;s peak.</text>
<rect x="36" y="5206" width="248" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5206" width="4" height="129" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="48" y="5225" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">UFM appliance</text>
<rect x="249.9" y="5212.0" width="26.1" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="262.9" y="5225.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×4</text>
<text x="48" y="5242" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">InfiniBand telemetry and analytics</text>
<text x="48" y="5260" text-anchor="start" style="fill:var(--c-text);font-size:11px">Watches thousands of ports and cables</text>
<text x="48" y="5275" text-anchor="start" style="fill:var(--c-text);font-size:11px">and finds the bad link (in Llama 3&#x27;s</text>
<text x="48" y="5290" text-anchor="start" style="fill:var(--c-text);font-size:11px">run, 8.4% of interruptions came from</text>
<text x="48" y="5305" text-anchor="start" style="fill:var(--c-text);font-size:11px">switches and cables). A pair of them</text>
<text x="48" y="5320" text-anchor="start" style="fill:var(--c-text);font-size:11px">takes one DGX&#x27;s place, hence 127 nodes.</text>
<rect x="296" y="5206" width="248" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5206" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="5225" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">In-band mgmt · SN4600C</text>
<rect x="509.9" y="5212.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="522.9" y="5225.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="308" y="5242" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 ports × 100 GbE (Ethernet)</text>
<text x="308" y="5260" text-anchor="start" style="fill:var(--c-text);font-size:11px">Provisioning, user logins, the file</text>
<text x="308" y="5275" text-anchor="start" style="fill:var(--c-text);font-size:11px">system and outside access (NGC, package</text>
<text x="308" y="5290" text-anchor="start" style="fill:var(--c-text);font-size:11px">repos) run here, leaving the compute</text>
<text x="308" y="5305" text-anchor="start" style="fill:var(--c-text);font-size:11px">fabric to GPU traffic.</text>
<rect x="556" y="5206" width="248" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5206" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="568" y="5225" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Out-of-band mgmt · SN2201</text>
<rect x="769.9" y="5212.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="782.9" y="5225.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="568" y="5242" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">48 ports × 1 GbE copper</text>
<text x="568" y="5260" text-anchor="start" style="fill:var(--c-text);font-size:11px">Reaches each server&#x27;s BMC chip and the</text>
<text x="568" y="5275" text-anchor="start" style="fill:var(--c-text);font-size:11px">PDUs, so a node can be powered and</text>
<text x="568" y="5290" text-anchor="start" style="fill:var(--c-text);font-size:11px">rebooted even when its OS hangs; kept</text>
<text x="568" y="5305" text-anchor="start" style="fill:var(--c-text);font-size:11px">apart from user networks.</text>
<rect x="36" y="5343" width="248" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5343" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="5362" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Management servers</text>
<rect x="249.9" y="5349.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="262.9" y="5362.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×5</text>
<text x="48" y="5379" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 sockets · 24+ cores · 384 GB RAM</text>
<text x="48" y="5397" text-anchor="start" style="fill:var(--c-text);font-size:11px">Cluster management and scheduling</text>
<text x="48" y="5412" text-anchor="start" style="fill:var(--c-text);font-size:11px">software runs on separate machines, so</text>
<text x="48" y="5427" text-anchor="start" style="fill:var(--c-text);font-size:11px">the GPU nodes are left for training</text>
<text x="48" y="5442" text-anchor="start" style="fill:var(--c-text);font-size:11px">alone.</text>
<rect x="296" y="5343" width="248" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5343" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="5362" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Racks and power</text>
<rect x="473.6" y="5349.0" width="62.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="504.8" y="5362.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">38 racks</text>
<text x="308" y="5379" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">38 racks · 108 PDUs</text>
<text x="308" y="5397" text-anchor="start" style="fill:var(--c-text);font-size:11px">4 DGX per rack (40.8 kW) and 8 DGX</text>
<text x="308" y="5412" text-anchor="start" style="fill:var(--c-text);font-size:11px">racks per SU; the 127 DGX alone draw</text>
<text x="308" y="5427" text-anchor="start" style="fill:var(--c-text);font-size:11px">127 × 10.2 kW ≈ 1.3 MW. Power and</text>
<text x="308" y="5442" text-anchor="start" style="fill:var(--c-text);font-size:11px">cooling are the cluster&#x27;s hidden fourth</text>
<text x="308" y="5457" text-anchor="start" style="fill:var(--c-text);font-size:11px">pillar.</text>
<text x="40" y="5498" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Software</text>
<rect x="36" y="5508" width="248" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5508" width="4" height="128" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="5527" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Base Command Manager</text>
<rect x="237.8" y="5514.0" width="38.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="256.9" y="5527.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">mgmt</text>
<text x="48" y="5544" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">cluster management · automated</text>
<text x="48" y="5558" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">provisioning</text>
<text x="48" y="5576" text-anchor="start" style="fill:var(--c-text);font-size:11px">Installs, updates and monitors one OS</text>
<text x="48" y="5591" text-anchor="start" style="fill:var(--c-text);font-size:11px">image across thousands of nodes;</text>
<text x="48" y="5606" text-anchor="start" style="fill:var(--c-text);font-size:11px">managing 127 machines by hand is not</text>
<text x="48" y="5621" text-anchor="start" style="fill:var(--c-text);font-size:11px">realistic.</text>
<rect x="296" y="5508" width="248" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5508" width="4" height="128" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="308" y="5527" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Slurm</text>
<rect x="479.6" y="5514.0" width="56.4" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="507.8" y="5527.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<text x="308" y="5544" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">batch workload manager</text>
<text x="308" y="5562" text-anchor="start" style="fill:var(--c-text);font-size:11px">Decides which GPUs a multi-node</text>
<text x="308" y="5577" text-anchor="start" style="fill:var(--c-text);font-size:11px">training job runs on and in what order;</text>
<text x="308" y="5592" text-anchor="start" style="fill:var(--c-text);font-size:11px">it can drain a failed node and requeue</text>
<text x="308" y="5607" text-anchor="start" style="fill:var(--c-text);font-size:11px">the job.</text>
<rect x="556" y="5508" width="248" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5508" width="4" height="128" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="568" y="5527" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Magnum IO</text>
<rect x="679.1" y="5514.0" width="116.9" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="737.6" y="5527.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">network + storage</text>
<text x="568" y="5544" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NCCL · GPUDirect RDMA / Storage · SHARP</text>
<text x="568" y="5562" text-anchor="start" style="fill:var(--c-text);font-size:11px">The library stack that speeds up data</text>
<text x="568" y="5577" text-anchor="start" style="fill:var(--c-text);font-size:11px">movement between GPUs and disks; steps</text>
<text x="568" y="5592" text-anchor="start" style="fill:var(--c-text);font-size:11px">1, 3 and 6 in part B run on it.</text>
<rect x="36" y="5644" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5644" width="4" height="114" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="48" y="5663" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">NGC catalog</text>
<rect x="219.7" y="5650.0" width="56.4" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="247.8" y="5663.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<text x="48" y="5680" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU-optimised containers</text>
<text x="48" y="5698" text-anchor="start" style="fill:var(--c-text);font-size:11px">CUDA, NCCL and frameworks such as</text>
<text x="48" y="5713" text-anchor="start" style="fill:var(--c-text);font-size:11px">PyTorch arrive as one matched set, so</text>
<text x="48" y="5728" text-anchor="start" style="fill:var(--c-text);font-size:11px">everyone starts from the same software</text>
<text x="48" y="5743" text-anchor="start" style="fill:var(--c-text);font-size:11px">stack.</text>
<rect x="296" y="5644" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5644" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="5663" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Base Command Platform</text>
<rect x="497.8" y="5650.0" width="38.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="516.9" y="5663.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">mgmt</text>
<text x="308" y="5680" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">software service for AI training</text>
<text x="308" y="5698" text-anchor="start" style="fill:var(--c-text);font-size:11px">Where data scientists submit jobs and</text>
<text x="308" y="5713" text-anchor="start" style="fill:var(--c-text);font-size:11px">manage datasets and results, without</text>
<text x="308" y="5728" text-anchor="start" style="fill:var(--c-text);font-size:11px">needing to know the hardware</text>
<text x="308" y="5743" text-anchor="start" style="fill:var(--c-text);font-size:11px">underneath.</text>
<rect x="556" y="5644" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5644" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="568" y="5663" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">NVIDIA AI Enterprise</text>
<rect x="757.8" y="5650.0" width="38.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="776.9" y="5663.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">mgmt</text>
<text x="568" y="5680" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tools, frameworks, orchestration</text>
<text x="568" y="5698" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gives developers frameworks and tools,</text>
<text x="568" y="5713" text-anchor="start" style="fill:var(--c-text);font-size:11px">and gives IT teams supported management</text>
<text x="568" y="5728" text-anchor="start" style="fill:var(--c-text);font-size:11px">and orchestration.</text>
<rect x="28.0" y="5786.0" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="5806" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: the further from the GPU, the narrower the road: HBM 3.35 TB/s → NVLink 450 GB/s → network 50 GB/s.</text>
<text x="420.0" y="5851" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">We know the parts. Now let&#x27;s watch them work together in one training step.</text>
<line x1="420.0" y1="5859.0" x2="420.0" y2="5879.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="5883" width="808" height="880" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="5907" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="5911.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">B</text>
<text x="56" y="5912" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Follow one training step: where does data go?</text>
<text x="28" y="5933" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Llama 3 405B on a 64-SU H100 SuperPOD: TP 8 × PP 16 × DP 128 = 16,384 GPUs</text>
<rect x="28" y="5945" width="784" height="264" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="290.0" y="5957.0" width="260.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="5972.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine → the other 127 DP replicas</text>
<rect x="48" y="6005" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="58" y="6021" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">node A · PP stage s</text>
<rect x="88.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="150.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="212.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="274.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="88.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="150.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="212.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="274.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="112.0" y1="6049.0" x2="112.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="112.0" y1="6075.0" x2="112.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="6049.0" x2="174.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="6075.0" x2="174.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="6049.0" x2="236.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="6075.0" x2="236.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="6049.0" x2="298.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="6075.0" x2="298.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="88.0" y="6057.0" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="205.0" y="6070.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<circle cx="64" cy="6039" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="64" cy="6066" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="64" y="6070.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="64" cy="6093" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<rect x="502" y="6005" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="512" y="6021" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">node B · stage s+1</text>
<rect x="530.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="592.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="654.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="716.0" y="6029.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="6043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="530.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="592.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="654.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="716.0" y="6083.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="6097.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="554.0" y1="6049.0" x2="554.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="554.0" y1="6075.0" x2="554.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="6049.0" x2="616.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="6075.0" x2="616.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="6049.0" x2="678.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="6075.0" x2="678.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="6049.0" x2="740.0" y2="6057.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="6075.0" x2="740.0" y2="6083.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="530.0" y="6057.0" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="6070.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<rect x="396.0" y="6015.0" width="48.0" height="96" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="6073" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">leaf</text>
<circle cx="420.0" cy="6041" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="420.0" y="6045.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="420.0" cy="6095" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="420.0" y="6099.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<line x1="338.0" y1="6041.0" x2="396.0" y2="6041.0" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<line x1="444.0" y1="6041.0" x2="500.0" y2="6041.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="502.0" y1="6095.0" x2="444.0" y2="6095.0" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:4 3"/>
<line x1="396.0" y1="6095.0" x2="340.0" y2="6095.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<line x1="420.0" y1="6013.0" x2="420.0" y2="5981.0" marker-start="url(#bp-gen)" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<circle cx="438.0" cy="5997" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438.0" y="6001.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="48.0" y="6157.0" width="744.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="420.0" y="6173" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">storage system (e.g. Lustre · Weka)</text>
<text x="420.0" y="6189" text-anchor="middle" style="fill:var(--c-text);font-size:11px">separate storage fabric · ≥ 40 GB/s per node</text>
<line x1="108.0" y1="6155.0" x2="108.0" y2="6124.0" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<circle cx="126" cy="6139" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="126" y="6143.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<line x1="208.0" y1="6124.0" x2="208.0" y2="6155.0" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5;stroke-dasharray:5 4"/>
<circle cx="226" cy="6139" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="226" y="6143.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="647.0" y="6141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(node B uses the same path)</text>
<rect x="28" y="6219" width="784" height="470" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="46.0" y1="6245.0" x2="46.0" y2="6571.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="46" cy="6245" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="46" y="6249.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="6250" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Read data</text>
<rect x="66.0" y="6259.0" width="171.3" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="151.7" y="6272.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">storage → HBM · every step</text>
<text x="66" y="6295" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">This step&#x27;s 16 million tokens reach HBM from storage;</text>
<text x="66" y="6311" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">thanks to GDS or mmap, no CPU copy gets in the way.</text>
<circle cx="46" cy="6343" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="46" y="6347.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="6348" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Forward pass</text>
<rect x="66.0" y="6357.0" width="177.4" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.7" y="6370.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM 3.35 TB/s · every layer</text>
<text x="66" y="6393" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">In every layer the SMs pull weights from HBM and</text>
<text x="66" y="6409" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">multiply them in the Tensor Cores; at ~295 operations</text>
<text x="66" y="6425" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">per byte read, the cores are fed.</text>
<circle cx="46" cy="6457" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="6461.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="6462" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">TP AllReduce</text>
<rect x="66.0" y="6471.0" width="195.5" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="163.8" y="6484.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink 450 GB/s · 2× per layer</text>
<text x="66" y="6507" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Each matrix is split over 8 GPUs, so the partial</text>
<text x="66" y="6523" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">results are summed over NVLink; that is why TP never</text>
<text x="66" y="6539" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">leaves the chassis.</text>
<circle cx="46" cy="6571" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="6575.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="66" y="6576" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Send activations (PP)</text>
<rect x="66.0" y="6585.0" width="183.4" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="157.7" y="6598.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">50 GB/s/GPU · per microbatch</text>
<text x="66" y="6621" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">When this node&#x27;s layers finish, the output (~16 MB)</text>
<text x="66" y="6637" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">crosses the network to the next node; the road is 9×</text>
<text x="66" y="6653" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">narrower, but the load is small.</text>
<line x1="438.0" y1="6245.0" x2="438.0" y2="6587.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="438" cy="6245" r="10" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="438" y="6249.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="6250" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Backward pass</text>
<rect x="458.0" y="6259.0" width="135.0" height="19" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="525.5" y="6272.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">same links, reversed</text>
<text x="458" y="6295" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The error signal (gradients) travels back the same way;</text>
<text x="458" y="6311" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">steps 3 and 4 repeat in reverse, at ~2× the forward</text>
<text x="458" y="6327" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">work.</text>
<circle cx="438" cy="6359" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438" y="6363.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="6364" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Gradient AllReduce (DP)</text>
<rect x="458.0" y="6373.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="543.6" y="6386.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">rail switches · every step</text>
<text x="458" y="6409" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The gradients of 128 replicas are averaged so all of</text>
<text x="458" y="6425" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">them make the same update; InfiniBand SHARP sums inside</text>
<text x="458" y="6441" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">the switch, cutting 2·B/W to B/W.</text>
<circle cx="438" cy="6473" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="438" y="6477.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<text x="458" y="6478" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Optimizer step</text>
<rect x="458.0" y="6487.0" width="153.2" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="534.6" y="6500.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">inside HBM · every step</text>
<text x="458" y="6523" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">AdamW updates the weights and keeps 12 bytes of FP32</text>
<text x="458" y="6539" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">state per parameter: master weight, momentum and</text>
<text x="458" y="6555" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">variance.</text>
<circle cx="438" cy="6587" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="438" y="6591.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="458" y="6592" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Checkpoint</text>
<rect x="458.0" y="6601.0" width="189.5" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="552.7" y="6614.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM → storage · every ~14 min</text>
<text x="458" y="6637" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Every ~14 minutes 7.5 TB of state goes to disk: ~31 s</text>
<text x="458" y="6653" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">at 250 GB/s, ~5 min on a legacy 25 GB/s path; after a</text>
<text x="458" y="6669" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">failure training resumes here.</text>
<line x1="420.0" y1="6233.0" x2="420.0" y2="6675.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="6701.0" width="784.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="6721" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: in one step data crosses four different roads (HBM, NVLink, network, storage); the step runs only as</text>
<text x="40" y="6738" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">fast as the slowest one.</text>
<text x="420.0" y="6783" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">So where do these roads jam? Each pillar has a wall.</text>
<line x1="420.0" y1="6791.0" x2="420.0" y2="6811.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="6815" width="808" height="1041" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="6839" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="6843.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">C</text>
<text x="56" y="6844" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Three walls: symptom → cause → fix</text>
<text x="28" y="6865" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Curves are computed from the article&#x27;s formulas; red is the wall, green the fix.</text>
<rect x="28" y="6877" width="784" height="312" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="6901" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Memory wall</text>
<text x="800" y="6901" text-anchor="end" style="fill:var(--c-accent);font-size:11px;font-weight:700">pillar 1 · compute</text>
<line x1="80.0" y1="6921.0" x2="80.0" y2="7047.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="7047.0" x2="512.0" y2="7047.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="6928.2" x2="512.0" y2="6928.2" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="6932.2" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">990</text>
<text x="76" y="7051" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="40" y="6920.2" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s</text>
<line x1="377.9" y1="6928.2" x2="377.9" y2="7047.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<polyline points="80.0,7046.6 84.5,7046.6 89.0,7046.5 93.5,7046.5 98.0,7046.4 102.5,7046.4 107.0,7046.3 111.5,7046.3 116.0,7046.2 120.5,7046.1 125.0,7046.0 129.5,7046.0 134.0,7045.9 138.5,7045.8 143.0,7045.6 147.5,7045.5 152.0,7045.4 156.5,7045.2 161.0,7045.1 165.5,7044.9 170.0,7044.7 174.5,7044.5 179.0,7044.3 183.5,7044.1 188.0,7043.8 192.5,7043.5 197.0,7043.2 201.5,7042.8 206.0,7042.5 210.5,7042.0 215.0,7041.6 219.5,7041.1 224.0,7040.6 228.5,7040.0 233.0,7039.4 237.5,7038.7 242.0,7037.9 246.5,7037.1 251.0,7036.2 255.5,7035.3 260.0,7034.2 264.5,7033.0 269.0,7031.8 273.5,7030.4 278.0,7028.9 282.5,7027.3 287.0,7025.5 291.5,7023.6 296.0,7021.5 300.5,7019.2 305.0,7016.8 309.5,7014.1 314.0,7011.1 318.5,7007.9 323.0,7004.4 327.5,7000.7 332.0,6996.6 336.5,6992.1 341.0,6987.2 345.5,6982.0 350.0,6976.2 354.5,6970.0 359.0,6963.3 363.5,6956.0 368.0,6948.1 372.5,6939.5 377.0,6930.2 381.5,6928.2 386.0,6928.2 390.5,6928.2 395.0,6928.2 399.5,6928.2 404.0,6928.2 408.5,6928.2 413.0,6928.2 417.5,6928.2 422.0,6928.2 426.5,6928.2 431.0,6928.2 435.5,6928.2 440.0,6928.2 444.5,6928.2 449.0,6928.2 453.5,6928.2 458.0,6928.2 462.5,6928.2 467.0,6928.2 471.5,6928.2 476.0,6928.2 480.5,6928.2 485.0,6928.2 489.5,6928.2 494.0,6928.2 498.5,6928.2 503.0,6928.2 507.5,6928.2 512.0,6928.2" style="fill:none;stroke:var(--c-accent);stroke-width:2.2"/>
<text x="80.0" y="7061" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="152.0" y="7061" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="224.0" y="7061" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="296.0" y="7061" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="368.0" y="7061" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="440.0" y="7061" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1,024</text>
<text x="512.0" y="7061" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4,096</text>
<text x="512" y="7077" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">batch B (tokens, log scale)</text>
<text x="150" y="6944.2" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">memory-bound</text>
<text x="470" y="7029" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">compute-bound</text>
<circle cx="80.0" cy="7046.6" r="4" style="fill:var(--c-danger)"/>
<text x="88.0" y="7037" text-anchor="start" style="fill:var(--c-text);font-size:11px">B = 1: 3.3 TFLOP/s (0.3%)</text>
<circle cx="332.0" cy="6996.6" r="4" style="fill:var(--c-accent)"/>
<text x="324.0" y="6990.6" text-anchor="end" style="fill:var(--c-text);font-size:11px">B = 128: 420 TFLOP/s (42%)</text>
<circle cx="377.9" cy="6928.2" r="4" style="fill:var(--c-success)"/>
<text x="385.9" y="6944.2" text-anchor="start" style="fill:var(--c-text);font-size:11px">B ≈ 310: ceiling 990</text>
<text x="40" y="7095" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP (D = 8,192, F = 28,672), H100 BF16: speed = min(990, 3.35·α)</text>
<text x="40" y="7111" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">How to read it: as the batch grows, every byte read does more work; the curve hits</text>
<text x="40" y="7126" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">the ceiling near B ≈ 310.</text>
<text x="552" y="6931" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Symptom</text>
<text x="552" y="6949" text-anchor="start" style="fill:var(--c-text);font-size:12px">The GPU looks busy, yet only 1–2% of its</text>
<text x="552" y="6966" text-anchor="start" style="fill:var(--c-text);font-size:12px">power is used.</text>
<text x="552" y="6994" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Cause</text>
<text x="552" y="7012" text-anchor="start" style="fill:var(--c-text);font-size:12px">Too little work per byte: while α = BDF</text>
<text x="552" y="7029" text-anchor="start" style="fill:var(--c-text);font-size:12px">/ (B(D+F) + DF) stays below the H100&#x27;s</text>
<text x="552" y="7046" text-anchor="start" style="fill:var(--c-text);font-size:12px">knee of 295, the Tensor Cores wait for</text>
<text x="552" y="7063" text-anchor="start" style="fill:var(--c-text);font-size:12px">data.</text>
<text x="552" y="7091" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Fix</text>
<rect x="552.0" y="7101.0" width="128.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="616.5" y="7115.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">batch ≥ ~310 tokens</text>
<rect x="552.0" y="7127.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="7141.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">fusion / FlashAttention</text>
<rect x="552.0" y="7153.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="7167.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">decode: 4–8 bit weights</text>
<line x1="540.0" y1="6917.0" x2="540.0" y2="7173.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="7199" width="784" height="321" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="7223" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Network wall</text>
<text x="800" y="7223" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">pillar 2 · network</text>
<line x1="184.0" y1="7241.0" x2="184.0" y2="7325.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="184" y="7339" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 GB/s</text>
<line x1="288.0" y1="7241.0" x2="288.0" y2="7325.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="288" y="7339" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 GB/s</text>
<line x1="392.0" y1="7241.0" x2="392.0" y2="7325.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="392" y="7339" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 TB/s</text>
<line x1="496.0" y1="7241.0" x2="496.0" y2="7325.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="496" y="7339" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 TB/s</text>
<text x="40" y="7259" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM (on chip)</text>
<rect x="184" y="7247" width="262.6" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="452.6" y="7259" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3.35 TB/s</text>
<text x="40" y="7285" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (in node)</text>
<rect x="184" y="7273" width="171.9" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="361.9" y="7285" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">450 GB/s</text>
<text x="420.7" y="7285" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷7.4</text>
<text x="40" y="7311" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (off node)</text>
<rect x="184" y="7299" width="72.7" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="262.7" y="7311" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">50 GB/s</text>
<text x="314.9" y="7311" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷9</text>
<rect x="40.0" y="7351.0" width="480.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="280.0" y="7365.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink: Y ≤ F / (C/W) = 28,672 / 2,200 ≈ 13 → TP = 8 fits ✓ (F: Llama 3 70B)</text>
<rect x="40.0" y="7377.0" width="480.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="7391.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">InfiniBand: Y ≤ 28,672 / 19,800 ≈ 1.45 → TP across nodes starves compute ✗</text>
<text x="40" y="7415" text-anchor="start" style="fill:var(--c-text);font-size:11px">PP = 16 stages, bubble = 15 / (15 + M): M = 16 → 48% idle · M = 128 → 10.5%</text>
<text x="40" y="7433" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">How to read it: each bar is one H100&#x27;s bandwidth on that link; leaving the node</text>
<text x="40" y="7448" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">shrinks it 9×.</text>
<text x="552" y="7253" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Symptom</text>
<text x="552" y="7271" text-anchor="start" style="fill:var(--c-text);font-size:12px">Adding nodes stretches the step; GPUs</text>
<text x="552" y="7288" text-anchor="start" style="fill:var(--c-text);font-size:12px">wait to talk.</text>
<text x="552" y="7316" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Cause</text>
<text x="552" y="7334" text-anchor="start" style="fill:var(--c-text);font-size:12px">Data leaving the node squeezes through a</text>
<text x="552" y="7351" text-anchor="start" style="fill:var(--c-text);font-size:12px">road 9× narrower; jobs that talk every</text>
<text x="552" y="7368" text-anchor="start" style="fill:var(--c-text);font-size:12px">layer, like TP, choke on it.</text>
<text x="552" y="7396" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Fix</text>
<rect x="552.0" y="7406.0" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="7420.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TP inside the node</text>
<rect x="552.0" y="7432.0" width="147.1" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="625.5" y="7446.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PP and DP across nodes</text>
<rect x="552.0" y="7458.0" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="7472.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SHARP: 2·B/W → B/W</text>
<rect x="552.0" y="7484.0" width="135.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="619.5" y="7498.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">DualPipe: hide comms</text>
<line x1="540.0" y1="7239.0" x2="540.0" y2="7504.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="7530" width="784" height="269" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="7554" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">I/O wall</text>
<text x="800" y="7554" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:700">pillar 3 · storage</text>
<line x1="80.0" y1="7578.0" x2="80.0" y2="7700.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="7700.0" x2="512.0" y2="7700.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="7670.0" x2="512.0" y2="7670.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7674.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10%</text>
<line x1="80.0" y1="7640.0" x2="512.0" y2="7640.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7644.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">20%</text>
<line x1="80.0" y1="7610.0" x2="512.0" y2="7610.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7614.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">30%</text>
<line x1="80.0" y1="7580.0" x2="512.0" y2="7580.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7584.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">40%</text>
<polyline points="80.0,7700.0 512.0,7625.0" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/>
<polyline points="148.4,7580.3 149.6,7582.0 150.8,7583.6 152.0,7585.1 153.2,7586.6 154.4,7588.0 155.6,7589.4 156.8,7590.7 158.0,7591.9 159.2,7593.2 160.4,7594.3 161.6,7595.5 162.8,7596.6 164.0,7597.6 165.2,7598.7 166.4,7599.7 167.6,7600.6 168.8,7601.6 170.0,7602.5 171.2,7603.3 172.4,7604.2 173.6,7605.0 174.8,7605.8 176.0,7606.5 177.2,7607.3 178.4,7608.0 179.6,7608.7 180.8,7609.4 182.0,7610.0 183.2,7610.6 184.4,7611.3 185.6,7611.8 186.8,7612.4 188.0,7613.0 189.2,7613.5 190.4,7614.1 191.6,7614.6 192.8,7615.1 194.0,7615.5 195.2,7616.0 196.4,7616.5 197.6,7616.9 198.8,7617.3 200.0,7617.7 201.2,7618.1 202.4,7618.5 203.6,7618.9 204.8,7619.3 206.0,7619.6 207.2,7620.0 208.4,7620.3 209.6,7620.6 210.8,7620.9 212.0,7621.2 213.2,7621.5 214.4,7621.8 215.6,7622.1 216.8,7622.4 218.0,7622.6 219.2,7622.9 220.4,7623.1 221.6,7623.3 222.8,7623.6 224.0,7623.8 225.2,7624.0 226.4,7624.2 227.6,7624.4 228.8,7624.6 230.0,7624.8 231.2,7625.0 232.4,7625.2 233.6,7625.3 234.8,7625.5 236.0,7625.7 237.2,7625.8 238.4,7626.0 239.6,7626.1 240.8,7626.2 242.0,7626.4 243.2,7626.5 244.4,7626.6 245.6,7626.7 246.8,7626.8 248.0,7626.9 249.2,7627.1 250.4,7627.1 251.6,7627.2 252.8,7627.3 254.0,7627.4 255.2,7627.5 256.4,7627.6 257.6,7627.7 258.8,7627.7 260.0,7627.8 261.2,7627.9 262.4,7627.9 263.6,7628.0 264.8,7628.0 266.0,7628.1 267.2,7628.1 268.4,7628.2 269.6,7628.2 270.8,7628.2 272.0,7628.3 273.2,7628.3 274.4,7628.3 275.6,7628.3 276.8,7628.4 278.0,7628.4 279.2,7628.4 280.4,7628.4 281.6,7628.4 282.8,7628.4 284.0,7628.4 285.2,7628.4 286.4,7628.4 287.6,7628.4 288.8,7628.4 290.0,7628.4 291.2,7628.4 292.4,7628.4 293.6,7628.4 294.8,7628.4 296.0,7628.4 297.2,7628.3 298.4,7628.3 299.6,7628.3 300.8,7628.3 302.0,7628.2 303.2,7628.2 304.4,7628.2 305.6,7628.2 306.8,7628.1 308.0,7628.1 309.2,7628.0 310.4,7628.0 311.6,7628.0 312.8,7627.9 314.0,7627.9 315.2,7627.8 316.4,7627.8 317.6,7627.7 318.8,7627.7 320.0,7627.6 321.2,7627.6 322.4,7627.5 323.6,7627.4 324.8,7627.4 326.0,7627.3 327.2,7627.3 328.4,7627.2 329.6,7627.1 330.8,7627.1 332.0,7627.0 333.2,7626.9 334.4,7626.9 335.6,7626.8 336.8,7626.7 338.0,7626.6 339.2,7626.6 340.4,7626.5 341.6,7626.4 342.8,7626.3 344.0,7626.2 345.2,7626.2 346.4,7626.1 347.6,7626.0 348.8,7625.9 350.0,7625.8 351.2,7625.7 352.4,7625.6 353.6,7625.6 354.8,7625.5 356.0,7625.4 357.2,7625.3 358.4,7625.2 359.6,7625.1 360.8,7625.0 362.0,7624.9 363.2,7624.8 364.4,7624.7 365.6,7624.6 366.8,7624.5 368.0,7624.4 369.2,7624.3 370.4,7624.2 371.6,7624.1 372.8,7624.0 374.0,7623.9 375.2,7623.8 376.4,7623.7 377.6,7623.6 378.8,7623.5 380.0,7623.3 381.2,7623.2 382.4,7623.1 383.6,7623.0 384.8,7622.9 386.0,7622.8 387.2,7622.7 388.4,7622.6 389.6,7622.4 390.8,7622.3 392.0,7622.2 393.2,7622.1 394.4,7622.0 395.6,7621.8 396.8,7621.7 398.0,7621.6 399.2,7621.5 400.4,7621.4 401.6,7621.2 402.8,7621.1 404.0,7621.0 405.2,7620.9 406.4,7620.7 407.6,7620.6 408.8,7620.5 410.0,7620.4 411.2,7620.2 412.4,7620.1 413.6,7620.0 414.8,7619.9 416.0,7619.7 417.2,7619.6 418.4,7619.5 419.6,7619.3 420.8,7619.2 422.0,7619.1 423.2,7618.9 424.4,7618.8 425.6,7618.7 426.8,7618.5 428.0,7618.4 429.2,7618.3 430.4,7618.1 431.6,7618.0 432.8,7617.9 434.0,7617.7 435.2,7617.6 436.4,7617.4 437.6,7617.3 438.8,7617.2 440.0,7617.0 441.2,7616.9 442.4,7616.7 443.6,7616.6 444.8,7616.5 446.0,7616.3 447.2,7616.2 448.4,7616.0 449.6,7615.9 450.8,7615.7 452.0,7615.6 453.2,7615.5 454.4,7615.3 455.6,7615.2 456.8,7615.0 458.0,7614.9 459.2,7614.7 460.4,7614.6 461.6,7614.4 462.8,7614.3 464.0,7614.1 465.2,7614.0 466.4,7613.8 467.6,7613.7 468.8,7613.5 470.0,7613.4 471.2,7613.2 472.4,7613.1 473.6,7612.9 474.8,7612.8 476.0,7612.6 477.2,7612.5 478.4,7612.3 479.6,7612.2 480.8,7612.0 482.0,7611.9 483.2,7611.7 484.4,7611.6 485.6,7611.4 486.8,7611.3 488.0,7611.1 489.2,7610.9 490.4,7610.8 491.6,7610.6 492.8,7610.5 494.0,7610.3 495.2,7610.2 496.4,7610.0 497.6,7609.8 498.8,7609.7 500.0,7609.5 501.2,7609.4 502.4,7609.2 503.6,7609.1 504.8,7608.9 506.0,7608.7 507.2,7608.6 508.4,7608.4 509.6,7608.3 510.8,7608.1 512.0,7607.9" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<line x1="286.1" y1="7628.4" x2="286.1" y2="7700.0" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="286.1" cy="7628.4" r="4" style="fill:var(--c-danger)"/>
<text x="286.1" y="7714" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700;font-family:var(--font-mono)">42.9</text>
<polyline points="87.2,7596.4 88.4,7610.8 89.6,7621.5 90.8,7629.9 92.0,7636.5 93.2,7641.9 94.4,7646.3 95.6,7650.0 96.8,7653.2 98.0,7655.9 99.2,7658.3 100.4,7660.3 101.6,7662.1 102.8,7663.7 104.0,7665.1 105.2,7666.4 106.4,7667.5 107.6,7668.5 108.8,7669.4 110.0,7670.2 111.2,7671.0 112.4,7671.6 113.6,7672.2 114.8,7672.8 116.0,7673.3 117.2,7673.7 118.4,7674.1 119.6,7674.5 120.8,7674.8 122.0,7675.2 123.2,7675.4 124.4,7675.7 125.6,7675.9 126.8,7676.1 128.0,7676.3 129.2,7676.5 130.4,7676.6 131.6,7676.8 132.8,7676.9 134.0,7677.0 135.2,7677.1 136.4,7677.1 137.6,7677.2 138.8,7677.3 140.0,7677.3 141.2,7677.3 142.4,7677.4 143.6,7677.4 144.8,7677.4 146.0,7677.4 147.2,7677.4 148.4,7677.3 149.6,7677.3 150.8,7677.3 152.0,7677.3 153.2,7677.2 154.4,7677.2 155.6,7677.1 156.8,7677.1 158.0,7677.0 159.2,7676.9 160.4,7676.9 161.6,7676.8 162.8,7676.7 164.0,7676.6 165.2,7676.6 166.4,7676.5 167.6,7676.4 168.8,7676.3 170.0,7676.2 171.2,7676.1 172.4,7676.0 173.6,7675.9 174.8,7675.8 176.0,7675.7 177.2,7675.5 178.4,7675.4 179.6,7675.3 180.8,7675.2 182.0,7675.1 183.2,7674.9 184.4,7674.8 185.6,7674.7 186.8,7674.6 188.0,7674.4 189.2,7674.3 190.4,7674.2 191.6,7674.0 192.8,7673.9 194.0,7673.7 195.2,7673.6 196.4,7673.5 197.6,7673.3 198.8,7673.2 200.0,7673.0 201.2,7672.9 202.4,7672.7 203.6,7672.6 204.8,7672.4 206.0,7672.3 207.2,7672.1 208.4,7672.0 209.6,7671.8 210.8,7671.7 212.0,7671.5 213.2,7671.3 214.4,7671.2 215.6,7671.0 216.8,7670.9 218.0,7670.7 219.2,7670.5 220.4,7670.4 221.6,7670.2 222.8,7670.0 224.0,7669.9 225.2,7669.7 226.4,7669.5 227.6,7669.4 228.8,7669.2 230.0,7669.0 231.2,7668.9 232.4,7668.7 233.6,7668.5 234.8,7668.4 236.0,7668.2 237.2,7668.0 238.4,7667.8 239.6,7667.7 240.8,7667.5 242.0,7667.3 243.2,7667.1 244.4,7667.0 245.6,7666.8 246.8,7666.6 248.0,7666.4 249.2,7666.3 250.4,7666.1 251.6,7665.9 252.8,7665.7 254.0,7665.6 255.2,7665.4 256.4,7665.2 257.6,7665.0 258.8,7664.8 260.0,7664.7 261.2,7664.5 262.4,7664.3 263.6,7664.1 264.8,7663.9 266.0,7663.7 267.2,7663.6 268.4,7663.4 269.6,7663.2 270.8,7663.0 272.0,7662.8 273.2,7662.6 274.4,7662.5 275.6,7662.3 276.8,7662.1 278.0,7661.9 279.2,7661.7 280.4,7661.5 281.6,7661.3 282.8,7661.2 284.0,7661.0 285.2,7660.8 286.4,7660.6 287.6,7660.4 288.8,7660.2 290.0,7660.0 291.2,7659.8 292.4,7659.7 293.6,7659.5 294.8,7659.3 296.0,7659.1 297.2,7658.9 298.4,7658.7 299.6,7658.5 300.8,7658.3 302.0,7658.1 303.2,7657.9 304.4,7657.8 305.6,7657.6 306.8,7657.4 308.0,7657.2 309.2,7657.0 310.4,7656.8 311.6,7656.6 312.8,7656.4 314.0,7656.2 315.2,7656.0 316.4,7655.8 317.6,7655.6 318.8,7655.5 320.0,7655.3 321.2,7655.1 322.4,7654.9 323.6,7654.7 324.8,7654.5 326.0,7654.3 327.2,7654.1 328.4,7653.9 329.6,7653.7 330.8,7653.5 332.0,7653.3 333.2,7653.1 334.4,7652.9 335.6,7652.7 336.8,7652.5 338.0,7652.4 339.2,7652.2 340.4,7652.0 341.6,7651.8 342.8,7651.6 344.0,7651.4 345.2,7651.2 346.4,7651.0 347.6,7650.8 348.8,7650.6 350.0,7650.4 351.2,7650.2 352.4,7650.0 353.6,7649.8 354.8,7649.6 356.0,7649.4 357.2,7649.2 358.4,7649.0 359.6,7648.8 360.8,7648.6 362.0,7648.4 363.2,7648.2 364.4,7648.0 365.6,7647.8 366.8,7647.6 368.0,7647.4 369.2,7647.2 370.4,7647.0 371.6,7646.8 372.8,7646.6 374.0,7646.5 375.2,7646.3 376.4,7646.1 377.6,7645.9 378.8,7645.7 380.0,7645.5 381.2,7645.3 382.4,7645.1 383.6,7644.9 384.8,7644.7 386.0,7644.5 387.2,7644.3 388.4,7644.1 389.6,7643.9 390.8,7643.7 392.0,7643.5 393.2,7643.3 394.4,7643.1 395.6,7642.9 396.8,7642.7 398.0,7642.5 399.2,7642.3 400.4,7642.1 401.6,7641.9 402.8,7641.7 404.0,7641.5 405.2,7641.3 406.4,7641.1 407.6,7640.9 408.8,7640.7 410.0,7640.5 411.2,7640.3 412.4,7640.1 413.6,7639.9 414.8,7639.7 416.0,7639.5 417.2,7639.3 418.4,7639.1 419.6,7638.9 420.8,7638.7 422.0,7638.5 423.2,7638.3 424.4,7638.1 425.6,7637.9 426.8,7637.7 428.0,7637.5 429.2,7637.3 430.4,7637.1 431.6,7636.9 432.8,7636.7 434.0,7636.5 435.2,7636.3 436.4,7636.1 437.6,7635.9 438.8,7635.7 440.0,7635.5 441.2,7635.3 442.4,7635.0 443.6,7634.8 444.8,7634.6 446.0,7634.4 447.2,7634.2 448.4,7634.0 449.6,7633.8 450.8,7633.6 452.0,7633.4 453.2,7633.2 454.4,7633.0 455.6,7632.8 456.8,7632.6 458.0,7632.4 459.2,7632.2 460.4,7632.0 461.6,7631.8 462.8,7631.6 464.0,7631.4 465.2,7631.2 466.4,7631.0 467.6,7630.8 468.8,7630.6 470.0,7630.4 471.2,7630.2 472.4,7630.0 473.6,7629.8 474.8,7629.6 476.0,7629.4 477.2,7629.2 478.4,7629.0 479.6,7628.8 480.8,7628.6 482.0,7628.4 483.2,7628.2 484.4,7628.0 485.6,7627.8 486.8,7627.6 488.0,7627.4 489.2,7627.2 490.4,7627.0 491.6,7626.8 492.8,7626.5 494.0,7626.3 495.2,7626.1 496.4,7625.9 497.6,7625.7 498.8,7625.5 500.0,7625.3 501.2,7625.1 502.4,7624.9 503.6,7624.7 504.8,7624.5 506.0,7624.3 507.2,7624.1 508.4,7623.9 509.6,7623.7 510.8,7623.5 512.0,7623.3" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<line x1="145.2" y1="7677.4" x2="145.2" y2="7700.0" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="145.2" cy="7677.4" r="4" style="fill:var(--c-success)"/>
<text x="145.2" y="7714" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">13.6</text>
<text x="80.0" y="7714" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="224.0" y="7714" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">30</text>
<text x="368.0" y="7714" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">60</text>
<text x="512.0" y="7714" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">90</text>
<text x="512" y="7730" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">checkpoint interval τ (minutes)</text>
<text x="40" y="7572" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">loss</text>
<rect x="300" y="7579" width="12" height="10" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1"/>
<text x="318" y="7588" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">A · 25 GB/s: at best 24% lost</text>
<rect x="300" y="7596" width="12" height="10" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1"/>
<text x="318" y="7605" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">B · 250 GB/s (GDS): at best 7.5%</text>
<text x="353.6" y="7668.5" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">rewind τ/2M (shared)</text>
<text x="40" y="7748" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">M = 3 h (on 16,384 H100s a failure every ~3 h) · loss = δ/τ + τ/2M</text>
<text x="40" y="7764" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">How to read it: frequent saves lose time to writing, rare saves lose it to rewinding;</text>
<text x="40" y="7779" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">the best interval is the bottom of the dip.</text>
<text x="552" y="7584" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Symptom</text>
<text x="552" y="7602" text-anchor="start" style="fill:var(--c-text);font-size:12px">The cluster stops on every save; a</text>
<text x="552" y="7619" text-anchor="start" style="fill:var(--c-text);font-size:12px">failure rewinds to the last one.</text>
<text x="552" y="7647" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Cause</text>
<text x="552" y="7665" text-anchor="start" style="fill:var(--c-text);font-size:12px">Writing 7.5 TB through the CPU page</text>
<text x="552" y="7682" text-anchor="start" style="fill:var(--c-text);font-size:12px">cache at 25 GB/s takes δ = 307 s, and</text>
<text x="552" y="7699" text-anchor="start" style="fill:var(--c-text);font-size:12px">the cluster waits the whole time.</text>
<text x="552" y="7727" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Fix</text>
<rect x="552.0" y="7737.0" width="116.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="610.4" y="7751.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPUDirect Storage</text>
<rect x="552.0" y="7763.0" width="135.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="619.5" y="7777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">async / burst writes</text>
<rect x="693.0" y="7763.0" width="104.8" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="745.4" y="7777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">pick τ = √(2δM)</text>
<line x1="540.0" y1="7570.0" x2="540.0" y2="7783.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="7809.0" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="7829" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: the three fixes share one idea: move less data, and keep computing while you move it.</text>
<text x="420.0" y="7876" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Let&#x27;s feel these gaps: push the same load through every road.</text>
<line x1="420.0" y1="7884.0" x2="420.0" y2="7904.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="7908" width="808" height="333" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="7932" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="7936.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">D</text>
<text x="56" y="7937" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Speed intuition: how long does 140 GB take?</text>
<text x="28" y="7958" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Llama 3 70B&#x27;s BF16 weights = 140 GB; the same load on every link (log scale).</text>
<rect x="28" y="7970" width="784" height="214" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="200.0" y1="7984.0" x2="200.0" y2="8108.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="200" y="8122" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 ms</text>
<line x1="292.0" y1="7984.0" x2="292.0" y2="8108.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="292" y="8122" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 ms</text>
<line x1="384.0" y1="7984.0" x2="384.0" y2="8108.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="384" y="8122" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 s</text>
<line x1="476.0" y1="7984.0" x2="476.0" y2="8108.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="476" y="8122" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 s</text>
<text x="40" y="8002" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3 (one H100)</text>
<rect x="200" y="7990" width="57.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="263.1" y="8002" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">42 ms</text>
<text x="40" y="8026" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (in node)</text>
<rect x="200" y="8014" width="137.3" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="343.3" y="8026" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0.31 s</text>
<text x="40" y="8050" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU path)</text>
<rect x="200" y="8038" width="215.3" height="16" rx="2" style="fill:var(--c-text-mute);fill-opacity:.45;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="421.3" y="8050" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2.2 s</text>
<text x="40" y="8074" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (per GPU)</text>
<rect x="200" y="8062" width="225.1" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="431.1" y="8074" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2.8 s</text>
<text x="40" y="8098" text-anchor="start" style="fill:var(--c-text);font-size:11px">legacy storage (cluster)</text>
<rect x="200" y="8086" width="252.8" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="458.8" y="8098" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5.6 s</text>
<text x="40" y="8150" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Longer bar = slower: each bar is the time to push the same 140 GB through that link.</text>
<rect x="544.0" y="7984.0" width="256.0" height="80" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554" y="8002" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Human scale: data that arrives from</text>
<text x="554" y="8018" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">HBM in 1 second</text>
<text x="554" y="8034" text-anchor="start" style="fill:var(--c-text);font-size:11px">takes 7.4 s over NVLink, 52 s over PCIe</text>
<text x="554" y="8050" text-anchor="start" style="fill:var(--c-text);font-size:11px">and 67 s over InfiniBand.</text>
<rect x="544.0" y="8074.0" width="256.0" height="96" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="554" y="8092" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Decode&#x27;s ceiling: every token reads</text>
<text x="554" y="8108" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">all the weights once</text>
<text x="554" y="8124" text-anchor="start" style="fill:var(--c-text);font-size:11px">on 8 H100s, 140 GB ÷</text>
<text x="554" y="8140" text-anchor="start" style="fill:var(--c-text);font-size:11px">(8 × 3.35 TB/s) ≈ 5.2 ms → at most</text>
<text x="554" y="8156" text-anchor="start" style="fill:var(--c-text);font-size:11px">~190 tokens/s at batch 1</text>
<line x1="532.0" y1="7984.0" x2="532.0" y2="8170.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="8196.0" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="8216" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: milliseconds in HBM, seconds on the network. Good design keeps data in the fast layer.</text>
<text x="420.0" y="8261" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Last stop: what does all this look like in a real cluster?</text>
<line x1="420.0" y1="8269.0" x2="420.0" y2="8289.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="8293" width="808" height="310" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="8317" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="8321.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">E</text>
<text x="56" y="8322" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">A real cluster: Meta&#x27;s Llama 3 405B run</text>
<text x="28" y="8343" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The same 16,384 H100s in Meta&#x27;s own RoCE cluster; numbers from the Llama 3 report.</text>
<rect x="28.0" y="8355" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="8363.0" width="237.3" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.7" y="8377.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Compute</text>
<text x="42.0" y="8401" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16,384 H100s</text>
<text x="42.0" y="8416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TP 8 × PP 16 × DP 128</text>
<text x="42.0" y="8431" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16M tokens per step</text>
<text x="42.0" y="8446" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">400 TFLOP/s per GPU</text>
<text x="42.0" y="8461" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BF16 MFU 41%</text>
<rect x="293.33333333333337" y="8355" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="301.3" y="8363.0" width="237.3" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="8377.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Network</text>
<text x="307.3" y="8401" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">RoCE · 400 Gb/s/GPU</text>
<text x="307.3" y="8416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod = 3,072 GPUs</text>
<text x="307.3" y="8431" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">full bisection in pod</text>
<text x="307.3" y="8446" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1:7 between 8 pods</text>
<text x="307.3" y="8461" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(smaller models: InfiniBand)</text>
<rect x="558.6666666666667" y="8355" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="566.7" y="8363.0" width="237.3" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="685.3" y="8377.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Storage</text>
<text x="572.7" y="8401" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Tectonic · 240 PB</text>
<text x="572.7" y="8416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7,500 SSD servers</text>
<text x="572.7" y="8431" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 TB/s sustained</text>
<text x="572.7" y="8446" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7 TB/s peak</text>
<text x="572.7" y="8461" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ckpt: 1 MB–4 GB/GPU</text>
<text x="40" y="8497" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">MFU: 43% on 8,192 H100s, 41% on 16,384, 38% at 128K context; scale makes comms grow.</text>
<rect x="28.0" y="8511.0" width="386.0" height="80" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="38" y="8529" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">466 interruptions in 54 days, 419 unexpected → a failure</text>
<text x="38" y="8545" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">every ~3 hours</text>
<text x="38" y="8561" text-anchor="start" style="fill:var(--c-text);font-size:11px">78% hardware (GPU 30.1% · HBM3 17.2%); automation kept effective</text>
<text x="38" y="8577" text-anchor="start" style="fill:var(--c-text);font-size:11px">time &gt; 90%</text>
<rect x="426.0" y="8511.0" width="386.0" height="80" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="436.0" y="8529" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Lesson: the three pillars are designed together.</text>
<text x="436.0" y="8546" text-anchor="start" style="fill:var(--c-text);font-size:11px">The slowest link (HBM, network or disk) sets the whole cluster&#x27;s</text>
<text x="436.0" y="8562" text-anchor="start" style="fill:var(--c-text);font-size:11px">pace; MFU mirrors it.</text>
</svg>

*Figure 1.1: The three pillars at a glance. A: the SM → GPU → DGX node (with its system topology) → SuperPOD scale ladder, a closer look at the SuperPOD's network (port count, traffic paths, scale) and the 14 building blocks that keep the cluster running; B: the path of data through one training step; C: the three walls with computed curves; D: speed intuition with a 140 GB load; E: real numbers from Llama 3 405B. Sources: NVIDIA DGX H100 and DGX SuperPOD documentation, The Llama 3 Herd of Models (2024).*

Read the figure top to bottom; most cards put the drawing on the left and its explanation on the right. Part A zooms out from one SM to a 1,016-GPU SuperPOD in four levels and explains why each hardware and software building block is needed, and Part B numbers the eight stops data makes in a single training step. The three curves in Part C are computed from this article's formulas: the roofline, the bandwidth staircase and the Young-Daly loss curve. Part D pushes the same 140 GB through every link, and Part E shows the three pillars as measured on Meta's Llama 3 405B cluster. The next three sections open the pillars one by one.

---

## 2. Pillar I — Compute: Streaming Multiprocessors, microarchitecture, and execution dynamics

### Silicon-level GPU anatomy: SMs, warp schedulers, and tensor cores

To grasp why GPUs behave so differently from TPUs, consider a commercial kitchen analogy. A Google TPU v5p behaves like a high-end French brigade run by just two colossal master chefs (the two TensorCore units). Each chef handles enormous cutting boards (Matrix Multiply Units, or MXUs) capable of processing 128x128 matrices in lockstep; if you hand them an irregular task or small batches, their massive tools sit idle. A modern NVIDIA GPU (such as Hopper H100 or Blackwell B200) is instead an army of 132 to 148 short-order line cooks, termed **Streaming Multiprocessors (SMs)**. Each line cook operates independently, executing separate tasks on modest prep stations, yet all sharing a central pantry.

In hardware, an SM is broken down into four identical quadrants called **SM subpartitions**. Each subpartition contains:
1. **A Tensor Core (TC):** A specialized systolic matrix multiplication engine. On an H100 running at 1.76 GHz, each Tensor Core crunches roughly 1,024 BF16 floating-point operations per clock cycle—equivalent to an 8x8x8 matrix multiplication. Across 132 SMs (528 Tensor Cores total), this yields the headline 990 TFLOP/s BF16 dense throughput. On Blackwell B200, this expands to 2,048 FLOPs per cycle, reaching 2,250 TFLOP/s dense BF16.
2. **A Warp Scheduler and Dispatch Unit:** Responsible for issuing instructions to sets of 32 threads, collectively known as a **warp**. An SM can keep up to 64 resident warps in flight simultaneously, rapidly context-switching between them on every clock cycle to hide memory load latencies.
3. **32 FP32 CUDA Cores (ALUs):** General-purpose vector execution units that compute elementwise activations (GELU, SiLU), normalizations (RMSNorm), and additions. Across the 132 SMs of an H100, there are 16,896 CUDA cores running at ~1.98 GHz boost clock.
4. **16,384 32-bit Registers:** Providing 64 KiB of ultra-fast register file per subpartition (256 KiB total per SM, amounting to ~33 MB chip-wide).

The architectural layout of a modern GPU and its internal SM breakdown are illustrated in the following structural diagram:

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

Beyond this abstract view, NVIDIA's actual GH100 silicon die layout and microarchitectural schematics reveal the physical placement of these compute engines:

![NVIDIA GPU Chip Layout and SM Microarchitecture](../../assets/img/gpu/gpu-diagram.png)
*Figure 2.1: Abstract silicon layout of an H100/B200 GPU containing 132 to 148 Streaming Multiprocessors connected to High Bandwidth Memory, as described in the JAX-ML Scaling Book.*

![NVIDIA H100 SM Subpartition Architecture](../../assets/img/gpu/blackwell-sm.png)
*Figure 2.2: Detailed internal schematic of an H100 SM showing the four subpartitions, register files, Warp Schedulers, Tensor Cores, and the shared 256 KiB L1/SMEM data cache.*

### SIMT execution dynamics and the threat of warp divergence

Unlike the TPU's pure SIMD (Single Instruction, Multiple Data) execution, NVIDIA GPUs operate under the **SIMT (Single Instruction, Multiple Threads)** paradigm. While all 32 CUDA cores within a warp execute the same instruction in a given cycle, each thread maintains its own program counter and private register space.

This flexibility introduces a perilous hardware penalty known as **warp divergence**. If your kernel contains a conditional branch (such as an `if/else` condition checking for MoE expert routing assignment or dynamic sequence padding), threads taking the branch cannot proceed independently. Instead, the warp scheduler executes the `if` path while masking out (stalling) the remaining cores, and then executes the `else` path while stalling the first set:

![Warp Divergence in NVIDIA GPUs](../../assets/img/gpu/warp-divergence.png)
*Figure 2.3: Hardware timeline of warp divergence. When threads within a 32-thread warp diverge along branch paths, inactive execution lanes are masked, cutting hardware efficiency in half.*

If 16 threads take branch A and 16 threads take branch B, execution time doubles and effective arithmetic throughput drops by exactly 50%.

<details>
<summary>Quiz: What is the arithmetic throughput penalty of warp divergence?</summary>

When half of the 32 threads in a warp follow a `then` branch while the other half follow an `else` branch, the hardware must serialize the two paths. During the first phase, 16 ALUs are active and 16 are masked; during the second phase, the reverse occurs. Total wall-clock time doubles, resulting in an exact 50% drop in sustained arithmetic throughput for that warp.
</details>

### Memory hierarchy and the arithmetic intensity roofline

Computation is meaningless without the memory bandwidth required to feed the execution units. The GPU memory subsystem forms a steep pyramid:
- **Register File:** 256 KiB per SM (~33 MB chip-wide). Single-cycle access latency.
- **Shared Memory (SMEM) / L1 Cache:** 256 KiB per SM (~33 MB chip-wide). Programmatically managed SRAM with sub-10 ns latency, critical for caching attention tiles in FlashAttention.
- **Tensor Memory (TMEM):** Introduced in Blackwell B200 (256 KiB per SM). Because B200 Tensor Cores produce matrix accumulators so massive they exceeded traditional register capacity, TMEM provides dedicated on-chip storage for matrix products.
- **Shared L2 Cache:** 50 MB on Hopper H100, 126 MB on Blackwell B200. Delivers roughly 5.5 TB/s of bidirectional bandwidth.
- **High Bandwidth Memory (HBM3 / HBM3e):** 80 GB on H100 (3.35 TB/s), 141 GB on H200 (4.8 TB/s), and 192 GB on B200 (8.0 TB/s).

The balance between compute capability and memory throughput is formalized by the Roofline Model:

$$\alpha_{\text{crit}} = \frac{C_{\text{peak}}}{W_{\text{mem}}}$$

Read this as: $\alpha_{\text{crit}}$ is the critical arithmetic intensity in FLOPs per byte, $C_{\text{peak}}$ is peak arithmetic throughput in FLOPs per second, and $W_{\text{mem}}$ is memory bandwidth in bytes per second.

For an H100 SXM5 running BF16 Tensor Core operations:
$$C_{\text{peak}} = 990 \times 10^{12} \text{ FLOP/s}, \quad W_{\text{mem}} = 3.35 \times 10^{12} \text{ B/s}$$
$$\alpha_{\text{crit}} = \frac{990 \times 10^{12}}{3.35 \times 10^{12}} \approx 295.5 \text{ FLOP/byte}$$

For a matrix multiplication multiplying an activation matrix $X \in \mathbb{R}^{B \times D}$ by a weight matrix $W \in \mathbb{R}^{D \times F}$ in 16-bit precision:
- **Total Math Operations:** $2 B D F \text{ FLOPs}$
- **Total Memory Transferred:** $2(BD + DF + BF) \text{ bytes}$
- **Operational Intensity:**

$$\alpha(B) = \frac{2 B D F}{2(BD + DF + BF)} = \frac{B D F}{B(D + F) + D F}$$

When the batch size $B$ is tiny ($B \to 1$, such as single-sequence auto-regressive decoding), the weight matrix dominates memory traffic ($DF \gg B(D+F)$), yielding:
$$\alpha(1) \approx \frac{D F}{D F} \cdot B \approx B = 1 \text{ FLOP/byte} \ll 295.5 \text{ FLOP/byte}$$

In this regime, the GPU is severely memory-bandwidth bound, utilizing less than 1% of its peak compute capability. To become compute-bound, the batch size must satisfy:

$$B \ge \frac{\alpha_{\text{crit}} \cdot D F}{D F - \alpha_{\text{crit}}(D + F)} \approx \alpha_{\text{crit}} \approx 295 \text{ tokens}$$

On both Hopper H100 and Blackwell B200, you require a batch size of roughly 280 to 300 tokens per SM to cross the roofline knee into the compute-bound zone.

<details>
<summary>Quiz: Why does auto-regressive token generation achieve less than 2% MFU?</summary>

In single-token decoding ($B=1$), the GPU must stream all model weights from HBM to register files to process a single vector. The arithmetic intensity is $\alpha(1) \approx 1 \text{ FLOP/byte}$. Because the H100 requires $\alpha_{\text{crit}} \approx 295.5 \text{ FLOP/byte}$ to saturate its Tensor Cores, the execution units spend over 99% of their clock cycles waiting for memory buses, capping MFU below 1-2%.
</details>

### Low-precision numerical formats: From FP32 to FP8 and NVFP4

To mitigate memory pressure and double arithmetic throughput, modern GPU generations have aggressively reduced numerical precision. Moving from FP32 (32 bits: 1 sign, 8 exponent, 23 mantissa) to 16-bit formats (FP16 and BF16) halved memory footprint and doubled FLOP rates.

The historical evolution of Tensor Core compute engines across architecture generations is summarized below:

| Architecture Generation | Compute Engine | HBM Memory & Bandwidth | BF16 / FP16 TFLOP/s | FP8 / INT8 TFLOP/s | FP4 TFLOP/s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Volta (V100)** | 1st Gen Tensor Core | 32 GB @ 0.9 TB/s | 125 | — | — |
| **Ampere (A100)** | 3rd Gen Tensor Core | 80 GB @ 2.0 TB/s | 312 | 624 | — |
| **Hopper (H100)** | 4th Gen Tensor Core | 80 GB @ 3.35 TB/s | 990 | 1,980 | — |
| **Hopper (H200)** | 4th Gen Tensor Core | 141 GB @ 4.8 TB/s | 990 | 1,980 | — |
| **Blackwell (B200)** | 5th Gen Tensor Core | 192 GB @ 8.0 TB/s | 2,250 | 4,500 | 9,000 |

In Hopper H100, NVIDIA introduced native hardware support for FP8 (8-bit floating point), split into two formats:
1. **E4M3 (1 sign, 4 exponent, 3 mantissa bits):** Offers higher numerical precision for forward activations and weights, where dynamic range is well-behaved.
2. **E5M2 (1 sign, 5 exponent, 2 mantissa bits):** Matches the dynamic range of FP16, preventing underflow during backward gradient passes at the expense of mantissa precision.

| Numerical Format | Sign Bit | Exponent Bits | Mantissa Bits | Dynamic Range ($E_{\text{min}} \dots E_{\text{max}}$) | Primary Role in LLM Scaling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IEEE FP16** | 1 | 5 | 10 | $\approx 6 \times 10^{-5} \dots 65,504$ | Legacy inference and baseline compute |
| **Google BF16** | 1 | 8 | 7 | $\approx 1 \times 10^{-38} \dots 3 \times 10^{38}$ | Robust foundation model pre-training |
| **FP8 (E4M3)** | 1 | 4 | 3 | $\approx \pm 448$ | Forward pass activations & weights |
| **FP8 (E5M2)** | 1 | 5 | 2 | $\approx \pm 57,344$ | Backward pass gradient propagation |
| **NVFP4 (E2M1)** | 1 | 2 | 1 | Block-scaled | Blackwell ultra-dense GEMM acceleration |

With Blackwell B200, NVIDIA advanced further with **NVFP4 (4-bit Microscaling)**. By using 4-bit numbers paired with fine-grained 8-bit or 16-bit scale factors across 16-element vectors, B200 delivers a staggering 4,500 TFLOP/s dense compute per chip, enabling massive matrix multiplications to run entirely within on-chip TMEM and L2 cache boundaries.

---

## 3. Pillar II — Network: Interconnect fabrics, collectives, and multi-dimensional parallelism

### Intra-node networking: NVLink and NVSwitch fabrics

When scaling beyond a single chip, standard PCIe buses (delivering a paltry 64 GB/s on PCIe Gen 5) become an intolerable choke point. NVIDIA bypasses PCIe through **NVLink** and **NVSwitch**.

Inside a standard DGX H100 or HGX H100 chassis, 8 GPUs are paired with 4 custom NVSwitch Gen 4 ASICs. Each H100 chip features 18 NVLink 4 ports. Each port transmits at 25 GB/s in each direction (50 GB/s bidirectional), aggregating to:
$$W_{\text{GPU egress}} = 18 \times 25 \text{ GB/s} = 450 \text{ GB/s unidirectional } (900 \text{ GB/s bidirectional})$$

The 4 central NVSwitches form a non-blocking crossbar fabric. The total switching capacity inside the node is:
$$4 \text{ switches} \times 1.6 \text{ TB/s} = 6.4 \text{ TB/s switching bandwidth}$$

Because each of the 8 GPUs can egress 450 GB/s, any even bisection partition of the node (splitting 4 GPUs against 4 GPUs) transmits:
$$\text{Bisection Bandwidth} = 4 \text{ GPUs} \times 450 \text{ GB/s} \times 2 \text{ directions} = 3.6 \text{ TB/s}$$

This intra-node crossbar network architecture is shown in the following system diagram:

<svg viewBox="0 0 560 410" role="img" aria-label="8-GPU DGX node NVSwitch crossbar interconnect architecture and bandwidth breakdown." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="nv-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="nv-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="nv-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Intra-Node NVLink &amp; NVSwitch Architecture</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8x GPU NVLink Domain (Hopper / Blackwell)</text>
<rect x="16" y="48" width="528" height="276" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="66" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX H100 / B200 Node</text>
<text x="532" y="66" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">Node Bisection: 3.6 TB/s</text>
<rect x="32" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="81.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 0</text>
<text x="81.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="160" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="209.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 1</text>
<text x="209.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="288" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="337.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 2</text>
<text x="337.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="416" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 3</text>
<text x="465.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="32" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 0</text>
<text x="81.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1.6 TB/s Unidir</text>
<rect x="160" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="209.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 1</text>
<text x="209.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1.6 TB/s Unidir</text>
<rect x="288" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="337.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 2</text>
<text x="337.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1.6 TB/s Unidir</text>
<rect x="416" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="465.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 3</text>
<text x="465.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1.6 TB/s Unidir</text>
<rect x="32" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="81.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 4</text>
<text x="81.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="160" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="209.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 5</text>
<text x="209.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="288" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="337.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 6</text>
<text x="337.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="416" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 7</text>
<text x="465.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<line x1="81.0" y1="132" x2="81.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="81.0" y1="204" x2="81.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="209.0" y1="132" x2="209.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="209.0" y1="204" x2="209.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="337.0" y1="132" x2="337.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="337.0" y1="204" x2="337.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="465.0" y1="132" x2="465.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="465.0" y1="204" x2="465.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="97" y1="132" x2="195" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="225" y1="132" x2="323" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="353" y1="132" x2="451" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="97" y1="230" x2="195" y2="204" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="225" y1="230" x2="323" y2="204" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="353" y1="230" x2="451" y2="204" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="16" y="332" width="528" height="66" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="352" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">NVSwitch Crossbar: Non-blocking all-to-all connectivity across all 8 GPUs.</text>
<text x="28" y="370" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Each GPU accesses all peer memories directly at 450 GB/s unidirectional,</text>
<text x="28" y="386" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">delivering 900 GB/s bidirectional full bisection bandwidth.</text>
</svg>

The physical hardware packaging of these 8-GPU baseboards and the internal crossbar routing of the NVSwitch 4 ASIC are documented below:

![NVIDIA HGX Intra-Node NVLink Topology](../../assets/img/gpu/nvlink-nodes.png)
*Figure 3.1: Physical architecture of the HGX 8-GPU baseboard, routing high-density NVLink traces between GPUs and the 4 onboard NVSwitches.*

![NVIDIA NVLink 4 Switch Crossbar Architecture](../../assets/img/gpu/nvlink4.png)
*Figure 3.2: Internal crossbar schematic of the 64-port NVLink 4 switch ASIC presented at Hot Chips 34, capable of routing 3.2 TB/s of bi-directional packet traffic with hardware reduction capability.*

With the Blackwell generation, NVIDIA fundamentally altered this equation with the **GB200 NVL72**. Instead of confining the NVLink domain to an 8-GPU chassis, liquid-cooled racks integrate 72 B200 GPUs interconnected by 18 NVSwitch 5 ASICs over copper backplanes:

![GB200 NVL72 Rack Architecture](../../assets/img/gpu/b200-node.png)
*Figure 3.3: Construction of a GB200 NVL72 rack, integrating 72 GPUs and 18 NVLink switches into a single, unified 900 GB/s NVLink domain.*

In NVL72, all 72 GPUs share a single NVLink domain where any chip can access any other chip's memory at a staggering 900 GB/s unidirectional rate, expanding the single-node boundary by 9x.

<details>
<summary>Quiz: How many NVLink ports does an H100 SXM5 GPU have, and what is its aggregate throughput?</summary>

An H100 SXM5 features 18 NVLink 4 ports. Each port operates at 25 GB/s per direction, yielding 450 GB/s unidirectional and 900 GB/s bidirectional aggregate throughput per chip.
</details>

### Scaling beyond the node: SuperPods and rail-optimized fat-tree topologies

To train modern models with hundreds of billions of parameters, clusters must scale beyond a single node to thousands of GPUs. This scale-out tier is powered by **InfiniBand** (or high-performance RoCE v2 Ethernet).

In a 1,024-GPU DGX H100 SuperPod:
- The cluster is partitioned into **Scalable Units (SUs)**, each containing 32 DGX nodes (256 GPUs).
- Each node houses 8 ConnectX-7 400 Gbps (NDR) NICs, providing $8 \times 50 \text{ GB/s} = 400 \text{ GB/s}$ of scale-out network egress per node (or 50 GB/s per GPU).
- Each SU contains 8 Leaf switches. Across 4 SUs, 16 central Spine switches connect the fabric into a non-blocking 1:1 **Fat-Tree topology** providing 51.2 TB/s of aggregate switching capacity.

Crucially, modern clusters employ **Rail-Optimized wiring**. In this topology, GPU rank $i$ in every node is wired to the exact same Leaf switch $i$. When running data-parallel or tensor-parallel collective operations across GPUs with the same index, packets travel purely within their designated rail without crossing the spine switches, eliminating spine hops and preventing fabric congestion:

<svg viewBox="0 0 560 440" role="img" aria-label="1024-GPU DGX H100 SuperPod Fat-Tree InfiniBand network topology diagram." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sp-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="sp-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">SuperPod Fat-Tree Scaling Topology</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1024 GPUs · Full Bisection Bandwidth</text>
<rect x="16" y="48" width="528" height="62" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280" y="68" text-anchor="middle" style="fill:var(--c-accent-2);font-size:15px;font-weight:700;letter-spacing:.06em">16x InfiniBand Spine Switches</text>
<text x="280" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Quantum-2 400 Gbps NDR · 51.2 TB/s Aggregate Switching Capacity</text>
<text x="280" y="99" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">Full bisection 12.8 TB/s link per Scalable Unit</text>
<rect x="16" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="78.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 0</text>
<text x="78.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switches</text>
<text x="78.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12.8 TB/s Egress</text>
<rect x="150" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="212.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 1</text>
<text x="212.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switches</text>
<text x="212.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12.8 TB/s Egress</text>
<rect x="284" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="346.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 2</text>
<text x="346.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switches</text>
<text x="346.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12.8 TB/s Egress</text>
<rect x="418" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="480.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 3</text>
<text x="480.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switches</text>
<text x="480.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12.8 TB/s Egress</text>
<line x1="78.0" y1="112" x2="78.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="212.0" y1="112" x2="212.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="346.0" y1="112" x2="346.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="480.0" y1="112" x2="480.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="280" y="131" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">Full Bisection (1:1 Non-blocking)</text>
<rect x="16" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="78.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Nodes (256 Chips)</text>
<text x="78.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPUs per node</text>
<text x="78.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NICs</text>
<text x="78.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<rect x="150" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="212.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Nodes (256 Chips)</text>
<text x="212.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPUs per node</text>
<text x="212.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NICs</text>
<text x="212.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<rect x="284" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="346.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Nodes (256 Chips)</text>
<text x="346.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPUs per node</text>
<text x="346.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NICs</text>
<text x="346.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<rect x="418" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="480.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Nodes (256 Chips)</text>
<text x="480.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPUs per node</text>
<text x="480.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NICs</text>
<text x="480.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<line x1="78.0" y1="214" x2="78.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="212.0" y1="214" x2="212.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="346.0" y1="214" x2="346.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="480.0" y1="214" x2="480.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="280" y="234" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">400 GB/s Constant Node Egress</text>
<rect x="16" y="336" width="528" height="88" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="356" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Fat-Tree: Upward cable counts preserve bandwidth across layers.</text>
<text x="28" y="374" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">While GPUs have 450 GB/s NVLink intra-node,</text>
<text x="28" y="390" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">nodes maintain 400 GB/s InfiniBand inter-node across the cluster.</text>
<text x="28" y="408" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Zero oversubscription: collective communication scales without fabric bottleneck.</text>
</svg>

The official structural topology of the H100 and GB200 SuperPods is detailed below:

![NVIDIA SuperPod InfiniBand Network Topology](../../assets/img/gpu/superpod-diagram.png)
*Figure 3.4: Logical layout of an NVIDIA SuperPod showing leaf-and-spine connections between compute nodes and storage fabrics.*

![DGX H100 SuperPod Cluster Architecture](../../assets/img/gpu/h100-superpod.png)
*Figure 3.5: Physical deployment of a 1,024-GPU H100 SuperPod consisting of 128 DGX nodes, rail-optimized InfiniBand switches, and dedicated management racks.*

![GB200 NVL72 SuperPod Architecture](../../assets/img/gpu/gb200-superpod.png)
*Figure 3.6: Multi-rack scale-out topology of the Blackwell GB200 NVL72 SuperPod, connecting 576 B200 GPUs over Quantum-X800 800 Gbps InfiniBand switches.*

### Collective communication: From ring algorithms to in-network SHARP

Distributed training relies on fundamental collective communication primitives implemented in NCCL (NVIDIA Collective Communications Library):
1. **AllGather:** Each rank holds a tensor slice of size $B/N$ and gathers all slices to assemble the complete tensor of size $B$.
2. **ReduceScatter:** Each rank holds an input tensor of size $B$; elements are summed elementwise across ranks and scattered so rank $i$ holds only the $i$-th slice of size $B/N$.
3. **AllReduce:** Equivalent to ReduceScatter followed by AllGather. Computes the elementwise sum across all ranks and replicates the final tensor of size $B$ across all $N$ GPUs.
4. **AllToAll:** Each rank transmits a distinct tensor slice to every other rank, forming the backbone of Mixture of Experts (MoE) dispatching.

The flow of an AllGather collective is illustrated in the following animation:

![AllGather Collective Communication Step](../../assets/img/gpu/all-gather.gif)
*Figure 3.7: Step-by-step communication pattern of an AllGather collective across four participating GPUs.*

In a standard **Ring AllGather** running across $N$ devices with network egress bandwidth $W$:
- Each device divides its tensor into $N$ chunks of size $B/N$.
- In each step, every rank sends one chunk to its neighbor while receiving another.
- The ring takes $N-1$ steps to complete:

$$T_{\text{AllGather, Ring}} = (N - 1) \cdot \frac{B / N}{W} = \frac{N - 1}{N} \cdot \frac{B}{W} \approx \frac{B}{W}$$

Because an AllReduce requires both a ReduceScatter and an AllGather, a classical ring AllReduce transfers $2 \times \frac{N-1}{N} B$ bytes, taking:
$$T_{\text{AllReduce, Ring}} \approx 2 \cdot \frac{B}{W}$$

To eliminate this $2\times$ overhead, modern InfiniBand switches employ **Mellanox SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)**. SHARP embeds floating-point arithmetic hardware directly into the Quantum switch ASICs:

<svg viewBox="0 0 560 400" role="img" aria-label="Ring-based collective communication versus Quantum switch in-network SHARP reduction." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="col-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="col-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="col-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="col-arr-suc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Collective Communication Primitives</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Ring AllGather vs In-Network SHARP AllReduce</text>
<rect x="16" y="48" width="256" height="276" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="68" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Classical Ring Collective</text>
<text x="144" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">AllGather &amp; ReduceScatter (NCCL)</text>
<rect x="48" y="102" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="78" y="127" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 0</text>
<rect x="180" y="102" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="210" y="127" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 1</text>
<rect x="180" y="202" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="210" y="227" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 2</text>
<rect x="48" y="202" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="78" y="227" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 3</text>
<line x1="110" y1="123" x2="178" y2="123" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="144" y="117" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Chunk</text>
<line x1="210" y1="146" x2="210" y2="200" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="224" y="176" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Chunk</text>
<line x1="178" y1="223" x2="110" y2="223" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="144" y="237" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Chunk</text>
<line x1="78" y1="200" x2="78" y2="146" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="64" y="176" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Chunk</text>
<text x="144" y="270" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Time: T ≈ B / W_egress</text>
<text x="144" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Circulates in N-1 hops</text>
<text x="144" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2x transfer cost for AllReduce (RS + AG)</text>
<rect x="288" y="48" width="256" height="276" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="68" text-anchor="middle" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">In-Network Reduction (SHARP)</text>
<text x="416" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Quantum-2 Hardware Summation</text>
<rect x="348" y="104" width="136" height="46" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="123" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Quantum Switch</text>
<text x="416" y="138" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10px;font-weight:600">In-ASIC FP Aggregation</text>
<rect x="300" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="326" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G0</text>
<rect x="358" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="384" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G1</text>
<rect x="416" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="442" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G2</text>
<rect x="474" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="500" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G3</text>
<line x1="326" y1="202" x2="380" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<line x1="384" y1="202" x2="380" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<line x1="442" y1="202" x2="450" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<line x1="500" y1="202" x2="450" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<text x="416" y="180" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">Single-phase sum and multicast</text>
<text x="416" y="270" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Time: T ≈ B / W_egress (1x)</text>
<text x="416" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Cuts network traffic by 50%</text>
<text x="416" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Halves AllReduce latency</text>
<rect x="16" y="334" width="528" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="354" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Intra-node NVLink AllGather time: T ≈ B / 450 GB/s (H100).</text>
<text x="28" y="372" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">With SHARP active, AllReduce matches single-pass transfer time instead of 2x (RS + AG).</text>
</svg>

Instead of circulating tensors around a ring, each GPU transmits its tensor up to the switch once. The switch ASIC accumulates the numbers in hardware and multicasts the reduced tensor back down to all GPUs in a single pass:

$$T_{\text{AllReduce, SHARP}} \approx \frac{B}{W}$$

SHARP cuts network traffic by exactly 50% and slashes AllReduce latency in half.

Empirical bandwidth curves measured across varying tensor sizes on GPU and TPU clusters highlight how bandwidth scales with buffer volume:

![GPU AllGather Bandwidth vs Buffer Size](../../assets/img/gpu/gpu-all-gather-bw.png)
*Figure 3.8: Empirical throughput of AllGather on H100 clusters as a function of tensor size, showing saturation above 16 MB.*

![GPU AllReduce Bandwidth vs Buffer Size](../../assets/img/gpu/gpu-all-reduce-bw.png)
*Figure 3.9: Empirical throughput of AllReduce collectives over intra-node NVLink and scale-out InfiniBand.*

![TPU AllGather Bandwidth vs Buffer Size](../../assets/img/gpu/tpu-all-gather-bw.png)
*Figure 3.10: AllGather throughput on Google TPU v4/v5p 3D torus fabrics for comparison.*

![TPU AllReduce Bandwidth vs Buffer Size](../../assets/img/gpu/tpu-all-reduce-bw.png)
*Figure 3.11: AllReduce throughput on Google TPU clusters.*

The latency profiles of sharded AllGather and SHARP hardware acceleration are compared below:

![Sharded AllGather Communication Cost](../../assets/img/gpu/sharded-all-gather-cost.png)
*Figure 3.12: Mathematical latency breakdown of sharded AllGather operations across increasing cluster scales.*

![Mellanox SHARP Tree-Based Reduction Algorithm](../../assets/img/gpu/sharp-algorithm.png)
*Figure 3.13: In-network aggregation tree executed by Mellanox Quantum switch ASICs during a SHARP AllReduce.*

![SHARP AllReduce Acceleration vs Ring Collective](../../assets/img/gpu/sharp-all-reduce-cost.png)
*Figure 3.14: Measured latency reduction of SHARP in-network compute compared to traditional two-pass host ring algorithms.*

For Mixture of Experts (MoE) architectures, token routing relies on irregular, sparse **AllToAll** communication:

![Sparse MoE AllToAll Communication Cost Approximation](../../assets/img/gpu/all-to-all-approx.png)
*Figure 3.15: Comparison between combinatorial expectation and analytical approximation for sparse MoE token dispatching.*

<details>
<summary>Quiz: How much network traffic does Mellanox SHARP eliminate compared to a classical ring AllReduce?</summary>

A classical ring AllReduce requires two passes (a ReduceScatter transferring $\frac{N-1}{N}B$ bytes, followed by an AllGather transferring $\frac{N-1}{N}B$ bytes), moving a total of nearly $2B$ bytes per GPU. SHARP reduces tensors in-flight inside the switch ASICs and multicasts the result in a single pass ($B$ bytes), cutting total wire traffic by exactly 50%.
</details>

### 4D parallelism rooflines for LLM scaling

Training frontier models requires combining four distinct parallelism dimensions: **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Expert Parallelism (EP)**, and **Data Parallelism (DP / FSDP)**.

#### Tensor Parallelism (TP) and the 8-GPU NVLink Ceiling

In Megatron-LM style tensor parallelism, the Transformer MLP block is sliced across $Y$ GPUs: $W_{\text{in}}$ is split column-wise ($[D, F/Y]$), and $W_{\text{out}}$ is split row-wise ($[F/Y, D]$).
- **Forward Math Time:** $T_{\text{math}} = \frac{4 B D F}{Y \cdot C_{\text{peak}}}$
- **Communication Time:** One AllReduce on activations: $T_{\text{comms}} = \frac{4 B D}{W_{\text{collective}}}$

For computation to stay faster than communication ($T_{\text{math}} \ge T_{\text{comms}}$):

$$Y \le \frac{F \cdot W_{\text{collective}}}{C_{\text{peak}}} = \frac{F}{\alpha_{\text{crit}}}$$

For an H100 using intra-node NVLink ($W_{\text{collective}} = 450 \text{ GB/s}$, $C_{\text{peak}} = 990 \text{ TFLOP/s}$):
$$\frac{C_{\text{peak}}}{W_{\text{collective}}} = \frac{990 \times 10^{12}}{450 \times 10^9} = 2,200$$

For LLaMA-3 70B ($F = 28,672$):
$$Y \le \frac{28,672}{2,200} \approx 13.03 \implies Y = 8 \text{ GPUs}$$

This mathematical limit proves **why Tensor Parallelism is strictly confined to the 8 GPUs inside a single NVLink chassis**. Attempting to run TP across nodes over InfiniBand ($W_{\text{collective}} = 400 \text{ GB/s}$ node egress, or 50 GB/s per GPU) inflates $C/W$ to 19,800, dropping allowable $Y$ below 1.5 and causing MFU to collapse.

#### Pipeline Parallelism (PP) and Pipeline Bubbles

Pipeline parallelism partitions model layers across $P$ pipeline stages. Adjacent stages exchange boundary activations of size $2BD / N_{\text{MB}}$ per microbatch over simple peer-to-peer links, making network volume trivial.

However, pipelining introduces the **Pipeline Bubble**—idle GPU time while waiting for microbatches to fill and drain the pipeline:

$$\text{Bubble Fraction} = \frac{P - 1}{P - 1 + M}$$

Read this as: $P$ is the number of pipeline stages, and $M$ is the number of microbatches per global batch.

![Pipeline Bubble Schedule in 1F1B](../../assets/img/gpu/pipeline-bubble.png)
*Figure 3.16: Pipeline execution schedule with 2 stages and 2 microbatches. White blocks represent idle bubble cycles, while 'G' represents gradient synchronization.*

![Activation Transfer Delay Impact on Pipeline Bubbles](../../assets/img/gpu/pipeline-transfer.png)
*Figure 3.17: Inter-stage activation transfer latency shifts pipeline execution, widening the bubble penalty.*

#### Data Parallelism (DP & FSDP) and Critical Batch Size

In Data Parallelism (including ZeRO-3 / FSDP), model weights and gradients are replicated or sharded across $X$ devices:
$$T_{\text{math}} = \frac{6 B D F}{X \cdot C_{\text{peak}}}, \quad T_{\text{comms}} = \frac{2 \cdot 2 \cdot DF \cdot (X - 1)}{X \cdot W_{\text{collective}}}$$

From $T_{\text{math}} \ge T_{\text{comms}}$, we derive the **Critical Local Batch Size**:

$$\frac{B}{X} \ge \frac{2}{3} \cdot \frac{C_{\text{peak}}}{W_{\text{collective}}} \approx 2,475 \text{ tokens per GPU}$$

If the per-GPU batch size falls below ~2,500 tokens, scale-out gradient synchronization takes longer than forward-backward math, and GPUs idle.

For Mixture of Experts (MoE) with $E$ total experts and $k$ active experts per token, this threshold expands dramatically:

$$\left(\frac{B}{X}\right)_{\text{crit, MoE}} \ge \frac{E}{k} \cdot \frac{C_{\text{peak}}}{W_{\text{collective}}}$$

In an architecture with $E = 128$ and $k = 4$, the critical batch size is multiplied by $32\times$, demanding over 79,000 tokens per GPU—an impossible requirement unless expert parameters are decoupled via Expert Parallelism (EP).

The four-dimensional parallelism decision tree and its governing roofline boundaries are summarized in the following structural matrix:

<svg viewBox="0 0 560 440" role="img" aria-label="LLM 4D parallelism decision matrix: Tensor, Pipeline, Expert, and Data Parallelism rooflines." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="par-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="par-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="par-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">LLM Parallelism Matrix and Roofline Trade-offs</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TP · PP · EP · DP / FSDP Decision Map</text>
<rect x="16" y="48" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="30" y="68" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Tensor Parallelism (TP)</text>
<text x="30" y="84" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Scope: Single Node (NVLink)</text>
<rect x="30" y="92" width="228" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="144.0" y="104.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">AllGather + ReduceScatter</text>
<text x="30" y="128" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Rule: Y ≤ F / 2475 (LLaMA: Y ≤ 8)</text>
<text x="30" y="146" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Cross-node TP quickly becomes comms-bound</text>
<text x="30" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Compute vs Communication Roofline Bound</text>
<rect x="288" y="48" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="302" y="68" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">Pipeline Parallelism (PP)</text>
<text x="302" y="84" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Scope: Cross-Node Layer Stages</text>
<rect x="302" y="92" width="228" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="416.0" y="104.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">P2P Activation Passing</text>
<text x="302" y="128" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Rule: Bubble = (P-1) / (P-1+M)</text>
<text x="302" y="146" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Minimal wire cost, complex schedules</text>
<text x="302" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Compute vs Communication Roofline Bound</text>
<rect x="16" y="192" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="30" y="212" text-anchor="start" style="fill:var(--c-warn);font-size:12px;font-weight:700">Expert Parallelism (EP)</text>
<text x="30" y="228" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Scope: MoE (1-8 Nodes)</text>
<rect x="30" y="236" width="228" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="144.0" y="248.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Ragged AllToAll Routing</text>
<text x="30" y="272" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Rule: 1-2 nodes if small F; multi-node if large</text>
<text x="30" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Solves parameter memory explosion</text>
<text x="30" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Compute vs Communication Roofline Bound</text>
<rect x="288" y="192" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="302" y="212" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Data Parallelism (DP / FSDP)</text>
<text x="302" y="228" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Scope: Full Cluster (ZeRO-1 / 3)</text>
<rect x="302" y="236" width="228" height="19" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="416.0" y="248.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">AllReduce / ReduceScatter</text>
<text x="302" y="272" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Rule: B_local ≥ 2500 tokens / GPU</text>
<text x="302" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Critical batch size inflates by E/k for MoE</text>
<text x="302" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Compute vs Communication Roofline Bound</text>
<rect x="16" y="336" width="528" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="356" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Parallelism Recipes of Frontier Models:</text>
<text x="28" y="374" text-anchor="start" style="fill:var(--c-accent-2);font-size:10.5px;font-family:var(--font-mono)">DeepSeek-V3 (2048x H800): 64-way EP (8 nodes) + 16-way PP + 2-way DP (ZeRO-1)</text>
<text x="28" y="392" text-anchor="start" style="fill:var(--c-accent);font-size:10.5px;font-family:var(--font-mono)">LLaMA-3 405B (16k H100):  8-way TP (1 node) + 16-way PP + 128-way DP (ZeRO-1)</text>
<text x="28" y="410" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">Principle: TP stays intra-node, PP and EP cross nodes, DP scales across the cluster.</text>
</svg>

Frontier models conquer this trade-off space using distinct production recipes:
- **LLaMA-3 405B (16,384 H100):** 8-way intra-node TP $\times$ 16-way PP $\times$ 128-way DP (ZeRO-1).
- **DeepSeek-V3 (2,048 H800):** 64-way EP (spanning 8 nodes over InfiniBand) $\times$ 16-way PP $\times$ 2-way DP (ZeRO-1), with 30,720 tokens per GPU.

The hyperparameter space of 3D parallelism explored by Megatron-LM is illustrated below:

![Megatron-LM 3D Parallelism Tuning Space](../../assets/img/gpu/megatron-hparams.png)
*Figure 3.18: Parameter efficiency landscapes of TP, PP, and DP across varying cluster dimensions from Megatron-LM research.*

### Frontier engineering: DeepSeek-V3 DualPipe and communication overlap

When running 64-way Expert Parallelism across nodes, routing tokens via AllToAll collectives threatens to swamp scale-out InfiniBand links. To eliminate this bottleneck, DeepSeek-V3 introduced **DualPipe**.

DualPipe executes two independent pipeline schedules simultaneously in opposite directions (forward and backward). By breaking layers into fine-grained chunks, the engine dispatches inter-node AllToAll token routing in the background while the GPU's Tensor Cores compute dense attention and feed-forward GEMMs for the other chunk. By perfectly overlapping communication with computation, DeepSeek hides over 95% of scale-out network latency, sustaining near-peak MFU even over constrained H800 fabrics.

---

## 4. Pillar III — Storage: Foundation Model Datasets, Checkpointing, and High-Throughput I/O

### The storage crisis in modern foundation model pre-training

While compute and networking dominate AI research discussions, foundation model training clusters spend thousands of hours paralyzed by storage I/O. In modern pre-training, storage systems face two severe challenges:

1. **Continuous Dataset Ingestion:** Training a frontier model like LLaMA 3 requires streaming 15+ trillion tokens of unstructured text, code, and multimodal tokens. Across 16,384 GPUs, the dataloader subsystem must sustain continuous reads of 2 to 4 Terabytes per second without dropping frames or stalling PyTorch worker threads.
2. **The Checkpointing Crisis:** Foundation model states must be snapshotted to persistent storage periodically to protect against hardware crashes. Consider the exact memory footprint of a 405-billion parameter model during training:

$$S_{\text{state}} = S_{\text{weights}} + S_{\text{gradients}} + S_{\text{AdamW}} = 2M + 2M + 12M = 16M \text{ bytes}$$

For $M = 405 \times 10^9$ parameters:
- **BF16 Model Weights:** $405 \times 10^9 \times 2 \text{ bytes} = 810 \text{ GB}$
- **BF16 Gradients:** $405 \times 10^9 \times 2 \text{ bytes} = 810 \text{ GB}$
- **FP32 AdamW Optimizer State:** Master weights (4 bytes) + Momentum (4 bytes) + Variance (4 bytes) = $405 \times 10^9 \times 12 \text{ bytes} = 4,860 \text{ GB} = 4.86 \text{ TB}$

Summing these yields $6,480 \text{ GB} \approx 6.48 \text{ TB}$. Adding activation buffers, random number generator (RNG) states, and optimizer step metadata expands the total checkpoint snapshot to **~7.5 Terabytes**.

If a cluster writes this 7.5 TB snapshot synchronously over a legacy 25 GB/s shared storage fabric, **the entire 16,384-GPU cluster must freeze for over 5 minutes ($307 \text{ seconds}$)**. In a multi-week training run, these checkpoint pauses waste millions of dollars in idle GPU cycles.

### The traditional I/O bottleneck: Host CPU page cache and bounce buffers

The root cause of storage stalls in traditional GPU clusters is the multi-hop legacy I/O architecture:
1. **NVMe Read/Write:** Data travels from the NVMe SSD over PCIe into the host CPU root complex.
2. **Kernel Page Cache Copy:** The Linux storage stack copies data into the kernel page cache in host DDR5 system memory.
3. **User Space Bounce Buffer:** PyTorch or the checkpoint manager copies data from kernel space into user space host RAM.
4. **Host-to-Device Transfer:** The host CPU initiates a secondary DMA transfer over PCIe Gen 5 to stream data into GPU High Bandwidth Memory (HBM).

This legacy path forces every byte to cross the PCIe bus twice and undergo two CPU memory copies (`memcpy`). A modern dual-socket host CPU saturates its memory bus at 50 to 100 GB/s while running at 100% CPU core utilization, turning storage I/O into a crippling bottleneck.

### GPUDirect Storage (GDS) and direct DMA pipelines

To eliminate this bottleneck, modern AI supercomputers implement **GPUDirect Storage (GDS)**:

<svg viewBox="0 0 560 420" role="img" aria-label="Comparison of traditional CPU-buffered storage I/O with GPUDirect Storage direct DMA architecture." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="gds-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="gds-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="gds-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="gds-arr-suc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
<marker id="gds-arr-warn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Storage Architecture: CPU Bounce vs GPUDirect Storage</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Checkpointing and Dataset Streaming Pipelines</text>
<rect x="16" y="48" width="256" height="286" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="68" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Traditional I/O Path</text>
<text x="144" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">CPU Page Cache &amp; Bounce Buffers</text>
<rect x="32" y="252" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="144" y="274" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">NVMe Storage (SSD)</text>
<rect x="32" y="150" width="224" height="54" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="144" y="170" text-anchor="middle" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">Host CPU &amp; DDR5 RAM</text>
<text x="144" y="188" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Kernel Page Cache (50-100 GB/s)</text>
<rect x="32" y="96" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="144" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU HBM3 (3.35 TB/s)</text>
<line x1="144" y1="250" x2="144" y2="206" marker-end="url(#gds-arr-warn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="150" y="232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">I/O Hop 1</text>
<line x1="144" y1="148" x2="144" y2="134" marker-end="url(#gds-arr-warn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="150" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">I/O Hop 2</text>
<rect x="24" y="302" width="240" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="144.0" y="316.3" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Bottleneck: CPU &amp; RAM Bounce</text>
<rect x="288" y="48" width="256" height="286" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="68" text-anchor="middle" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">GPUDirect Storage (GDS)</text>
<text x="416" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Direct DMA (cuFile API)</text>
<rect x="304" y="252" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="416" y="274" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">NVMe Storage (SSD)</text>
<rect x="304" y="162" width="224" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="180" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:700">PCIe Gen 5 Switch / DMA</text>
<text x="416" y="196" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Direct Hardware Path (64+ GB/s)</text>
<rect x="304" y="96" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU HBM3 (3.35 TB/s)</text>
<line x1="416" y1="250" x2="416" y2="206" marker-end="url(#gds-arr-suc)" style="stroke:var(--c-success);stroke-width:2"/>
<line x1="416" y1="160" x2="416" y2="134" marker-end="url(#gds-arr-suc)" style="stroke:var(--c-success);stroke-width:2"/>
<text x="424" y="232" text-anchor="start" style="fill:var(--c-success);font-size:10px;font-weight:600">Direct DMA</text>
<rect x="296" y="302" width="240" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="416.0" y="316.3" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Zero CPU Overhead · Direct DMA</text>
<rect x="16" y="344" width="528" height="64" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="364" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Checkpointing Challenge: 405B model saves ~7.5 TB state periodically.</text>
<text x="28" y="382" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">With GDS, storage streams directly into GPU HBM bypassing system RAM,</text>
<text x="28" y="398" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">eliminating CPU bounce latency and preventing training stalls.</text>
</svg>

GDS relies on the `cuFile` API and the `nvidia-fs.ko` kernel driver. It establishes a direct DMA (Direct Memory Access) channel between local NVMe drives (or remote NVMe-oF over InfiniBand) and GPU High Bandwidth Memory through the PCIe switch fabric, completely bypassing the host CPU, user space bounce buffers, and the Linux kernel page cache.

By eliminating CPU memory copies:
- Storage throughput jumps to line-rate PCIe Gen 5 speeds (**64+ GB/s per NIC/SSD controller**).
- End-to-end I/O latency drops by **80%**.
- Host CPU utilization plummets from **100% to near 0%**, leaving host cores free for token preprocessing and orchestration.

<details>
<summary>Quiz: Why does GPUDirect Storage (GDS) reduce CPU utilization to near zero during checkpointing?</summary>

Traditional I/O requires the host CPU to execute memory copy instructions (`memcpy`) moving gigabytes of data between kernel page cache buffers and user space memory. GDS programs the PCIe DMA engines to stream data directly between NVMe storage controllers and GPU HBM, completely bypassing host CPU cores and system DDR5 RAM.
</details>

### Exabyte-scale distributed filesystems: The Meta Tectonic case study

In a production cluster spanning thousands of nodes, local NVMe drives cannot store global checkpoints. Instead, clusters connect to exabyte-scale distributed filesystems.

In Meta's official LLaMA 3 infrastructure report, the 16,384-GPU cluster is backed by **Tectonic**, Meta's distributed filesystem:
- **Scale:** 7,500 dedicated SSD-backed storage servers providing 240 Petabytes of storage capacity.
- **Sustained Read Throughput:** Delivers a constant **2 TB/s** of read bandwidth to feed dataloader worker processes across the 2,048 compute nodes.
- **Burst Checkpoint Throughput:** Sustains **7 TB/s of peak burst write throughput**, enabling the cluster to snapshot a 7.5 TB checkpoint across all ranks in approximately **one second** using asynchronous copy-on-write buffers.
- **Reliability:** Uses Reed-Solomon erasure coding ($9+3$ chunks) to survive multi-drive and rack-level storage server failures without corrupting model state.

### High-throughput dataloading: Token binary indexing and zero-copy mmap

Feeding tokens to thousands of GPUs requires specialized dataloader engineering:
- **Why standard PyTorch DataLoader fails:** Loading raw JSONL or text files forces Python workers to parse strings, tokenize text, and allocate dynamic Python objects, causing the Python Global Interpreter Lock (GIL) to stall worker threads.
- **Contiguous Binary Layout (.bin / .idx):** Production frameworks (such as Megatron-LM and Lit-GPT) pre-tokenize datasets into monolithic binary arrays of `uint16` or `int32` token IDs contiguous on disk (`.bin`), accompanied by an index file (`.idx`) mapping document byte offsets.
- **Zero-Copy Memory Mapping (`mmap`):** Worker processes use POSIX `mmap()` to map the binary dataset directly into the process virtual address space. The operating system transparently pages in token blocks on demand without user-space buffer allocations.
- **NUMA-Aware Prefetching:** Compute nodes feature two CPU sockets, each tied to a dedicated NUMA node and 4 GPUs (in the node topology of Figure 1.1, CPU 0 feeds GPUs 0–3 and CPU 1 feeds GPUs 4–7). Workers are pinned to the specific CPU socket physically wired to the GPU's PCIe root complex, utilizing asynchronous CUDA streams (`pin_memory=True`) to stream tokens into HBM without crossing NUMA interconnects.

### Cluster MTBF, fault tolerance, and the Young-Daly checkpoint formula

In a cluster of 16,384 GPUs (2,048 nodes), hardware failures are not rare anomalies—they are continuous operating conditions:
- If a single DGX node has a Mean Time Between Failures (MTBF) of 3 years (~26,000 hours), the aggregate cluster MTBF is:

$$M_{\text{cluster}} = \frac{26,000 \text{ hours}}{2,048 \text{ nodes}} \approx 12.7 \text{ hours}$$

In practice, factoring in silent data corruption (SDC), HBM3 ECC errors, optical transceiver drops, and network timeouts, a 16k GPU cluster experiences a fatal interruption every **2 to 3 hours**.

When a node crashes, all GPUs must restart from the last saved checkpoint. This introduces a fundamental mathematical trade-off:
- If you checkpoint too frequently, you waste massive time writing data ($\delta$).
- If you checkpoint too infrequently, you waste hours of computation re-running lost steps after a crash.

The mathematically optimal checkpoint interval is governed by the classical **Young-Daly Formula**:

$$\tau_{\text{opt}} = \sqrt{2 \cdot \delta \cdot M}$$

Where:
- $\delta$ is the time required to save a checkpoint snapshot (in seconds).
- $M$ is the cluster Mean Time Between Failures (MTBF, in seconds).
- $\tau_{\text{opt}}$ is the optimal time between consecutive checkpoints.

Let us evaluate this trade-off for LLaMA-3 405B on 16k GPUs ($M = 3 \text{ hours} = 10,800 \text{ seconds}$):
- **Scenario A: Legacy Storage Fabric (25 GB/s cluster aggregate write speed):**
  $$\delta = \frac{7.5 \times 1024 \text{ GB}}{25 \text{ GB/s}} \approx 307.2 \text{ s} \approx 5.12 \text{ minutes}$$
  $$\tau_{\text{opt}} = \sqrt{2 \cdot 307.2 \cdot 10,800} \approx \sqrt{6,635,520} \approx 2,576 \text{ s} \approx 42.9 \text{ minutes}$$
  Writing checkpoints every 43 minutes causes $\frac{5.12}{42.9} \approx 11.9\%$ direct checkpoint overhead, plus $\frac{\tau}{2M} \approx 11.9\%$ expected recomputation loss—**wasting nearly 24% of the entire supercomputer's compute capacity!**
- **Scenario B: GPUDirect Storage & Tectonic Burst (250 GB/s write fabric):**
  $$\delta = \frac{7.5 \times 1024 \text{ GB}}{250 \text{ GB/s}} \approx 30.72 \text{ s} \approx 0.51 \text{ minutes}$$
  $$\tau_{\text{opt}} = \sqrt{2 \cdot 30.72 \cdot 10,800} \approx \sqrt{663,552} \approx 814.6 \text{ s} \approx 13.6 \text{ minutes}$$
  Checkpoint overhead becomes $\frac{0.51}{13.6} \approx 3.8\%$ and the expected recomputation loss $\frac{\tau}{2M} \approx 3.8\%$, so total lost compute drops **from 24% to about 7.5%**, recovering millions of dollars in productive training time. A 10× faster write cuts the loss by $\sqrt{10} \approx 3.2\times$, because at $\tau_{\text{opt}}$ both loss terms scale with $\sqrt{\delta}$.

<details>
<summary>Quiz: What is the optimal checkpoint interval for a cluster with MTBF = 4 hours and checkpoint duration = 2 minutes?</summary>

Applying the Young-Daly equation:
$\tau_{\text{opt}} = \sqrt{2 \cdot \delta \cdot M} = \sqrt{2 \cdot 2 \text{ min} \cdot 240 \text{ min}} = \sqrt{960} \approx 30.98 \text{ minutes}$.
Checkpointing roughly every 31 minutes minimizes total lost training time.
</details>

---

## 5. Synthesis: Training vs inference and operational trade-offs

### The two engineering lenses: Training vs inference

Hardware requirements diverge sharply between foundation model pre-training and real-time inference:

| Architectural Dimension | Training (Pre-training / SFT) | Inference (Prefill & Decode) |
| :--- | :--- | :--- |
| **Primary Memory Footprint** | Weights ($2B$) + Gradients ($2B$) + AdamW ($12-16B$) + Activations | Model Weights ($4-8$ bit quantized) + Dynamically allocated KV Cache |
| **Compute / Memory Regime** | Heavily **Compute-Bound** (GEMM with thousands of tokens per batch) | Prefill is **Compute-Bound**; Decode is strictly **Memory-Bandwidth-Bound** ($B=1$, GEMV) |
| **Dominant Network Primitive** | High-throughput **AllReduce** & **ReduceScatter** over InfiniBand | Ultra-low latency **AllGather** & **P2P Tensor Parallelism** over NVLink |
| **Key Performance Metrics** | Model FLOPs Utilization (MFU), Step Time (ms), Gradient Stability | Time to First Token (TTFT), Inter-Token Latency (TPOT), Concurrent Requests |
| **Critical Failure Modes** | Loss spikes, silent data corruption (SDC), pipeline bubbles | KV Cache Out-of-Memory (OOM), memory fragmentation, tail latency spikes |

### End-to-end cluster walkthrough: 70B and 405B models on 16k GPUs

To consolidate our three pillars, consider pre-training LLaMA-3 70B and 405B on an NVIDIA H100 SuperPod (Meta's actual 405B cluster used 400 Gb/s RoCE per GPU rather than InfiniBand, so the SHARP step below belongs to the SuperPod scenario):

1. **LLaMA-3 70B on 8x H100 DGX Node:**
   - 70 billion parameters in BF16 consume 140 GB for weights.
   - 8 H100 SXM5 GPUs provide $8 \times 80 = 640 \text{ GB}$ of HBM3.
   - Tensor Parallelism is set to $Y = 8$, fitting weights and activations entirely within the 3.6 TB/s NVLink crossbar domain.
   - Every Transformer layer performs two AllReduces over NVLink (latency: $<15 \, \mu\text{s}$ per collective).
   - Sustained MFU exceeds 48-52%.
2. **LLaMA-3 405B on 16,384 H100 GPUs (2,048 Nodes):**
   - 4D Parallelism: 8-way intra-node TP $\times$ 16-way PP $\times$ 128-way DP (ZeRO-1).
   - TP is strictly confined to individual 8-GPU chassis over NVLink.
   - 16 pipeline stages exchange microbatch activations over scale-out InfiniBand (P2P message size: $\sim 16 \text{ MB}$).
   - 128-way ZeRO-1 data-parallel ranks synchronize optimizer gradients using Mellanox SHARP hardware-accelerated AllReduce across leaf switches.
   - Storage streams 2 TB/s of continuous token data using zero-copy `mmap`, while GPUDirect Storage snapshots 7.5 TB model checkpoints every ~15 minutes with ~30-second burst write pauses.

---

## The whole story in six lines

- A modern GPU supercomputer is governed by three interconnected physical pillars: Streaming Multiprocessor compute, multi-tiered network fabrics, and high-throughput storage I/O.
- The roofline model dictates a critical arithmetic intensity of ~295 FLOP/byte on H100, requiring a batch size of ~300 tokens per SM to escape memory-bound latency.
- Intra-node NVLink crossbars sustain 450 GB/s unidirectional bandwidth per GPU (3.6 TB/s bisection), keeping 8-way Tensor Parallelism compute-bound.
- Scale-out InfiniBand fat-trees guarantee full bisection bandwidth across thousands of GPUs, dropping link speeds to 50 GB/s per chip (400 GB/s per node).
- Mellanox SHARP performs floating-point accumulation directly inside switch crossbars, cutting cluster-wide AllReduce traffic and latency in half.
- GPUDirect Storage eliminates host CPU bounce buffers to snapshot 7.5 TB checkpoints in seconds, matching the Young-Daly optimal interval to protect cluster MFU.

---

## Glossary

- **Streaming Multiprocessor (SM)** — The core multi-threaded execution cell of an NVIDIA GPU containing Tensor Cores, Warp Schedulers, CUDA cores, and on-chip SRAM.
- **Warp** — The basic scheduling unit in NVIDIA architectures, consisting of 32 parallel threads executed synchronously by an SM warp scheduler.
- **Warp Divergence** — Performance loss occurring when threads within a single warp branch into divergent execution paths, forcing inactive cores to stall.
- **NVLink** — NVIDIA's proprietary high-bandwidth interconnect providing 450 GB/s unidirectional (900 GB/s bidirectional) memory transfer per H100 GPU.
- **NVSwitch** — On-node crossbar ASIC enabling non-blocking, all-to-all communication across all 8 GPUs in an HGX chassis.
- **InfiniBand Fat-Tree** — A hierarchical multi-tier network topology ensuring full bisection bandwidth without oversubscription across cluster nodes.
- **Mellanox SHARP** — Scalable Hierarchical Aggregation and Reduction Protocol; switch hardware that executes floating-point AllReduces in-flight.
- **Model FLOPs Utilization (MFU)** — The ratio of theoretically required model FLOPs to the actual sustained hardware floating-point throughput.
- **Tensor Parallelism (TP)** — Intra-layer model partitioning that splits individual weight matrices across GPUs, bounded strictly by NVLink bandwidth.
- **Pipeline Parallelism (PP)** — Inter-layer model partitioning that assigns consecutive Transformer layers to distinct GPUs, incurring bubble overhead.
- **GPUDirect Storage (GDS)** — Direct DMA technology connecting NVMe storage directly to GPU High Bandwidth Memory, bypassing host CPU page caches.
- **Young-Daly Formula** — Mathematical formulation ($\tau = \sqrt{2 \delta M}$) determining the optimal checkpoint interval based on write duration and cluster MTBF.

---

## Going deeper

- Jacob Austin, Swapnil Patil, Adam Paszke, Reiner Pope (2025). [How to Scale Your Model: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). JAX-ML Scaling Book, Chapter 12.
- SemiAnalysis (2025). [NVIDIA Tensor Core Evolution from Volta to Blackwell](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/). Microarchitectural retrospective.
- DeepSeek-AI (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv:2412.19437. Detailed 64-way Expert Parallelism and DualPipe implementation on 2,048 H800 GPUs.
- Meta AI (2024). [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). arXiv:2407.21783. 16,384 GPU 4D parallelism, Tectonic storage, and training infrastructure.
- NVIDIA Corporation. [DGX SuperPOD Reference Architecture (DGX H100): Key Components](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-components.html) and [Network Fabrics](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html). Scalable units, the four networks, 127 nodes with a UFM slot, and the separate storage fabric; diagrams in Figure 1.1 were inspired by these.
- NVIDIA Corporation. [Introduction to NVIDIA DGX H100/H200 Systems](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). Node hardware: CPUs, memory, NICs and NVMe drives; its DGX H100/H200 system topology diagram inspired the node level of Figure 1.1.
- NVIDIA Corporation. [Fabric Manager for NVIDIA NVSwitch Systems](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). How the 18 NVLinks of each H100 split as 5 + 4 + 4 + 5 over four NVSwitches, and why the NVLink fabric does not come up without Fabric Manager.
- NVIDIA Corporation. [DGX SuperPOD Architecture (DGX H100)](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html). Scaling table: two tiers up to 8 SUs, a core tier from 16 SUs; 512 leaf, 512 spine and 256 core switches for 2,048 DGX.
- NVIDIA Corporation. [DGX SuperPOD Data Center Design (H100): Planning](https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/planning.html). Why a full SuperPOD has 127 DGX (a pair of UFM appliances takes one slot), 31 to 127 DGX per build, and 4 DGX (40.8 kW) per rack.
- NVIDIA Developer Blog (2022). [Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/). PXN: moving cross-rail traffic onto NVLink first so it skips the spine.
- Deepak Narayanan et al. (2021). [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473). arXiv:2104.04473. Why tensor parallelism stays on NVLink inside a server while pipeline and data parallelism run across servers.
- Charles Clos (1953). [A Study of Non-Blocking Switching Networks](https://doi.org/10.1002/j.1538-7305.1953.tb01433.x). Bell System Technical Journal 32(2), 406–424. The multi-stage switching idea behind today's leaf–spine networks.
- Charles E. Leiserson (1985). [Fat-trees: Universal Networks for Hardware-Efficient Supercomputing](https://doi.org/10.1109/TC.1985.6312192). IEEE Transactions on Computers C-34(10), 892–901. Where the fat-tree name comes from: links that thicken toward the top.
- NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Official microarchitecture guide.
- On this blog: [A Prompt's Journey (8): The Output Layer (LM Head) — Megatron-LM and the Memory Wall](post.html?slug=inside-the-output-layer).
- On this blog: [A Prompt's Journey (9): Inside the Context Window — From the 2K Wall to FlashAttention](post.html?slug=inside-the-context-window).
