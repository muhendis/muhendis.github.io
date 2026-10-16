On paper, an NVIDIA DGX SuperPOD of 127 DGX H100 nodes looks unstoppable: 1,016 H100 GPUs, 1.0 ExaFLOP/s of dense BF16 compute, 81 TB of HBM3, a 51.2 TB/s spine network and 1.39 MW of power. Meta trained Llama 3 405B on 16,384 of the same H100s, and each GPU sustained about 400 of its 990 TFLOP/s: 41% Model FLOPs Utilization (MFU), and 38% once the context grew to 128K tokens. Over the 54 days the report covers, the run was also interrupted 419 times without warning, about once every three hours.

The missing 59% is not a bug that a better kernel will fix. It is the time data spends waiting. Tensor Cores wait for operands that are ~480 cycles away in HBM. GPUs wait for gradients that leave the chassis on a road nine times narrower than NVLink. The whole cluster waits while a 7.5 TB checkpoint drains to the file system. Each part of this story has its own document: a whitepaper for the SM, a reference architecture for the SuperPOD, a paper for 4D parallelism, an operations report for failures. Read separately, each looks fine. What gets lost is how they connect, and the cluster runs only as fast as its slowest link.

This article puts the connections on one map. Figure 1.1 is a single poster in five parts. It zooms out from one Streaming Multiprocessor to a 16,384-GPU cluster, follows one Llama 3 405B training step through every road it uses, prices the three walls (memory, network and storage) with computed curves, pushes the same 140 GB through every link, and ends on Meta's measured numbers. The text below is the poster's legend: sections 2 to 4 give the formulas and derivations behind each card, pillar by pillar, while section 5 synthesizes the hardware hierarchy and bandwidth gaps. It is the first article of the AI Infrastructure series, and its job is to teach the structure: what the machine is made of, how its parts are wired, and which road each byte takes. The second article, [AI Infrastructure (2): Diagnosing an AI Supercomputer — Why 1,024 H100s Stall at 38% MFU](post.html?slug=from-gpus-to-ai-supercomputers), shows how this structure behaves under training load, 4D parallelism, failures, and Model FLOPs Utilization (MFU) diagnostics.

**In this article**

- [1. How to read the map: three pillars, five parts](#1-how-to-read-the-map-three-pillars-five-parts)
- [2. Pillar I — Compute: inside the SM and the H100 chip](#2-pillar-i-compute-inside-the-sm-and-the-h100-chip)
- [3. Pillar II — Network: from the NVLink chassis to the SuperPOD fabric](#3-pillar-ii-network-from-the-nvlink-chassis-to-the-superpod-fabric)
- [4. Pillar III — Storage: Direct DMA and the high-throughput datapath](#4-pillar-iii-storage-direct-dma-and-the-high-throughput-datapath)
- [5. Synthesis: Hardware hierarchy and latency gaps](#5-synthesis-hardware-hierarchy-and-latency-gaps)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. How to read the map: three pillars, five parts

A modern AI supercomputer is not merely a cluster of standalone accelerators; it is an integrated **co-design** spanning silicon compute, in-chassis interconnects, cluster fabric topology, and high-throughput storage paths. This massive machine is founded upon three physical hardware pillars:

1. **Compute Pillar:** The on-chip hierarchy — Streaming Multiprocessors (SMs), Warp Schedulers, Tensor Cores, and 3.35 TB/s HBM3 memory stacks.
2. **Fabric / Interconnect Pillar:** In-chassis scale-up via 450 GB/s (unidirectional) NVLink 4 and the NVSwitch crossbar matrix; inter-chassis scale-out via a 50 GB/s (400 Gb/s) non-blocking rail-optimized InfiniBand fat-tree fabric.
3. **Storage Pillar:** Direct DMA bypassing the host CPU and kernel page cache via GPUDirect Storage (GDS) across a dedicated storage fabric.

These three pillars unite across a physical scale ladder extending from microscopic threads to multi-thousand-GPU datacenters:

| Level | Size | Hardware Component and Architectural Role |
| :--- | :--- | :--- |
| **Thread** | Smallest logical execution instance | Scalar CUDA execution lane with private register file |
| **Warp** | 32 threads | Fundamental scheduling unit executed synchronously (SIMT) by the SM warp scheduler |
| **SM** | Up to 64 warps, 2,048 threads | 4 subpartitions (SMSP), Tensor Cores, 256 KiB register file, and 228 KiB L1/Shared Memory |
| **H100 SXM5** | 132 SMs, 80 GB HBM3 | 990 TFLOP/s dense BF16 compute, 50 MB L2 cache, and 3.35 TB/s HBM3 bandwidth |
| **DGX H100** | 8 GPUs, 4 NVSwitches | 3.6 TB/s all-to-all NVLink crossbar fabric in-chassis and dual-socket host CPU |
| **Scalable Unit (SU)** | 32 DGX, 256 GPUs | Rail-aligned connectivity mapped across 8 independent leaf switch planes |
| **SuperPOD** | 127 DGX, 1,016 GPUs | Two-tier non-blocking rail-optimized InfiniBand fat-tree fabric |
| **Multi-Pod Cluster** | 16,384 H100s | Datacenter-scale 3-tier spine fabric with dedicated storage network |

A note on the numbers: four SUs of 32 servers are nominally 128 × 8 = 1,024 GPUs, but in NVIDIA's reference design one server slot goes to the UFM appliances, so a built SuperPOD has 127 DGX nodes and 1,016 GPUs. Both numbers appear in the literature, describing different aspects.

The poster below draws all three pillars on one reference system, NVIDIA H100 SXM5 → DGX H100 → DGX SuperPOD with an InfiniBand fat-tree, and keeps every other accelerator or cluster in parentheses or in its own part:

<svg viewBox="0 0 840 19489" role="img" aria-label="A five-part poster of AI infrastructure. Part A zooms out in four levels. Level 1, the Streaming Multiprocessor, one of 132 in an H100 SXM5, numbered like the SM list in the text: four identical subpartitions, each with (1) a fourth-generation Tensor Core doing 1,024 BF16 or 2,048 FP8 FLOPs per cycle, (2) a warp scheduler that issues one instruction per cycle from up to 16 warps, (3) 32 FP32 CUDA cores plus 16 INT32 and 16 FP64 units, and (4) a 64 KiB register file of 16,384 registers; the four share 256 KiB of SMEM and L1, fed from L2 and HBM by the TMA copy engine. 528 Tensor Cores times 1,024 FLOPs times 1.83 GHz give about 989 TFLOP/s. Measured load latencies on Hopper are about 29 cycles for SMEM, 41 for L1, 263 for L2 and 479 for HBM, so the scheduler hides the wait by switching every cycle to a ready warp at no cost; a timeline shows four warps that each compute for 2 units and wait 6 keeping the scheduler busy, while two warps leave it idle half the time, since warps needed are about 1 plus wait over compute. More registers per thread mean fewer resident warps: 32 registers allow 64 warps, 64 allow 32, 128 allow 16 and 255 allow 8; fast Hopper kernels such as FlashAttention-3 instead run few warps and hide latency with TMA copies and wgmma. In the kitchen model the SM is a line cook with order tickets, registers and SMEM are the counter, the Tensor Core is the knife and stove, and HBM is the shared pantry. Level 2, the H100 SXM5 chip, after the text&#x27;s memory hierarchy and roofline subsection: an 814 square millimetre GH100 die with 80 billion transistors on TSMC 4N, built as 8 GPCs of 9 TPCs with 2 SMs each, 144 slots of which 132 are enabled for yield, an NVLink 4 port with 18 links and a PCIe Gen5 x16 port, a 50 MB L2 in two partitions, and 5 active HBM3 stacks of 16 GB in 6 sites, each behind two 512-bit memory controllers; the full chip has 144 SMs, 60 MB of L2 and 6 stacks. 5,120 bits at about 5.2 Gb/s per pin give 3.35 TB/s. The memory pyramid runs from 256 KiB of registers and 256 KiB of SMEM or L1 per SM, about 33 MB each chip-wide, to 50 MB of L2 at about 263 cycles and 5.5 TB/s and 80 GB of HBM3 at about 479 cycles and 3.35 TB/s. A log-log roofline, min of 990 and 3.35 times alpha, has its ridge at alpha 295.5 FLOPs per byte: a BF16 addition at alpha 0.17 and decode at batch 1 with alpha about 1 are memory bound, batch 128 reaches 420 TFLOP/s and batch 4,096 sits on the 990 roof. In the kitchen model the GPU is the kitchen of 132 cooks, L2 the shared shelf, HBM the pantry with 5 doors, and NVLink and PCIe the service door. 8 GPUs make one DGX H100 node: 4 NVSwitches, 900 GB/s NVLink per GPU, 3.6 TB/s bisection, 2 Xeon CPUs with 2 TB RAM on PCIe Gen5 at 64 GB/s, one 400 Gb/s compute NIC per GPU, 2 dual-port NICs for storage and management, and 8 NVMe drives. The node card, after the text&#x27;s intra-node networking subsection, derives 450 GB/s as 18 links of 25 GB/s and the 3.6 TB/s bisection as 4 GPUs times 450 GB/s times 2; a link matrix shows every GPU sending 5, 4, 4 and 5 NVLinks to NVSwitches 0 to 3, which receive 40, 32, 32 and 40 of the 144 links, so the fabric is non-blocking, each 64-port NVSwitch carries 3.2 TB/s both ways and can sum data itself with NVLink SHARP. Per GPU, NVLink gives 450 GB/s, PCIe 64 GB/s and the NIC 50 GB/s: TP AllReduce uses NVLink twice per layer, PP activations and DP gradients use the NIC to leaf i, data and checkpoints use the CX-7 storage fabric, and the data loader uses PCIe from the same socket. By the text&#x27;s TP rule, Y is at most 28,672 over 2,200, about 13, on NVLink but only 1.45 on the network, so TP 8 stays in the chassis. In the restaurant model the node has eight kitchens, NVLink is the inner corridor, the CPUs are two managers and the NICs are courier doors. 32 nodes make a scalable unit and 4 make a DGX SuperPOD of 127 nodes, 1,016 GPUs: 32 leaf and 16 spine switches in a rail-optimised fat-tree where NIC i of every node goes to leaf i, a separate storage fabric with at least 40 GB/s per node, and in-band and out-of-band Ethernet management. The cluster card, after the text&#x27;s scale-out subsection, compares a top-of-rack layout, where GPU 3 of node A reaches GPU 3 of node B through leaf, spine and leaf in 3 hops, with NVIDIA&#x27;s rail layout, where every node&#x27;s NIC i goes to leaf i and the same message takes 1 hop; because TP 8 stays inside the chassis, DP and PP traffic runs between same-numbered GPUs. A scale ladder multiplies BF16 compute from 7.5 TFLOP/s per SM to 990 per H100, 7.9 PFLOP/s per DGX, 253 per SU, 1.0 EFLOP/s per SuperPOD and 16.2 EFLOP/s for 64 SUs, with HBM growing from 80 GB to 1.31 PB, while the road out narrows per GPU from 3.35 TB/s to 450 and 50 GB/s. In the restaurant-chain model leaves are local exchanges, spines trunk lines, and storage and management the depot and head office. Part B follows one training step of Llama 3 405B on 16,384 GPUs. A 4D parallelism card explains the split. The problem: training Llama 3 405B on one H100 would take about 3,000 years, and its 6,480 GB training state is 81 times the GPU&#x27;s 80 GB. Four steps each solve one problem: TP 8 splits every layer across a node&#x27;s 8 GPUs, 6,480 to 810 GB per GPU and 3,000 to 380 years; PP 16 spreads the 126 layers over 16 nodes into a 128-GPU chain, 810 to 50.6 GB and 24 years; DP 128 copies the chain for 16,384 GPUs and about 67 days; FSDP splits the optimizer state across the copies, 50.6 to 12.9 GB. A summary chart shows each step against the 80 GB line. TP 8 is the GPUs in a node, PP 16 the nodes in a chain and DP 128 the copies, so each GPU&#x27;s address is [TP rank, PP rank, DP rank], and GPUs with the same TP rank share a rail. A table says what each of the five dimensions splits, what one GPU does, where, how often it talks and why it is needed, under the rule that the more often a dimension talks the further inside it sits; EP for MoE and CP for long context are the fourth dimension, depending on the source. Four pictures contrast a small example with Llama 3&#x27;s version. Eight steps follow the data: 1 read 67 MB of tokens per step; 2 forward pass, about 2.1 s; 3 tensor-parallel AllReduce of 268 MB over NVLink about 500 times at 1 ms; 4 send a 34 MB activation slice per micro-batch over the rail; 5 backward pass, about 4.1 s; 6 FSDP gradient sync of 12.7 plus 6.3 GB, about 0.38 s, overlapped; 7 AdamW updates each GPU&#x27;s 24.7 million parameters in HBM; 8 every ~14 minutes write a 7.5 TB checkpoint, ~31 s at 250 GB/s. A budget chart puts a 6.2 s step of compute against 2.5 s at peak, 0.53 s of NVLink, 0.38 s of network sync, 0.23 s of amortised checkpoint and 0.02 s of pipeline sends. Part C shows three walls with computed curves. Memory wall: for a Llama 3 70B MLP on H100, achieved speed rises with batch size from 3.3 TFLOP/s at batch 1 to 420 at batch 128 and reaches the 990 ceiling near batch 310; fixes are bigger batches, kernel fusion and low-bit weights for decode. Network wall: bandwidth per GPU falls from 3.35 TB/s in HBM to 450 GB/s on NVLink and 50 GB/s off the node; tensor parallelism fits 13 GPUs on NVLink so 8, but only 1.45 over InfiniBand; a 16-stage pipeline idles 48 percent with 16 microbatches and 10.5 percent with 128; fixes are TP in the node, SHARP and overlapping communication. I/O wall: Young-Daly loss curves against checkpoint interval, with a minimum of 24 percent at 42.9 minutes for 25 GB/s and 7.5 percent at 13.6 minutes for 250 GB/s; fixes are GPUDirect Storage, asynchronous writes and the square-root interval. Part D times the same 140 GB load: 42 milliseconds from HBM, 0.31 s over NVLink, 2.2 s over PCIe, 2.8 s over InfiniBand per GPU, 5.6 s for legacy storage shared by the whole cluster; decode on 8 H100s is capped near 190 tokens per second. Part E lists Meta&#x27;s Llama 3 405B run: 16,384 H100s at 400 TFLOP/s and 41 percent MFU, a RoCE network with 3,072-GPU pods, Tectonic storage of 240 PB with 2 TB/s sustained and 7 TB/s peak, and 466 interruptions in 54 days, about one every 3 hours. The slowest link sets the pace of the whole cluster. After the cluster level, a building-blocks card shows a DGX&#x27;s four networks: 8 × 400 Gb/s InfiniBand to the compute fabric of 48 QM9700 switches (400 GB/s), 2 × 400 Gb/s to the storage fabric of 16 QM9700 (100 GB/s, target at least 40 GB/s), 2 × 100 GbE to 8 SN4600C in-band switches and 1 GbE from the BMC to 8 SN2201 out-of-band switches, each kept apart so its traffic doesn&#x27;t slow the AllReduce. A parts list gives each count and its origin: 127 DGX as 4 SUs × 32 slots minus one for the UFM pair, 32 leaves plus 16 spines, 4 UFM appliances, 5 management servers, 38 racks and 108 PDUs. Power adds up to 1.39 MW on average, 1.295 MW of it, 92.9%, from 127 DGX at 10.2 kW. Storage carries checkpoints, not tokens: Llama 3 70B on 1,016 GPUs reads only about 3.8 MB/s of tokens against NVIDIA&#x27;s 60 GB/s NLP target, while a 1.12 TB checkpoint stops training for 37, 14 or 4.5 seconds at 30, 80 or 250 GB/s. Six software steps carry a job: Base Command Platform submits it, Slurm allocates nodes, Base Command Manager installs one image, NGC and AI Enterprise start one stack, Magnum IO moves the data with NCCL, SHARP and GPUDirect Storage, and UFM finds failures; in Llama 3&#x27;s run only 3 of 419 interruptions needed manual intervention. In the restaurant model storage is the supply line, in-band the service corridor, out-of-band the fuse box and the schedulers the shift manager. The node level redraws NVIDIA&#x27;s DGX H100 system topology: CPU 0 and CPU 1 joined by UPI; CPU 0 feeds GPUs 0 to 3 and NICs 0 to 3, CPU 1 feeds GPUs 4 to 7 and NICs 4 to 7, each GPU paired with its own network card; two upper PCIe switches carry the dual-port storage cards and four NVMe drives each; CPU 0 has two M.2 OS drives and CPU 1 a 100 GbE card; every GPU sends 18 NVLinks, split 5, 4, 4 and 5, to four NVSwitches, which a lower PCIe switch connects to CPU 0 for management. Six numbered circles on the node diagram match six questions below it: who feeds whom (two NUMA halves), how a GPU reaches the network (its own NIC with GPUDirect RDMA), how GPUs talk (18 NVLinks to four NVSwitches), where storage traffic goes (upper PCIe switches), what joins the halves (CPU communication, UPI) and who manages the NVSwitches (Fabric Manager over PCIe from CPU 0). The cluster level opens with the hierarchy H100 SXM5, DGX H100, scalable unit of 32 DGX, and DGX SuperPOD of 4 units. A closer-look card, after the text&#x27;s scale-out and collectives subsections, counts one leaf&#x27;s ports: 32 down to the nodes, 32 up with two cables to each of the 16 spines, 1,024 leaf-spine cables in all, so the fabric is non-blocking: 16 × 64 × 50 GB/s = 51.2 TB/s of spine capacity against 1,016 × 50 GB/s = 50.8 TB/s from the GPUs. Four routes compare their price: the same rail in the same SU takes 1 switch, the same rail in another SU and cross-rail traffic take 3, and NCCL&#x27;s PXN turns cross-rail traffic into an NVSwitch hop plus 1 switch, which DeepSeek-V3 mirrors by sending tokens over IB to the same-index GPU first. For a Llama 3 70B gradient of 17.5 GB per GPU over 127 ranks, a ring AllReduce puts 34.7 GB on the wire in 0.69 s and SHARP 17.5 GB in 0.35 s. Two tiers stop at 64 leaves × 32 = 2,048 GPUs; from 16 SUs a core tier is added, 2,048 DGX use 512 leaf, 512 spine and 256 core switches, and the longest path grows from 3 to 5 switches. In the restaurant-chain model rails are same-numbered kitchens, PXN a corridor, SHARP a vote count at the exchange and the core tier a national backbone. A note explains why the SuperPOD has 127 DGX rather than 128: a pair of UFM appliances takes one DGX slot, and smaller builds start from a single unit, between 31 and 127 DGX. The cluster drawing follows NVIDIA&#x27;s compute-fabric diagram: Spine-01 to Spine-16 above; SU 1 with Leaf-01-01 to 08 and nodes H100-001 to 032; SU 2 with Leaf-02-01 to 08 and nodes H100-033 to 064; SU 3, elided, repeats the pattern; SU 4 with Leaf-04-01 to 08, nodes H100-097 to 127 and the UFM pair C-UFM 1 and 2; spine-leaf links of 2 × 400 Gb/s, DGX-leaf links of 400 Gb/s and UFM-leaf links of 200 Gb/s. Under the hierarchy, a what-does-what table explains each level: the H100 does the maths and holds its share of the model in HBM; the DGX joins 8 GPUs into one NVLink domain where tensor parallelism lives, while its CPUs run the OS and data loader; the SU is the repeating unit of 32 DGX and 8 leaf switches where same-numbered GPUs meet in one hop; the SuperPOD joins the SUs through the spine, adds storage, management and software, and spreads data parallelism over the whole cluster. A definitions box explains leaf and spine: a leaf is the lower-tier switch that servers plug into, 8 per SU, one per rail, with 32 ports down and 32 up; a spine is the upper-tier switch that joins leaves and has no servers, 16 in all, each linked to all 32 leaves; since every leaf reaches every spine, any two servers meet in at most leaf, spine, leaf." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bp-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bp-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="bp-wrn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="bp-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<text x="16" y="27" text-anchor="start" style="fill:var(--c-accent);font-size:17px;font-weight:700">The Big Picture: The 3 Pillars of AI Infrastructure</text>
<text x="16" y="48" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">From silicon micro-architecture to a 16,384-GPU cluster: Hardware tiers, data flow paths, and physical bottlenecks</text>
<rect x="16" y="60" width="808" height="197" rx="10" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="40" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="108.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">A · Hardware</text>
<text x="108.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SM → SuperPOD</text>
<line x1="178.0" y1="96.0" x2="194.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="196" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="264.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">B · Training Step</text>
<text x="264.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">End-to-End Data Path</text>
<line x1="334.0" y1="96.0" x2="350.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="352" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="420.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">C · 3 Walls</text>
<text x="420.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Memory · Fabric · I/O</text>
<line x1="490.0" y1="96.0" x2="506.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="508" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="576.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">D · Speed &amp; Latency</text>
<text x="576.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Bandwidth Gradient</text>
<line x1="646.0" y1="96.0" x2="662.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="664" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="732.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">E · LLaMA 3 405B</text>
<text x="732.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16k GPU Production Cluster</text>
<text x="40" y="144" text-anchor="start" style="fill:var(--c-text);font-size:12px">This architectural map models how hardware tiers interact from microscopic silicon to data-center scale in an LLM training iteration,</text>
<text x="40" y="161" text-anchor="start" style="fill:var(--c-text);font-size:12px">where data flows, which physical walls the cluster hits, and how modern engineering designs circumvent them.</text>
<rect x="28.0" y="175.0" width="784.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Reference Architecture: NVIDIA H100 SXM5 (Chip) → DGX H100 (Server Node) → DGX SuperPOD (InfiniBand Fat-Tree).</text>
<text x="40" y="211" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">All primary FLOP/s, bandwidth, and latency figures belong to this reference system; alternative accelerators appear for comparison.</text>
<rect x="40.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="56" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Compute</text>
<rect x="195.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="211.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Network &amp; Fabric</text>
<rect x="355.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="371.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Storage &amp; I/O</text>
<rect x="525.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="541.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Bottleneck / Wall</text>
<rect x="710.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="726.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Architectural Fix</text>
<rect x="16" y="269" width="808" height="13038" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="293" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="297.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">A</text>
<text x="56" y="298" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Zoom out: from one SM to the whole cluster</text>
<text x="28" y="319" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Each level is ×N copies of the one before; the dark highlight marks that previous level.</text>
<rect x="28" y="331" width="784" height="1830" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="357" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">1 · SM: the GPU&#x27;s basic worker (Streaming Multiprocessor)</text>
<text x="40" y="377" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">An H100 SXM5 has 132 SMs, and each SM is four identical subpartitions (SMSPs). The numbered circles are items 1–4 of the SM</text>
<text x="40" y="394" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">list in the text: Tensor Core, warp scheduler, FP32 CUDA cores and registers.</text>
<text x="40" y="424" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Hardware: inside one subpartition</text>
<rect x="40" y="438" width="430" height="324" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="52" y="458" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">SM · one of 132</text>
<text x="460" y="458" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">4 subpartitions (SMSP 0–3)</text>
<rect x="50" y="468" width="410" height="186" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="60" y="486" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Subpartition 0 (SMSP 0) · close-up</text>
<rect x="82.0" y="496.0" width="368.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="266.0" y="510" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Warp scheduler + dispatch</text>
<text x="266.0" y="524" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">each cycle issues an instruction from one of 16 warps</text>
<circle cx="68" cy="513" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="68" y="517.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<line x1="266.0" y1="531.0" x2="266.0" y2="541.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="82.0" y="542.0" width="368.0" height="34" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="266.0" y="556" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Register file · 16,384 × 32 bit = 64 KiB</text>
<text x="266.0" y="570" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">the threads&#x27; variables; operands are read from here</text>
<circle cx="68" cy="559" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="68" y="563.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<line x1="187.0" y1="577.0" x2="187.0" y2="587.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="384.0" y1="577.0" x2="384.0" y2="587.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="82.0" y="588.0" width="210.0" height="56" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="187" y="604" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Tensor Core (4th gen)</text>
<text x="187" y="620" text-anchor="middle" style="fill:var(--c-text);font-size:11px">BF16: 1,024 FLOP/cycle (8×8×8)</text>
<text x="187" y="635" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">FP8: 2,048 FLOP/cycle (dense)</text>
<circle cx="68" cy="616" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="68" y="620.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<rect x="318.0" y="588.0" width="132.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="384.0" y="601.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">32 × FP32 CUDA cores</text>
<rect x="318.0" y="609.0" width="132.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="384.0" y="622.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">16 INT32 · 16 FP64</text>
<text x="384" y="641" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">GELU · SiLU · RMSNorm</text>
<circle cx="305" cy="597" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="305" y="601.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<rect x="50" y="662" width="132" height="50" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="60" y="678" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Subpartition 1</text>
<rect x="58.0" y="686.0" width="40.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="78.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">sched</text>
<rect x="102.0" y="686.0" width="46.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="125.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">64 KiB</text>
<rect x="152.0" y="686.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="165.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TC</text>
<rect x="189" y="662" width="132" height="50" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="199" y="678" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Subpartition 2</text>
<rect x="197.0" y="686.0" width="40.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="217.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">sched</text>
<rect x="241.0" y="686.0" width="46.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="264.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">64 KiB</text>
<rect x="291.0" y="686.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="304.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TC</text>
<rect x="328" y="662" width="132" height="50" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="338" y="678" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Subpartition 3</text>
<rect x="336.0" y="686.0" width="40.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="356.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">sched</text>
<rect x="380.0" y="686.0" width="46.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="403.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">64 KiB</text>
<rect x="430.0" y="686.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="443.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TC</text>
<rect x="50.0" y="720.0" width="410.0" height="34" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="255.0" y="734" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Shared memory (SMEM) / L1 · 256 KiB</text>
<text x="255.0" y="748" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">shared by the 4 subpartitions; up to 228 KiB can be SMEM</text>
<line x1="255.0" y1="765.0" x2="255.0" y2="785.0" marker-start="url(#bp-acc)" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.6"/>
<text x="267" y="779" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">TMA: asynchronous copy</text>
<rect x="40.0" y="788.0" width="430.0" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="255.0" y="803.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Outside the SM: L2 (50 MB, the shelf all SMs share) → HBM3 (80 GB)</text>
<rect x="490" y="438" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="500" y="460" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">4</text>
<text x="500" y="477" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">subpartitions</text>
<rect x="649" y="438" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="659" y="460" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1,024</text>
<text x="659" y="477" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FLOP/cycle (TC)</text>
<rect x="490" y="494" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="500" y="516" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">256 KiB</text>
<text x="500" y="533" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">registers / SM</text>
<rect x="649" y="494" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="659" y="516" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">64</text>
<text x="659" y="533" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">resident warps</text>
<text x="490" y="570" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Times 132 SMs</text>
<text x="490" y="588" text-anchor="start" style="fill:var(--c-text);font-size:11px">528 Tensor Cores · 16,896 CUDA cores</text>
<text x="490" y="604" text-anchor="start" style="fill:var(--c-text);font-size:11px">~33 MB registers + ~33 MB SMEM/L1</text>
<text x="490" y="620" text-anchor="start" style="fill:var(--c-text);font-size:11px">8,448 resident warps (270,336 threads)</text>
<rect x="490.0" y="634.0" width="310.0" height="128" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="502" y="654" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 990 TFLOP/s come from?</text>
<text x="502" y="673" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">528 TC × 1,024 FLOP × 1.83 GHz</text>
<text x="502" y="689" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">≈ 989 TFLOP/s ≈ 990 (BF16, dense)</text>
<text x="502" y="708" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1.83 GHz is the H100&#x27;s clock under Tensor Core</text>
<text x="502" y="723" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">load (CUDA cores boost to 1.98 GHz). FP8 does</text>
<text x="502" y="738" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">twice the work per cycle: ~1,979 TFLOP/s. (B200:</text>
<text x="502" y="753" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">148 SMs, 2,048 FLOP/cycle per TC.)</text>
<text x="490" y="784" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">In the text: Section 2, “Silicon-level GPU anatomy”.</text>
<line x1="480.0" y1="438.0" x2="480.0" y2="810.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="850" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Latency hiding: how the warp scheduler works</text>
<text x="40" y="876" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">How far away is the data? (latency of one load, cycles)</text>
<text x="40" y="904" text-anchor="start" style="fill:var(--c-text);font-size:11px">register</text>
<text x="112" y="904" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">never loaded; read directly by the instruction</text>
<text x="40" y="928" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SMEM</text>
<rect x="112" y="916" width="18.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="136.0" y="928" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">29</text>
<text x="40" y="952" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L1</text>
<rect x="112" y="940" width="25.2" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="143.2" y="952" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">41</text>
<text x="40" y="976" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2</text>
<rect x="112" y="964" width="163.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="281.1" y="976" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">263</text>
<text x="40" y="1000" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM3</text>
<rect x="112" y="988" width="296.9" height="16" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="414.9" y="1000" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">479</text>
<text x="40" y="1030" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Measured on Hopper (H800), Luo et al., 2024. HBM is ~16× farther away than</text>
<text x="40" y="1045" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">SMEM.</text>
<text x="490" y="876" text-anchor="start" style="fill:var(--c-text);font-size:12px">Item 2 in the text: an SM keeps 64 warps resident</text>
<text x="490" y="893" text-anchor="start" style="fill:var(--c-text);font-size:12px">and switches between them with zero overhead. Here</text>
<text x="490" y="910" text-anchor="start" style="fill:var(--c-text);font-size:12px">is what that means. When a warp asks HBM for data,</text>
<text x="490" y="927" text-anchor="start" style="fill:var(--c-text);font-size:12px">the answer arrives ~480 cycles later; in that time</text>
<text x="490" y="944" text-anchor="start" style="fill:var(--c-text);font-size:12px">one Tensor Core could have done 480 × 1,024 ≈ 490</text>
<text x="490" y="961" text-anchor="start" style="fill:var(--c-text);font-size:12px">thousand FLOPs. The scheduler does not wait: every</text>
<text x="490" y="978" text-anchor="start" style="fill:var(--c-text);font-size:12px">cycle it picks a ready warp. Every warp&#x27;s</text>
<text x="490" y="995" text-anchor="start" style="fill:var(--c-text);font-size:12px">registers already sit on the chip, so nothing is</text>
<text x="490" y="1012" text-anchor="start" style="fill:var(--c-text);font-size:12px">saved and restored as in a CPU context switch.</text>
<line x1="480.0" y1="864.0" x2="480.0" y2="1045.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1079" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">One scheduler&#x27;s pool: 16 warps (4 × 16 = 64 per SM)</text>
<rect x="40.0" y="1091.0" width="220.0" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="150.0" y="1106.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">scheduler 0 → this cycle: W5</text>
<rect x="40.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="61.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="87.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="108.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="134.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="155.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="181.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="202.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W3</text>
<rect x="228.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="249.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W4</text>
<rect x="275.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="296.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">W5</text>
<rect x="322.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="343.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W6</text>
<rect x="369.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="390.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W7</text>
<rect x="416.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="437.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W8</text>
<rect x="463.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="484.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W9</text>
<rect x="510.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="531.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W10</text>
<rect x="557.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="578.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W11</text>
<rect x="604.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="625.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W12</text>
<rect x="651.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="672.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W13</text>
<rect x="698.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="719.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W14</text>
<rect x="745.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="766.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W15</text>
<path d="M150 1113 V1122 H296.0 V1129" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="40.0" y="1167.0" width="14.0" height="12" rx="5" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60" y="1177" text-anchor="start" style="fill:var(--c-text);font-size:11px">running</text>
<rect x="124.7" y="1167.0" width="14.0" height="12" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="144.7" y="1177" text-anchor="start" style="fill:var(--c-text);font-size:11px">ready</text>
<rect x="197.2" y="1167.0" width="14.0" height="12" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="217.2" y="1177" text-anchor="start" style="fill:var(--c-text);font-size:11px">waiting on memory</text>
<text x="800" y="1177" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">States are illustrative. A warp is 32 threads that run the same instruction together (SIMT).</text>
<text x="40" y="1213" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Timeline: the same work with different numbers of warps</text>
<text x="40" y="1230" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Toy example: each warp computes for 2 units, then waits 6 units for memory.</text>
<text x="40" y="1251" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">4 warps: the scheduler is never idle</text>
<text x="40" y="1273" text-anchor="start" style="fill:var(--c-text);font-size:11px">scheduler</text>
<text x="40" y="1297" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="112.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="149.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="256.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="293.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="40" y="1317" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="112.0" y="1311" width="36.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="148.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="185.0" y="1308" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="292.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="329.0" y="1308" width="70.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="40" y="1337" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="112.0" y="1331" width="72.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="184.0" y="1327" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="221.0" y="1328" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="328.0" y="1327" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="365.0" y="1328" width="34.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="40" y="1357" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W3</text>
<rect x="112.0" y="1351" width="108.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="220.0" y="1347" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="257.0" y="1348" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="364.0" y="1347" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="112.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="148.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="166.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="184.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="202.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="220.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="238.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W3</text>
<rect x="256.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="274.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="292.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="310.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="328.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="346.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="364.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="382.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W3</text>
<line x1="112.0" y1="1373.0" x2="400.0" y2="1373.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="112" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="256" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="400" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="400" y="1401" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">time →</text>
<text x="432" y="1251" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">2 warps: the scheduler is idle 50% of the time</text>
<text x="432" y="1273" text-anchor="start" style="fill:var(--c-text);font-size:11px">scheduler</text>
<text x="432" y="1297" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="504.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="541.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="648.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="685.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="432" y="1317" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="504.0" y="1311" width="36.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="540.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="577.0" y="1308" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="684.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="721.0" y="1308" width="70.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="504.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="522.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="540.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="558.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="576.0" y="1261.0" width="72.0" height="16" rx="5" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="612.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">idle</text>
<rect x="648.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="666.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="684.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="702.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="720.0" y="1261.0" width="72.0" height="16" rx="5" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="756.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">idle</text>
<line x1="504.0" y1="1373.0" x2="792.0" y2="1373.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="504" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="648" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="792" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="792" y="1401" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">time →</text>
<line x1="420.0" y1="1239.0" x2="420.0" y2="1401.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40" y="1417.0" width="18" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="64" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">compute</text>
<rect x="130.7" y="1421.0" width="18" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55;stroke:var(--c-success);stroke-width:0"/>
<text x="154.7" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">ready (queued)</text>
<rect x="264.09999999999997" y="1418.0" width="18" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="288.1" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">waiting on memory</text>
<rect x="415.79999999999995" y="1417.0" width="18" height="12" rx="2" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1"/>
<text x="439.8" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">scheduler idle</text>
<rect x="40.0" y="1441.0" width="760.0" height="24" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="420.0" y="1457.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">warps needed ≈ 1 + wait ÷ compute = 1 + 6 ÷ 2 = 4</text>
<text x="40" y="1489" text-anchor="start" style="fill:var(--c-text);font-size:12px">While one warp waits, the others must take turns issuing work. The longer the wait relative to the compute, the more ready</text>
<text x="40" y="1506" text-anchor="start" style="fill:var(--c-text);font-size:12px">warps you need; HBM&#x27;s ~480 cycles therefore call for either a crowded pool or copies started ahead of time.</text>
<text x="40" y="1546" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Bottleneck: when the pool runs dry</text>
<text x="40" y="1572" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">More registers per thread → fewer warps fit on the SM</text>
<text x="40" y="1596" text-anchor="start" style="fill:var(--c-text);font-size:11px">32 registers/thread</text>
<rect x="182" y="1584" width="190.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="378.0" y="1596" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">64 warps · 100%</text>
<text x="40" y="1620" text-anchor="start" style="fill:var(--c-text);font-size:11px">64 registers/thread</text>
<rect x="182" y="1608" width="95.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="283.0" y="1620" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">32 warps · 50%</text>
<text x="40" y="1644" text-anchor="start" style="fill:var(--c-text);font-size:11px">128 registers/thread</text>
<rect x="182" y="1632" width="47.5" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="235.5" y="1644" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16 warps · 25%</text>
<text x="40" y="1668" text-anchor="start" style="fill:var(--c-text);font-size:11px">255 registers/thread</text>
<rect x="182" y="1656" width="23.8" height="16" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="211.8" y="1668" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">8 warps · 12.5%</text>
<text x="40" y="1696" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">warps = 65,536 ÷ (registers/thread × 32), at most 64; 255 registers round up</text>
<text x="40" y="1711" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">to 256.</text>
<text x="490" y="1572" text-anchor="start" style="fill:var(--c-text);font-size:12px">The registers of item 4 are finite: 65,536 per SM.</text>
<text x="490" y="1589" text-anchor="start" style="fill:var(--c-text);font-size:12px">If each thread wants many, fewer warps fit. When</text>
<text x="490" y="1606" text-anchor="start" style="fill:var(--c-text);font-size:12px">every warp is waiting on memory, the scheduler has</text>
<text x="490" y="1623" text-anchor="start" style="fill:var(--c-text);font-size:12px">nothing to issue; the SM stalls and the Tensor</text>
<text x="490" y="1640" text-anchor="start" style="fill:var(--c-text);font-size:12px">Cores sit idle. Occupancy in profiling tools is</text>
<text x="490" y="1657" text-anchor="start" style="fill:var(--c-text);font-size:12px">how full this pool is.</text>
<line x1="480.0" y1="1560.0" x2="480.0" y2="1711.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="1727.0" width="760.0" height="67" rx="5" style="fill:var(--c-success);fill-opacity:.08;stroke:var(--c-success);stroke-width:1.2"/>
<text x="52" y="1747" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-success);font-weight:700">But few warps are not always bad.</tspan> Hopper&#x27;s fast GEMM and FlashAttention-3 kernels deliberately run few warps with many</text>
<text x="52" y="1764" text-anchor="start" style="fill:var(--c-text);font-size:12px">registers. They hide latency with copies started ahead of time rather than with warp count: producer warps move data</text>
<text x="52" y="1781" text-anchor="start" style="fill:var(--c-text);font-size:12px">from HBM into SMEM with TMA while consumer warps compute on the Tensor Cores with wgmma.</text>
<text x="40" y="1834" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Mental model: the kitchen</text>
<text x="40" y="1852" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The analogy from the text: a GPU is 132 line cooks sharing one HBM pantry (a TPU v5p is two master chefs with giant 128×128 cutting</text>
<text x="40" y="1867" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">boards).</text>
<rect x="40.0" y="1879" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="1879" width="4" height="216" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Line cook</text>
<rect x="52.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SM + warp scheduler</text>
<text x="54.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">Up to 64 order tickets</text>
<text x="54.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">(warps) wait in front of</text>
<text x="54.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">them. While one dish waits</text>
<text x="54.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">for the pantry, they move</text>
<text x="54.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">to a ticket that is ready;</text>
<text x="54.0" y="2020" text-anchor="start" style="fill:var(--c-text);font-size:11px">the switch is free because</text>
<text x="54.0" y="2035" text-anchor="start" style="fill:var(--c-text);font-size:11px">every ticket&#x27;s ingredients</text>
<text x="54.0" y="2050" text-anchor="start" style="fill:var(--c-text);font-size:11px">stay on the counter.</text>
<text x="54.0" y="2083" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">64 warps · 1 instr/cycle</text>
<rect x="233.0" y="1879" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="1879" width="4" height="216" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="247.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Counter</text>
<rect x="245.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="323.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">registers + SMEM/L1</text>
<text x="247.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">Within arm&#x27;s reach. What</text>
<text x="247.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">sits in registers is</text>
<text x="247.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">instant, SMEM arrives in</text>
<text x="247.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">~29 cycles; but space is</text>
<text x="247.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">tight: 256 KiB + 256 KiB</text>
<text x="247.0" y="2020" text-anchor="start" style="fill:var(--c-text);font-size:11px">per SM.</text>
<text x="247.0" y="2083" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">~29 cycles (SMEM)</text>
<rect x="426.0" y="1879" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="1879" width="4" height="216" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="440.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Knife and stove</text>
<rect x="438.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="516.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Tensor Core</text>
<text x="440.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">With ingredients on the</text>
<text x="440.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">counter it does 1,024 BF16</text>
<text x="440.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">operations every cycle.</text>
<text x="440.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">Without them, however</text>
<text x="440.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">powerful, it stays cold.</text>
<text x="440.0" y="2083" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">4 × 1,024 FLOP/cycle</text>
<rect x="619.0" y="1879" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="1879" width="4" height="216" rx="2" style="fill:var(--c-danger);fill-opacity:.85"/>
<text x="633.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Shared pantry</text>
<rect x="631.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="709.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM3</text>
<text x="633.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">A big 80 GB store, but far</text>
<text x="633.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">away: one round trip is</text>
<text x="633.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">~480 cycles. All 132 cooks</text>
<text x="633.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">share the same pantry; the</text>
<text x="633.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">shared shelf in between is</text>
<text x="633.0" y="2020" text-anchor="start" style="fill:var(--c-text);font-size:11px">L2.</text>
<text x="633.0" y="2083" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">~480 cycles · 3.35 TB/s</text>
<rect x="36.0" y="2107.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="2124" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Golden rule: while waiting for tomatoes from the pantry, the cook chops onions for another order (a warp switch). If no prep work</text>
<text x="46" y="2139" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">is left on the counter, because every order is waiting on the pantry, even the world&#x27;s fastest stove stays cold.</text>
<line x1="420.0" y1="2164" x2="420.0" y2="2186" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="2179" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 132 SMs = 1 GPU</text>
<rect x="28" y="2189" width="784" height="1568" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2215" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">2 · GPU: the H100 SXM5 chip</text>
<text x="40" y="2235" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">132 SMs, a 50 MB L2 split in two and 5 HBM3 stacks sit in one package. This card draws the text&#x27;s “Memory hierarchy and the</text>
<text x="40" y="2252" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">arithmetic intensity roofline” subsection: the pyramid, α_crit ≈ 295 and the effect of batch size.</text>
<text x="40" y="2282" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Hardware: a map of the chip</text>
<rect x="40" y="2296" width="480" height="316" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="52" y="2314" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Package · GH100 die: 814 mm², 80 billion transistors, TSMC 4N</text>
<rect x="50.0" y="2324.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="73" y="2362" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="73" y="2378" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="101.0" y="2324.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="118.0" y="2370.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="50.0" y="2421.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="73" y="2459" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="73" y="2475" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="101.0" y="2421.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="118.0" y="2467.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="50.0" y="2518.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="73" y="2556" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="73" y="2572" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="101.0" y="2518.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="118.0" y="2564.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="466.0" y="2324.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="489" y="2362" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="489" y="2378" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="427.0" y="2324.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="444.0" y="2370.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="466.0" y="2421.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="489" y="2459" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="489" y="2475" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="427.0" y="2421.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="444.0" y="2467.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="466.0" y="2518.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.04;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="489" y="2556" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">empty</text>
<text x="489" y="2572" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">site</text>
<rect x="140" y="2324" width="282" height="278" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="150.0" y="2332.0" width="262.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="281.0" y="2345.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink 4 · 18 links (to other GPUs)</text>
<rect x="148" y="2358" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2371" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 0</text>
<rect x="155" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.95"/>
<rect x="155" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="285" y="2358" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2371" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 1</text>
<rect x="292" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2376" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2387" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2376" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2387" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="148" y="2406" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2419" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 2</text>
<rect x="155" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="155" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2424" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="259" y="2435" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="285" y="2406" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2419" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 3</text>
<rect x="292" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2424" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="383" y="2435" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2424" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2435" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="148" y="2488" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2501" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 4</text>
<rect x="155" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="155" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="285" y="2488" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2501" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 5</text>
<rect x="292" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2506" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2517" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2506" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2517" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="148" y="2536" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2549" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 6</text>
<rect x="155" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="155" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2554" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="259" y="2565" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="285" y="2536" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2549" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 7</text>
<rect x="292" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2554" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2565" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2554" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2565" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="148.0" y="2454.0" width="129.0" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="212.5" y="2472.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2 · partition 0</text>
<rect x="285.0" y="2454.0" width="129.0" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="349.5" y="2472.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2 · partition 1</text>
<rect x="150.0" y="2582.0" width="262.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="281.0" y="2595.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe Gen5 ×16 (to the CPU)</text>
<rect x="40" y="2621" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<text x="57" y="2630" text-anchor="start" style="fill:var(--c-text);font-size:11px">active SM</text>
<rect x="129.9" y="2621" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<text x="146.9" y="2630" text-anchor="start" style="fill:var(--c-text);font-size:11px">disabled for yield</text>
<rect x="274.7" y="2621" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.95"/>
<text x="291.7" y="2630" text-anchor="start" style="fill:var(--c-text);font-size:11px">the SM from level 1</text>
<text x="40" y="2646" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">MC: 512-bit memory controller</text>
<text x="40" y="2666" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Each column is one TPC (2 SMs): 8 GPCs × 9 TPCs × 2 = 144 slots, and the product</text>
<text x="40" y="2681" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">enables 132. Which 6 TPCs are off varies from chip to chip; the drawing is one</text>
<text x="40" y="2696" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">example. GPCs on the left sit closer to L2 partition 0, those on the right to</text>
<text x="40" y="2711" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">partition 1.</text>
<rect x="540" y="2296" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="2318" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">132</text>
<text x="550" y="2335" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SMs</text>
<rect x="674" y="2296" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="2318" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">990</text>
<text x="684" y="2335" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (BF16)</text>
<rect x="540" y="2352" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="2374" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">80 GB</text>
<text x="550" y="2391" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM3</text>
<rect x="674" y="2352" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="2374" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">3.35 TB/s</text>
<text x="684" y="2391" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM speed</text>
<text x="540" y="2430" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Full chip and product</text>
<text x="712" y="2448" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GH100 (full)</text>
<text x="800" y="2448" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">H100 SXM5</text>
<line x1="540.0" y1="2454.0" x2="800.0" y2="2454.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="540" y="2470" text-anchor="start" style="fill:var(--c-text);font-size:11px">SMs</text>
<text x="712" y="2470" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">144</text>
<text x="800" y="2470" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">132</text>
<text x="540" y="2486" text-anchor="start" style="fill:var(--c-text);font-size:11px">L2</text>
<text x="712" y="2486" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">60 MB</text>
<text x="800" y="2486" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">50 MB</text>
<text x="540" y="2502" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM stacks</text>
<text x="712" y="2502" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">6</text>
<text x="800" y="2502" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="540" y="2518" text-anchor="start" style="fill:var(--c-text);font-size:11px">memory controllers</text>
<text x="712" y="2518" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">12</text>
<text x="800" y="2518" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">10</text>
<text x="540" y="2542" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The die is so large that few come out</text>
<text x="540" y="2557" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">flawless; switching a few units off raises the</text>
<text x="540" y="2572" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">yield.</text>
<rect x="540.0" y="2588.0" width="260.0" height="128" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="2608" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 3.35 TB/s come from?</text>
<text x="552" y="2627" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5 stacks × 1,024 bit = 5,120 bit</text>
<text x="552" y="2643" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">× ~5.2 Gb/s ÷ 8 ≈ 3.35 TB/s</text>
<text x="552" y="2662" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">The pin rate is derived from the official</text>
<text x="552" y="2677" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">bandwidth: ~670 GB/s and 16 GB per stack.</text>
<text x="552" y="2692" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">The board&#x27;s power limit is 700 W. (B200:</text>
<text x="552" y="2707" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">148 SMs, 192 GB, 8 TB/s.)</text>
<text x="540" y="2738" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">In the text: Section 2, “Memory hierarchy and</text>
<text x="540" y="2753" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">the arithmetic intensity roofline”.</text>
<line x1="530.0" y1="2296.0" x2="530.0" y2="2753.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2793" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Memory hierarchy: where does data live on the chip?</text>
<text x="205" y="2829" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">small and fast</text>
<text x="382" y="2829" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">latency · bandwidth</text>
<polygon points="130.0,2839 280.0,2839 299.0,2881 111.0,2881" style="fill:var(--c-accent);fill-opacity:0.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205" y="2857" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Register file</text>
<text x="205" y="2872" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">256 KiB/SM · ~33 MB</text>
<line x1="305.0" y1="2860.0" x2="374.0" y2="2860.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2857" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">instruction operand</text>
<polygon points="108.0,2885 302.0,2885 321.0,2927 89.0,2927" style="fill:var(--c-accent);fill-opacity:0.15;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205" y="2903" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SMEM / L1</text>
<text x="205" y="2918" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">256 KiB/SM · ~33 MB</text>
<line x1="327.0" y1="2906.0" x2="374.0" y2="2906.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2903" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">~29–41 cycles</text>
<polygon points="86.0,2931 324.0,2931 343.0,2973 67.0,2973" style="fill:var(--c-accent);fill-opacity:0.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205" y="2949" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">L2 (2 partitions)</text>
<text x="205" y="2964" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">50 MB</text>
<line x1="349.0" y1="2952.0" x2="374.0" y2="2952.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2949" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">~263 cycles</text>
<text x="382" y="2964" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">~5.5 TB/s</text>
<polygon points="64.0,2977 346.0,2977 365.0,3019 45.0,3019" style="fill:var(--c-danger);fill-opacity:0.25;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="205" y="2995" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">HBM3</text>
<text x="205" y="3010" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80 GB</text>
<line x1="371.0" y1="2998.0" x2="374.0" y2="2998.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2995" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">~479 cycles</text>
<text x="382" y="3010" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">3.35 TB/s</text>
<text x="205" y="3031" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">large and slow</text>
<text x="205" y="3047" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">capacity (chip-wide)</text>
<text x="540" y="2817" text-anchor="start" style="fill:var(--c-text);font-size:12px">This is the text&#x27;s “steep pyramid”: the</text>
<text x="540" y="2834" text-anchor="start" style="fill:var(--c-text);font-size:12px">upper levels are small but fast, the lower</text>
<text x="540" y="2851" text-anchor="start" style="fill:var(--c-text);font-size:12px">ones large but slow. Going from L2 to HBM,</text>
<text x="540" y="2868" text-anchor="start" style="fill:var(--c-text);font-size:12px">capacity grows 1,600 times while latency</text>
<text x="540" y="2885" text-anchor="start" style="fill:var(--c-text);font-size:12px">almost doubles. The SRAM inside the SMs</text>
<text x="540" y="2902" text-anchor="start" style="fill:var(--c-text);font-size:12px">(~33 MB of registers + ~33 MB of SMEM/L1)</text>
<text x="540" y="2919" text-anchor="start" style="fill:var(--c-text);font-size:12px">is even larger than L2; good kernels such</text>
<text x="540" y="2936" text-anchor="start" style="fill:var(--c-text);font-size:12px">as FlashAttention keep data in these</text>
<text x="540" y="2953" text-anchor="start" style="fill:var(--c-text);font-size:12px">levels and go to HBM rarely. (B200 adds a</text>
<text x="540" y="2970" text-anchor="start" style="fill:var(--c-text);font-size:12px">256 KiB TMEM per SM to this pyramid.)</text>
<line x1="530.0" y1="2805.0" x2="530.0" y2="3047.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="3087" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Roofline: does memory or compute set the speed?</text>
<text x="40" y="3143.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (log)</text>
<line x1="88.0" y1="3153.0" x2="88.0" y2="3317.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="88.0" y1="3317.0" x2="498.0" y2="3317.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="88.0" y1="3277.0" x2="498.0" y2="3277.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3281.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="88.0" y1="3237.0" x2="498.0" y2="3237.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3241.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<line x1="88.0" y1="3197.0" x2="498.0" y2="3197.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3201.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<line x1="88.0" y1="3157.0" x2="498.0" y2="3157.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3161.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1,000</text>
<text x="88" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0.1</text>
<text x="170" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="252" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<text x="334" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<text x="416" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1,000</text>
<text x="498" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10,000</text>
<text x="498.0" y="3348" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">arithmetic intensity α (FLOP/byte, log)</text>
<polyline points="88.0,3296.0 372.6,3157.2 498.0,3157.2" style="fill:none;stroke:var(--c-accent);stroke-width:2.4"/>
<line x1="372.6" y1="3157.2" x2="372.6" y2="3317.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="372.6" cy="3157.2" r="4" style="fill:var(--c-success)"/>
<text x="372.6" y="3148.2" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">α_crit ≈ 295.5</text>
<text x="276.7" y="3247" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">memory bound</text>
<text x="498.0" y="3209.2" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700">compute bound</text>
<circle cx="106.2" cy="3287.1" r="4.5" style="fill:var(--c-danger);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="115.2" y="3302.1" text-anchor="start" style="fill:var(--c-text);font-size:11px">addition (BF16) · α ≈ 0.17</text>
<circle cx="170.0" cy="3256.0" r="4.5" style="fill:var(--c-danger);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="179.0" y="3271.0" text-anchor="start" style="fill:var(--c-text);font-size:11px">decode, B = 1 · α ≈ 1</text>
<circle cx="342.1" cy="3172.1" r="4.5" style="fill:var(--c-danger);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="333.1" y="3165.1" text-anchor="end" style="fill:var(--c-text);font-size:11px">B = 128: 420 TFLOP/s (42%)</text>
<circle cx="448.5" cy="3157.2" r="4.5" style="fill:var(--c-accent);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="448.5" y="3175.2" text-anchor="middle" style="fill:var(--c-text);font-size:11px">B = 4,096: at the roof</text>
<rect x="40.0" y="3361.0" width="480.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="3377.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">α_crit = C_peak ÷ W_mem = 990 ÷ 3.35 ≈ 295.5 FLOP/byte</text>
<text x="40" y="3405" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Points use the Llama 3 70B MLP sizes (D = 8,192, F = 28,672); a BF16 addition is 1</text>
<text x="40" y="3420" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">FLOP per 6 bytes. The roof is min(990, 3.35 · α).</text>
<text x="540" y="3111" text-anchor="start" style="fill:var(--c-text);font-size:12px">The formula from the text: the chip does</text>
<text x="540" y="3128" text-anchor="start" style="fill:var(--c-text);font-size:12px">990 trillion operations a second but reads</text>
<text x="540" y="3145" text-anchor="start" style="fill:var(--c-text);font-size:12px">only 3.35 TB a second from HBM. Unless</text>
<text x="540" y="3162" text-anchor="start" style="fill:var(--c-text);font-size:12px">each byte read feeds ~295 operations, the</text>
<text x="540" y="3179" text-anchor="start" style="fill:var(--c-text);font-size:12px">Tensor Cores wait. For a matrix multiply</text>
<text x="540" y="3196" text-anchor="start" style="fill:var(--c-text);font-size:12px">α(B) = BDF / (B(D+F) + DF): in decode</text>
<text x="540" y="3213" text-anchor="start" style="fill:var(--c-text);font-size:12px">(B = 1) α ≈ 1 and the GPU uses 0.3% of its</text>
<text x="540" y="3230" text-anchor="start" style="fill:var(--c-text);font-size:12px">power; at large batches such as B = 4,096,</text>
<text x="540" y="3247" text-anchor="start" style="fill:var(--c-text);font-size:12px">α ≈ 2,500 and it hits the roof. The memory</text>
<text x="540" y="3264" text-anchor="start" style="fill:var(--c-text);font-size:12px">wall in Part C is the same curve drawn</text>
<text x="540" y="3281" text-anchor="start" style="fill:var(--c-text);font-size:12px">against batch size.</text>
<line x1="530.0" y1="3099.0" x2="530.0" y2="3420.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="3460" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Mental model: the kitchen</text>
<text x="40" y="3478" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The analogy from the text, continued: if an SM is a line cook, the GPU is the kitchen where 132 cooks work.</text>
<rect x="40.0" y="3490" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3490" width="4" height="186" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kitchen</text>
<rect x="52.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU chip · 132 SMs</text>
<text x="54.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">132 line cooks share one</text>
<text x="54.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">kitchen, each at their own</text>
<text x="54.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">counter. The factory builds</text>
<text x="54.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">144 counters and closes the</text>
<text x="54.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">12 most likely to be</text>
<text x="54.0" y="3631" text-anchor="start" style="fill:var(--c-text);font-size:11px">flawed.</text>
<text x="54.0" y="3664" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">132 cooks · 990 TFLOP/s</text>
<rect x="233.0" y="3490" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="3490" width="4" height="186" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="247.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Shared shelf</text>
<rect x="245.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="323.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">L2 · 50 MB</text>
<text x="247.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">Ingredients used often are</text>
<text x="247.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">taken from here without a</text>
<text x="247.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">trip to the pantry. The</text>
<text x="247.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">shelf is split in two; the</text>
<text x="247.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">half across the kitchen is</text>
<text x="247.0" y="3631" text-anchor="start" style="fill:var(--c-text);font-size:11px">a little farther away.</text>
<text x="247.0" y="3664" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">~263 cycles · ~5.5 TB/s</text>
<rect x="426.0" y="3490" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="3490" width="4" height="186" rx="2" style="fill:var(--c-danger);fill-opacity:.85"/>
<text x="440.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Pantry</text>
<rect x="438.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="516.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM3 · 80 GB</text>
<text x="440.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">The big store that holds</text>
<text x="440.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">every ingredient. It has 5</text>
<text x="440.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">doors (stacks), and</text>
<text x="440.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">together they pass 3.35 TB</text>
<text x="440.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">of ingredients a second.</text>
<text x="440.0" y="3664" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">~479 cycles · 3.35 TB/s</text>
<rect x="619.0" y="3490" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="3490" width="4" height="186" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="633.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Service door</text>
<rect x="631.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="709.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink · PCIe</text>
<text x="633.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">The door to other kitchens:</text>
<text x="633.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">450 GB/s over NVLink,</text>
<text x="633.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">64 GB/s to the CPU over</text>
<text x="633.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe (one way). It is 7×</text>
<text x="633.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">narrower than the pantry;</text>
<text x="633.0" y="3631" text-anchor="start" style="fill:var(--c-text);font-size:11px">the next level is about it.</text>
<text x="633.0" y="3664" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">450 GB/s · 64 GB/s</text>
<rect x="36.0" y="3688.0" width="768.0" height="59" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="3705" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Golden rule: the pantry door passes 3.35 TB a second, and the cooks do 990 trillion operations a second. Unless every byte of</text>
<text x="46" y="3720" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">ingredients from the pantry feeds ~295 operations, the cooks wait at the door. That is why so many optimisations share one idea:</text>
<text x="46" y="3735" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">read less from HBM and compute more per byte.</text>
<line x1="420.0" y1="3760" x2="420.0" y2="3782" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="3775" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 8 GPUs = 1 node</text>
<rect x="28" y="3785" width="784" height="2678" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="3811" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">3 · Node: inside the DGX H100</text>
<text x="40" y="3831" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">8 GPUs, 4 NVSwitches, 2 CPUs and 10 ConnectX-7 network cards in one chassis. This card draws the text&#x27;s “Intra-node</text>
<text x="40" y="3848" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">networking: NVLink and NVSwitch fabrics” subsection; the numbered circles on the diagram match the six questions below.</text>
<text x="40" y="3878" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Hardware: NVIDIA&#x27;s system topology</text>
<rect x="36" y="3892" width="488" height="418" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="44.0" y="3904" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="76.0" y="3918" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="76.0" y="3932" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 data</text>
<rect x="130.0" y="3904" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="158.0" y="3918" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="158.0" y="3932" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="206.0" y="3904" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="239.0" y="3918" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="239.0" y="3932" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dual-port</text>
<rect x="288.0" y="3904" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="321.0" y="3918" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="321.0" y="3932" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dual-port</text>
<rect x="374.0" y="3904" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="402.0" y="3918" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="402.0" y="3932" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="452.0" y="3904" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="484.0" y="3918" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="484.0" y="3932" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 data</text>
<line x1="109.0" y1="3922" x2="128.0" y2="3922" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="187.0" y1="3922" x2="204.0" y2="3922" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="356.0" y1="3922" x2="372.0" y2="3922" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="432.0" y1="3922" x2="450.0" y2="3922" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="158.0" y1="3942" x2="158.0" y2="3958" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="402.0" y1="3942" x2="402.0" y2="3958" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="120.0" y="3960" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="158" y="3980" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 0</text>
<text x="158" y="3996" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="364.0" y="3960" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="402" y="3980" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="402" y="3996" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="210.0" y="3962" width="62.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="241.0" y="3976" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 × M.2 OS</text>
<line x1="198.0" y1="3972" x2="208.0" y2="3972" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="288.0" y="3962" width="62.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="319.0" y="3976" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<line x1="352.0" y1="3972" x2="362.0" y2="3972" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="199.0" y1="3994" x2="361.0" y2="3994" marker-start="url(#bp-acc)" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<path d="M196 4004H280V4129A8 18 0 0 1 280 4165V4226" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="40" y="4032" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="286" y="4032" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<path d="M132 4008V4016H69V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M148 4008V4023H127V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M164 4008V4023H185V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M180 4008V4016H243V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M376 4008V4016H317V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M392 4008V4023H375V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M408 4008V4023H433V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M424 4008V4016H491V4038" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="46.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="69.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0</text>
<line x1="69.0" y1="4066" x2="69.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="44.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.34;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="69.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<rect x="104.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="127.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 1</text>
<line x1="127.0" y1="4066" x2="127.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="102.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="127.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<rect x="162.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="185.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 2</text>
<line x1="185.0" y1="4066" x2="185.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="160.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="185.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<rect x="220.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="243.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 3</text>
<line x1="243.0" y1="4066" x2="243.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="218.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="243.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<rect x="294.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="317.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 4</text>
<line x1="317.0" y1="4066" x2="317.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="292.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 4</text>
<rect x="352.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="375.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 5</text>
<line x1="375.0" y1="4066" x2="375.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="350.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="375.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 5</text>
<rect x="410.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="433.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 6</text>
<line x1="433.0" y1="4066" x2="433.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="408.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="433.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 6</text>
<rect x="468.0" y="4040" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="491.0" y="4056" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 7</text>
<line x1="491.0" y1="4066" x2="491.0" y2="4080" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="466.0" y="4082" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="491.0" y="4102.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 7</text>
<!-- GPU 0 (center 69): 4 NVLink drop lines -->
<line x1="60" y1="4114" x2="60" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="66" y1="4114" x2="66" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="72" y1="4114" x2="72" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="78" y1="4114" x2="78" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 1 (center 127): 4 NVLink drop lines -->
<line x1="118" y1="4114" x2="118" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="124" y1="4114" x2="124" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="130" y1="4114" x2="130" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="136" y1="4114" x2="136" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 2 (center 185): 4 NVLink drop lines -->
<line x1="176" y1="4114" x2="176" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="182" y1="4114" x2="182" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="188" y1="4114" x2="188" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="194" y1="4114" x2="194" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 3 (center 243): 4 NVLink drop lines -->
<line x1="234" y1="4114" x2="234" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="240" y1="4114" x2="240" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="246" y1="4114" x2="246" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="252" y1="4114" x2="252" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 4 (center 317): 4 NVLink drop lines -->
<line x1="308" y1="4114" x2="308" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="314" y1="4114" x2="314" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="320" y1="4114" x2="320" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="326" y1="4114" x2="326" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 5 (center 375): 4 NVLink drop lines -->
<line x1="366" y1="4114" x2="366" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="372" y1="4114" x2="372" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="378" y1="4114" x2="378" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="384" y1="4114" x2="384" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 6 (center 433): 4 NVLink drop lines -->
<line x1="424" y1="4114" x2="424" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="430" y1="4114" x2="430" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="436" y1="4114" x2="436" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="442" y1="4114" x2="442" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 7 (center 491): 4 NVLink drop lines -->
<line x1="482" y1="4114" x2="482" y2="4135" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="488" y1="4114" x2="488" y2="4143" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="494" y1="4114" x2="494" y2="4151" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="500" y1="4114" x2="500" y2="4159" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- 4 Crossbar Horizontal Rails spanning across all 8 GPUs -->
<line x1="56" y1="4135" x2="486" y2="4135" style="stroke:#e11d48;stroke-width:1.8"/>
<line x1="62" y1="4143" x2="492" y2="4143" style="stroke:#0284c7;stroke-width:1.8"/>
<line x1="68" y1="4151" x2="498" y2="4151" style="stroke:#ea580c;stroke-width:1.8"/>
<line x1="74" y1="4159" x2="504" y2="4159" style="stroke:#9333ea;stroke-width:1.8"/>
<!-- Feed bundles into 4 NVSwitches from respective rails -->
<!-- Rail 0 (Pink) -> NVSwitch 0 (5 links) -->
<line x1="88" y1="4135" x2="88" y2="4179" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="92" y1="4135" x2="92" y2="4179" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="96" y1="4135" x2="96" y2="4179" style="stroke:#e11d48;stroke-width:1.4"/>
<line x1="100" y1="4135" x2="100" y2="4179" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="104" y1="4135" x2="104" y2="4179" style="stroke:#e11d48;stroke-width:1.2"/>
<!-- Rail 1 (Blue) -> NVSwitch 1 (4 links) -->
<line x1="194" y1="4143" x2="194" y2="4179" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="198" y1="4143" x2="198" y2="4179" style="stroke:#0284c7;stroke-width:1.3"/>
<line x1="202" y1="4143" x2="202" y2="4179" style="stroke:#0284c7;stroke-width:1.3"/>
<line x1="206" y1="4143" x2="206" y2="4179" style="stroke:#0284c7;stroke-width:1.2"/>
<!-- Rail 2 (Orange) -> NVSwitch 2 (4 links) -->
<line x1="354" y1="4151" x2="354" y2="4179" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="358" y1="4151" x2="358" y2="4179" style="stroke:#ea580c;stroke-width:1.3"/>
<line x1="362" y1="4151" x2="362" y2="4179" style="stroke:#ea580c;stroke-width:1.3"/>
<line x1="366" y1="4151" x2="366" y2="4179" style="stroke:#ea580c;stroke-width:1.2"/>
<!-- Rail 3 (Purple) -> NVSwitch 3 (5 links) -->
<line x1="456" y1="4159" x2="456" y2="4179" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="460" y1="4159" x2="460" y2="4179" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="464" y1="4159" x2="464" y2="4179" style="stroke:#9333ea;stroke-width:1.4"/>
<line x1="468" y1="4159" x2="468" y2="4179" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="472" y1="4159" x2="472" y2="4179" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- 4 NVSwitch Chips (Green, matching NVIDIA official diagram) -->
<rect x="52.0" y="4179" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="96.0" y="4196" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 0</text>
<rect x="156.0" y="4179" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="200.0" y="4196" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 1</text>
<rect x="316.0" y="4179" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="360.0" y="4196" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 2</text>
<rect x="420.0" y="4179" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="464.0" y="4196" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 3</text>
<!-- Bottom PCIe Switch (Purple, connecting CPU 0 to NVSwitches) -->
<rect x="250.0" y="4229" width="60.0" height="26" rx="5" style="fill:#8b5cf6;fill-opacity:.18;stroke:#8b5cf6;stroke-width:1.3"/>
<text x="280" y="4241" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">PCIe</text>
<text x="280" y="4252" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">switch</text>
<!-- PCIe Management lines from PCIe Switch UP to NVSwitches (Fabric Manager control plane) -->
<path d="M250 4251H96V4207" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M250 4240H200V4207" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M310 4240H360V4207" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M310 4251H464V4207" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<!-- Explanatory note -->
<text x="280" y="4268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-style:italic">PCIe Control Path: CPU 0 → PCIe switch → 4 NVSwitches (Fabric Manager control plane)</text>
<!-- Component Legend inside Node Box matching NVIDIA official diagram -->
<!-- Legend Row 1: Chips & Modules -->
<rect x="42" y="4279" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1"/>
<text x="56" y="4287" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">ConnectX-7</text>
<rect x="120" y="4279" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.3;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="134" y="4287" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">CX-7 Module</text>
<rect x="200" y="4279" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.4;stroke:var(--c-warn);stroke-width:1"/>
<text x="214" y="4287" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVMe</text>
<rect x="256" y="4279" width="10" height="10" rx="2" style="fill:#8b5cf6;fill-opacity:.3;stroke:#8b5cf6;stroke-width:1"/>
<text x="270" y="4287" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">PCIe Switch</text>
<rect x="336" y="4279" width="10" height="10" rx="2" style="fill:#10b981;fill-opacity:.3;stroke:#10b981;stroke-width:1"/>
<text x="350" y="4287" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVSwitch</text>
<rect x="406" y="4279" width="10" height="10" rx="2" style="fill:#ec4899;fill-opacity:.3;stroke:#ec4899;stroke-width:1"/>
<text x="420" y="4287" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">100 GbE</text>
<!-- Legend Row 2: Interconnects & Fabrics -->
<line x1="50" y1="4301" x2="72" y2="4301" style="stroke:#8b5cf6;stroke-width:1.5"/>
<text x="76" y="4304" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">PCIe Gen5</text>
<line x1="140" y1="4299" x2="162" y2="4299" style="stroke:#e11d48;stroke-width:1.5"/>
<line x1="140" y1="4301" x2="162" y2="4301" style="stroke:#0284c7;stroke-width:1.5"/>
<line x1="140" y1="4303" x2="162" y2="4303" style="stroke:#ea580c;stroke-width:1.5"/>
<line x1="140" y1="4305" x2="162" y2="4305" style="stroke:#9333ea;stroke-width:1.5"/>
<text x="166" y="4304" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVLink (4 Rails)</text>
<line x1="250" y1="4301" x2="272" y2="4301" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="276" y="4304" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">CPU Communication (UPI)</text>
<!-- Question Callout Circles -->
<circle cx="102" cy="3984" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="102" y="3988" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<circle cx="52" cy="4020" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="52" y="4024" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="98" cy="4144" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="98" y="4148" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="239" cy="3951" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="239" y="3955" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="280" cy="3994" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="3998" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<circle cx="280" cy="4211" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="4215" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="540" y="3892" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="3914" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">8</text>
<text x="550" y="3931" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs · 640 GB</text>
<rect x="674" y="3892" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="3914" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">450 GB/s</text>
<text x="684" y="3931" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink, one way</text>
<rect x="540" y="3948" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="3970" text-anchor="start" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">64 GB/s</text>
<text x="550" y="3987" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe, CPU path</text>
<rect x="674" y="3948" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="3970" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">50 GB/s</text>
<text x="684" y="3987" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">network / GPU</text>
<text x="540" y="4018" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 NVSwitch · 3.6 TB/s bisection · 2 × Xeon</text>
<text x="540" y="4033" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8480C (112 cores), 2 TB RAM · 8 × 3.84 TB NVMe</text>
<text x="540" y="4048" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">data · power 10.2 kW</text>
<rect x="540.0" y="4066" width="260.0" height="194" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="4088" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Worked path: one gradient&#x27;s trip</text>
<text x="552" y="4105" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">through the node</text>
<text x="552" y="4128" text-anchor="start" style="fill:var(--c-text);font-size:12px">A gradient from the neighbouring node</text>
<text x="552" y="4145" text-anchor="start" style="fill:var(--c-text);font-size:12px">enters NIC 5 and, via GPUDirect RDMA,</text>
<text x="552" y="4162" text-anchor="start" style="fill:var(--c-text);font-size:12px">is written straight into GPU 5&#x27;s</text>
<text x="552" y="4179" text-anchor="start" style="fill:var(--c-text);font-size:12px">memory without touching CPU 1. If GPU</text>
<text x="552" y="4196" text-anchor="start" style="fill:var(--c-text);font-size:12px">5 then shares it with GPU 2, it</text>
<text x="552" y="4213" text-anchor="start" style="fill:var(--c-text);font-size:12px">travels over NVLink through the</text>
<text x="552" y="4230" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitches, not over the network or</text>
<text x="552" y="4247" text-anchor="start" style="fill:var(--c-text);font-size:12px">through the CPUs.</text>
<rect x="540.0" y="4276.0" width="260.0" height="158" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="4296" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 450 GB/s come from?</text>
<text x="552" y="4315" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">18 links × 25 GB/s = 450 GB/s</text>
<text x="552" y="4331" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">bisection: 4 × 450 × 2 = 3.6 TB/s</text>
<text x="552" y="4350" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Each NVLink carries 25 GB/s each way,</text>
<text x="552" y="4365" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">900 GB/s per GPU both ways. Cut the</text>
<text x="552" y="4380" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">chassis in half (GPUs 0–3 | 4–7) and</text>
<text x="552" y="4395" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">3.6 TB/s crosses the cut. (GB200 NVL72:</text>
<text x="552" y="4410" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">72 GPUs in one NVLink domain, 900 GB/s one</text>
<text x="552" y="4425" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">way per GPU.)</text>
<text x="540" y="4456" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">In the text: Section 3, “Intra-node</text>
<text x="540" y="4471" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">networking: NVLink and NVSwitch fabrics”.</text>
<rect x="36" y="4485" width="768" height="502" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="4507" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Legend: what is each part?</text>
<text x="48" y="4525" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Names come from NVIDIA&#x27;s diagram legend; colours are the poster&#x27;s pillar colours. Roles come from the DGX H100 user guide, and</text>
<text x="48" y="4540" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">those marked “likely” are inferences.</text>
<rect x="48.0" y="4558" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="4572" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CX-7</text>
<text x="126" y="4572" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7” · top, 2 cards</text>
<text x="126" y="4588" text-anchor="start" style="fill:var(--c-text);font-size:11px">Dual-port network card; per the user guide, for</text>
<text x="126" y="4603" text-anchor="start" style="fill:var(--c-text);font-size:11px">the storage fabric and in-band management. Each</text>
<text x="126" y="4618" text-anchor="start" style="fill:var(--c-text);font-size:11px">sits on an upper PCIe switch.</text>
<rect x="48.0" y="4632" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="4646" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0–7</text>
<text x="126" y="4646" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7 Network Module” · 8</text>
<text x="126" y="4662" text-anchor="start" style="fill:var(--c-text);font-size:11px">Single-port 400 Gb/s compute-fabric cards, one per</text>
<text x="126" y="4677" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU; they leave the node through 4 OSFP ports. The</text>
<text x="126" y="4692" text-anchor="start" style="fill:var(--c-text);font-size:11px">grey frames are the modules that carry them</text>
<text x="126" y="4707" text-anchor="start" style="fill:var(--c-text);font-size:11px">(likely).</text>
<rect x="48.0" y="4721" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="4735" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe</text>
<text x="126" y="4735" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · top, 4 + 4</text>
<text x="126" y="4751" text-anchor="start" style="fill:var(--c-text);font-size:11px">Data drives: 8 × 3.84 TB U.2 NVMe as a RAID 0</text>
<text x="126" y="4766" text-anchor="start" style="fill:var(--c-text);font-size:11px">cache, local space for reads and checkpoints.</text>
<rect x="48.0" y="4780" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="4794" text-anchor="middle" style="fill:var(--c-text);font-size:11px">M.2 OS</text>
<text x="126" y="4794" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · next to CPU 0</text>
<text x="126" y="4810" text-anchor="start" style="fill:var(--c-text);font-size:11px">Likely the 2 × 1.92 TB M.2 operating-system drives</text>
<text x="126" y="4825" text-anchor="start" style="fill:var(--c-text);font-size:11px">(RAID 1); the diagram uses the same icon as the</text>
<text x="126" y="4840" text-anchor="start" style="fill:var(--c-text);font-size:11px">data drives.</text>
<rect x="48.0" y="4854" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="4868" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe sw</text>
<text x="126" y="4868" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe Switches” · 3</text>
<text x="126" y="4884" text-anchor="start" style="fill:var(--c-text);font-size:11px">Fan one CPU link out to many devices: the upper</text>
<text x="126" y="4899" text-anchor="start" style="fill:var(--c-text);font-size:11px">two carry NVMe and CX-7, the lower one links the</text>
<text x="126" y="4914" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVSwitches to CPU 0.</text>
<rect x="48.0" y="4928" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="4942" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<text x="126" y="4942" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “100 GbE” · on CPU 1</text>
<text x="126" y="4958" text-anchor="start" style="fill:var(--c-text);font-size:11px">An extra Ethernet card; the diagram does not say</text>
<text x="126" y="4973" text-anchor="start" style="fill:var(--c-text);font-size:11px">what for (likely a service or management network).</text>
<rect x="432.0" y="4558" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="465.0" y="4572" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVSwitch</text>
<text x="510" y="4572" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVSwitch” · 4</text>
<text x="510" y="4588" text-anchor="start" style="fill:var(--c-text);font-size:11px">The NVLink switch between GPUs; every GPU spreads</text>
<text x="510" y="4603" text-anchor="start" style="fill:var(--c-text);font-size:11px">its 18 links over these four as 5 + 4 + 4 + 5.</text>
<rect x="432.0" y="4617" width="66.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="4631" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CPU</text>
<text x="510" y="4631" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU 0 / CPU 1”</text>
<text x="510" y="4647" text-anchor="start" style="fill:var(--c-text);font-size:11px">Intel Xeon 8480C (56 cores each); each feeds four</text>
<text x="510" y="4662" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPUs and their NICs over PCIe.</text>
<line x1="432.0" y1="4686" x2="498.0" y2="4686" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="510" y="4690" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU communication”</text>
<text x="510" y="4706" text-anchor="start" style="fill:var(--c-text);font-size:11px">The link that joins the two sockets; on Intel Xeon</text>
<text x="510" y="4721" text-anchor="start" style="fill:var(--c-text);font-size:11px">this link is called UPI.</text>
<line x1="432.0" y1="4745" x2="498.0" y2="4745" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="510" y="4749" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe”</text>
<text x="510" y="4765" text-anchor="start" style="fill:var(--c-text);font-size:11px">The general road from the CPUs to NICs, GPUs,</text>
<text x="510" y="4780" text-anchor="start" style="fill:var(--c-text);font-size:11px">drives and NVSwitches (PCIe Gen5).</text>
<line x1="432.0" y1="4804" x2="498.0" y2="4804" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="510" y="4808" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Missing from NVIDIA&#x27;s legend: NVLink</text>
<text x="510" y="4824" text-anchor="start" style="fill:var(--c-text);font-size:11px">The coloured bundles below the GPUs are NVLinks:</text>
<text x="510" y="4839" text-anchor="start" style="fill:var(--c-text);font-size:11px">18 links per GPU, 900 GB/s both ways in total.</text>
<line x1="420.0" y1="4554" x2="420.0" y2="4973" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5013" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Reading the topology: six questions</text>
<text x="40" y="5032" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Follow the numbered circles on the diagram in order; each one answers a question.</text>
<circle cx="49" cy="5058" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="5062" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="5062" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Who feeds whom?</text>
<text x="66" y="5082" text-anchor="start" style="fill:var(--c-text);font-size:12px">The node is two mirror-image halves. CPU 0 feeds GPUs</text>
<text x="66" y="5099" text-anchor="start" style="fill:var(--c-text);font-size:12px">0–3 and NICs 0–3 over PCIe; CPU 1 feeds GPUs 4–7 and</text>
<text x="66" y="5116" text-anchor="start" style="fill:var(--c-text);font-size:12px">NICs 4–7. Each half has its own memory, which is called</text>
<text x="66" y="5133" text-anchor="start" style="fill:var(--c-text);font-size:12px">a NUMA domain.</text>
<text x="66" y="5152" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> the process that prepares data for GPU 5</text>
<text x="66" y="5169" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">should run on CPU 1 and use CPU 1&#x27;s memory; otherwise</text>
<text x="66" y="5186" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">every batch first crosses the link between the CPUs.</text>
<circle cx="49" cy="5212" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="5216" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="5216" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">How does a GPU reach the network?</text>
<text x="66" y="5236" text-anchor="start" style="fill:var(--c-text);font-size:12px">Right above every GPU sits its own 400 Gb/s NIC (GPU i</text>
<text x="66" y="5253" text-anchor="start" style="fill:var(--c-text);font-size:12px">↔ NIC i). With GPUDirect RDMA, data goes from GPU</text>
<text x="66" y="5270" text-anchor="start" style="fill:var(--c-text);font-size:12px">memory straight to the NIC and never stops in CPU</text>
<text x="66" y="5287" text-anchor="start" style="fill:var(--c-text);font-size:12px">memory.</text>
<text x="66" y="5306" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> 8 × 50 = 400 GB/s leave the chassis. In</text>
<text x="66" y="5323" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">the cluster NIC i plugs into leaf switch i, which is</text>
<text x="66" y="5340" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">why same-numbered GPUs on different nodes meet in a</text>
<text x="66" y="5357" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">single hop.</text>
<circle cx="49" cy="5383" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="5387" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="5387" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">How do GPUs talk to each other?</text>
<text x="66" y="5407" text-anchor="start" style="fill:var(--c-text);font-size:12px">Each GPU&#x27;s 18 NVLinks run down to the four NVSwitches:</text>
<text x="66" y="5424" text-anchor="start" style="fill:var(--c-text);font-size:12px">5 links to two of them and 4 to the other two. Any two</text>
<text x="66" y="5441" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPUs talk through a single switch hop at 450 GB/s each</text>
<text x="66" y="5458" text-anchor="start" style="fill:var(--c-text);font-size:12px">way.</text>
<text x="66" y="5477" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> this road is 9× wider than the network, so</text>
<text x="66" y="5494" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">tensor parallelism, which talks in every layer, stays</text>
<text x="66" y="5511" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">inside these eight GPUs.</text>
<circle cx="441" cy="5058" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="5062" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="458" y="5062" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Where does storage traffic go?</text>
<text x="458" y="5082" text-anchor="start" style="fill:var(--c-text);font-size:12px">The two upper PCIe switches carry the dual-port CX-7</text>
<text x="458" y="5099" text-anchor="start" style="fill:var(--c-text);font-size:12px">cards (storage fabric and in-band management) and four</text>
<text x="458" y="5116" text-anchor="start" style="fill:var(--c-text);font-size:12px">U.2 NVMe data drives each. Two M.2 operating-system</text>
<text x="458" y="5133" text-anchor="start" style="fill:var(--c-text);font-size:12px">drives also sit next to CPU 0.</text>
<text x="458" y="5152" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> reading data and writing checkpoints use</text>
<text x="458" y="5169" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">different cards and cables from gradient traffic, so a</text>
<text x="458" y="5186" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">big checkpoint does not slow down AllReduce.</text>
<circle cx="441" cy="5212" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="5216" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="5216" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">What joins the two halves?</text>
<text x="458" y="5236" text-anchor="start" style="fill:var(--c-text);font-size:12px">The two CPUs are linked by what NVIDIA calls “CPU</text>
<text x="458" y="5253" text-anchor="start" style="fill:var(--c-text);font-size:12px">communication” (UPI on Xeon). That link carries the</text>
<text x="458" y="5270" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPUs&#x27; own traffic: GPUs in different halves talk over</text>
<text x="458" y="5287" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVLink, not through the CPUs.</text>
<text x="458" y="5306" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> a CPU can reach the other half&#x27;s memory,</text>
<text x="458" y="5323" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">but more slowly; NUMA pinning exists to avoid that</text>
<text x="458" y="5340" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">penalty.</text>
<circle cx="441" cy="5366" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="5370" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="5370" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Who manages the NVSwitches?</text>
<text x="458" y="5390" text-anchor="start" style="fill:var(--c-text);font-size:12px">A PCIe line from CPU 0 reaches the four NVSwitches</text>
<text x="458" y="5407" text-anchor="start" style="fill:var(--c-text);font-size:12px">through the lower PCIe switch. It is a control path:</text>
<text x="458" y="5424" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA Fabric Manager sets up the switches&#x27; routing</text>
<text x="458" y="5441" text-anchor="start" style="fill:var(--c-text);font-size:12px">tables through it, and no GPU data flows on it.</text>
<text x="458" y="5460" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">In practice:</tspan> if Fabric Manager is not running, the</text>
<text x="458" y="5477" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">NVLink fabric never comes up and multi-GPU jobs fail to</text>
<text x="458" y="5494" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">start, even though every GPU looks healthy on its own.</text>
<line x1="420.0" y1="5046" x2="420.0" y2="5511" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5555" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">The NVSwitch fabric: count the links</text>
<text x="156" y="5585" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 0</text>
<text x="240" y="5585" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 1</text>
<text x="324" y="5585" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 2</text>
<text x="408" y="5585" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 3</text>
<text x="487" y="5585" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">total</text>
<line x1="40.0" y1="5591.0" x2="520.0" y2="5591.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="5612" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 0</text>
<rect x="118.0" y="5599.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5612.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5599.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5612.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5599.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5612.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5599.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5612.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5612" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5634" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 1</text>
<rect x="118.0" y="5621.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5634.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5621.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5634.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5621.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5634.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5621.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5634.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5634" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5656" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 2</text>
<rect x="118.0" y="5643.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5656.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5643.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5656.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5643.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5656.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5643.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5656.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5656" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5678" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 3</text>
<rect x="118.0" y="5665.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5678.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5665.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5678.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5665.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5678.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5665.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5678.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5678" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<line x1="40.0" y1="5688.0" x2="520.0" y2="5688.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="40" y="5706" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 4</text>
<rect x="118.0" y="5693.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5706.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5693.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5706.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5693.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5706.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5693.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5706.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5706" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5728" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 5</text>
<rect x="118.0" y="5715.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5728.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5715.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5728.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5715.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5728.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5715.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5728.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5728" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5750" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 6</text>
<rect x="118.0" y="5737.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5750.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5737.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5750.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5737.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5750.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5737.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5750.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5750" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5772" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 7</text>
<rect x="118.0" y="5759.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5759.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5759.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5759.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5772" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<line x1="40.0" y1="5782.0" x2="520.0" y2="5782.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="5802" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">total</text>
<text x="156" y="5802" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">40</text>
<text x="240" y="5802" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">32</text>
<text x="324" y="5802" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">32</text>
<text x="408" y="5802" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">40</text>
<text x="487" y="5802" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">144</text>
<line x1="40.0" y1="5825.0" x2="62.0" y2="5825.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="68" y="5829" text-anchor="start" style="fill:var(--c-text);font-size:11px">cut: GPUs 0–3 | GPUs 4–7 → 4 × 450 × 2 = 3.6 TB/s</text>
<text x="40" y="5847" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Each cell is the number of NVLinks from one GPU down to one NVSwitch (Hot Chips 34).</text>
<text x="540" y="5579" text-anchor="start" style="fill:var(--c-text);font-size:12px">The arithmetic from the text: each GPU</text>
<text x="540" y="5596" text-anchor="start" style="fill:var(--c-text);font-size:12px">spreads its 18 NVLinks over the four</text>
<text x="540" y="5613" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitches as 5 + 4 + 4 + 5. 144 links</text>
<text x="540" y="5630" text-anchor="start" style="fill:var(--c-text);font-size:12px">reach the switches, so all 8 GPUs can send</text>
<text x="540" y="5647" text-anchor="start" style="fill:var(--c-text);font-size:12px">at full speed at once: the fabric is</text>
<text x="540" y="5664" text-anchor="start" style="fill:var(--c-text);font-size:12px">non-blocking and every GPU reaches the</text>
<text x="540" y="5681" text-anchor="start" style="fill:var(--c-text);font-size:12px">other seven in a single switch hop. Each</text>
<text x="540" y="5698" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitch chip has 64 NVLink ports and</text>
<text x="540" y="5715" text-anchor="start" style="fill:var(--c-text);font-size:12px">3.2 TB/s of two-way capacity; the four</text>
<text x="540" y="5732" text-anchor="start" style="fill:var(--c-text);font-size:12px">together make 6.4 TB/s one way. NVSwitch</text>
<text x="540" y="5749" text-anchor="start" style="fill:var(--c-text);font-size:12px">can also do the summing itself (NVLink</text>
<text x="540" y="5766" text-anchor="start" style="fill:var(--c-text);font-size:12px">SHARP), so in an AllReduce the data goes</text>
<text x="540" y="5783" text-anchor="start" style="fill:var(--c-text);font-size:12px">to the switch only once.</text>
<line x1="530.0" y1="5567.0" x2="530.0" y2="5847.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5887" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Which traffic takes which road?</text>
<text x="40" y="5913" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Bandwidth per GPU, one way</text>
<text x="40" y="5937" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (NVSwitch)</text>
<rect x="178" y="5925" width="240.0" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="424.0" y="5937" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">450 GB/s</text>
<text x="40" y="5961" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU)</text>
<rect x="178" y="5949" width="34.1" height="16" rx="2" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="218.1" y="5961" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">64 GB/s</text>
<text x="282.1" y="5961" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷7</text>
<text x="40" y="5985" text-anchor="start" style="fill:var(--c-text);font-size:11px">NIC (network, rail i)</text>
<rect x="178" y="5973" width="26.7" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="210.7" y="5985" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">50 GB/s</text>
<text x="274.7" y="5985" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷9</text>
<text x="40" y="6019" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">job</text>
<text x="232" y="6019" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">road</text>
<text x="370" y="6019" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">how often</text>
<line x1="40.0" y1="6025.0" x2="520.0" y2="6025.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="6043" text-anchor="start" style="fill:var(--c-text);font-size:11px">TP AllReduce</text>
<text x="232" y="6043" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVLink</text>
<text x="370" y="6043" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2× per layer</text>
<text x="40" y="6061" text-anchor="start" style="fill:var(--c-text);font-size:11px">PP activations · DP gradients</text>
<text x="232" y="6061" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NIC → leaf i</text>
<text x="370" y="6061" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">microbatch · step</text>
<text x="40" y="6079" text-anchor="start" style="fill:var(--c-text);font-size:11px">reading data · checkpoints</text>
<text x="232" y="6079" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7 → storage fabric</text>
<text x="370" y="6079" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">step · ~14 min</text>
<text x="40" y="6097" text-anchor="start" style="fill:var(--c-text);font-size:11px">data loader · copies</text>
<text x="232" y="6097" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">CPU → PCIe</text>
<text x="370" y="6097" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">always (same socket)</text>
<text x="40" y="6123" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Each of the 8 steps in Part B uses one of these roads.</text>
<rect x="540.0" y="5901.0" width="260.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="670.0" y="5916.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVLink: Y ≤ 28,672 ÷ 2,200 ≈ 13 ✓</text>
<rect x="540.0" y="5929.0" width="260.0" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="670.0" y="5944.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">network: Y ≤ 28,672 ÷ 19,800 ≈ 1.45 ✗</text>
<text x="540" y="5973" text-anchor="start" style="fill:var(--c-text);font-size:12px">The TP rule from the text: when a layer is</text>
<text x="540" y="5990" text-anchor="start" style="fill:var(--c-text);font-size:12px">split over Y GPUs, the communication hides</text>
<text x="540" y="6007" text-anchor="start" style="fill:var(--c-text);font-size:12px">behind the compute only while Y ≤ F ÷</text>
<text x="540" y="6024" text-anchor="start" style="fill:var(--c-text);font-size:12px">(C/W) (C = 990 TFLOP/s, W = the road&#x27;s</text>
<text x="540" y="6041" text-anchor="start" style="fill:var(--c-text);font-size:12px">bandwidth). C/W is 2,200 on NVLink and</text>
<text x="540" y="6058" text-anchor="start" style="fill:var(--c-text);font-size:12px">19,800 on the network. For Llama 3 70B</text>
<text x="540" y="6075" text-anchor="start" style="fill:var(--c-text);font-size:12px">(F = 28,672), TP = 8 fits inside the</text>
<text x="540" y="6092" text-anchor="start" style="fill:var(--c-text);font-size:12px">chassis; TP that spills out of it starves</text>
<text x="540" y="6109" text-anchor="start" style="fill:var(--c-text);font-size:12px">the compute. That is why the chassis wall</text>
<text x="540" y="6126" text-anchor="start" style="fill:var(--c-text);font-size:12px">is also the TP wall.</text>
<line x1="530.0" y1="5899.0" x2="530.0" y2="6126.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="6166" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Mental model: the restaurant</text>
<text x="40" y="6184" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The analogy from the text, continued: if a GPU is a kitchen, a DGX H100 is a restaurant with eight kitchens.</text>
<rect x="40.0" y="6196" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="6196" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="6216" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Restaurant</text>
<rect x="52.0" y="6224.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="6238.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">DGX H100 · 8 GPUs</text>
<text x="54.0" y="6262" text-anchor="start" style="fill:var(--c-text);font-size:11px">Eight kitchens (GPUs) work</text>
<text x="54.0" y="6277" text-anchor="start" style="fill:var(--c-text);font-size:11px">in one building, each with</text>
<text x="54.0" y="6292" text-anchor="start" style="fill:var(--c-text);font-size:11px">its own pantry (80 GB of</text>
<text x="54.0" y="6307" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3). The whole building</text>
<text x="54.0" y="6322" text-anchor="start" style="fill:var(--c-text);font-size:11px">draws 10.2 kW.</text>
<text x="54.0" y="6385" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">8 kitchens · 640 GB</text>
<rect x="233.0" y="6196" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="6196" width="4" height="201" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="247.0" y="6216" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Inner corridor</text>
<rect x="245.0" y="6224.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.5" y="6238.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink + 4 NVSwitches</text>
<text x="247.0" y="6262" text-anchor="start" style="fill:var(--c-text);font-size:11px">A wide corridor joins the</text>
<text x="247.0" y="6277" text-anchor="start" style="fill:var(--c-text);font-size:11px">kitchens: each reaches the</text>
<text x="247.0" y="6292" text-anchor="start" style="fill:var(--c-text);font-size:11px">other seven through one</text>
<text x="247.0" y="6307" text-anchor="start" style="fill:var(--c-text);font-size:11px">door at 450 GB/s. The</text>
<text x="247.0" y="6322" text-anchor="start" style="fill:var(--c-text);font-size:11px">chattiest work (TP) stays</text>
<text x="247.0" y="6337" text-anchor="start" style="fill:var(--c-text);font-size:11px">here.</text>
<text x="247.0" y="6385" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">450 GB/s · one hop</text>
<rect x="426.0" y="6196" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="6196" width="4" height="201" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="440.0" y="6216" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Managers&#x27; offices</text>
<rect x="438.0" y="6224.0" width="157.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="516.5" y="6238.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 CPUs + 2 TB DRAM</text>
<text x="440.0" y="6262" text-anchor="start" style="fill:var(--c-text);font-size:11px">Two managers each look</text>
<text x="440.0" y="6277" text-anchor="start" style="fill:var(--c-text);font-size:11px">after four kitchens (NUMA).</text>
<text x="440.0" y="6292" text-anchor="start" style="fill:var(--c-text);font-size:11px">They prepare the orders</text>
<text x="440.0" y="6307" text-anchor="start" style="fill:var(--c-text);font-size:11px">(the data loader); fast in</text>
<text x="440.0" y="6322" text-anchor="start" style="fill:var(--c-text);font-size:11px">their own half, slower when</text>
<text x="440.0" y="6337" text-anchor="start" style="fill:var(--c-text);font-size:11px">they run to the other.</text>
<text x="440.0" y="6385" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">PCIe 64 GB/s · UPI</text>
<rect x="619.0" y="6196" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="6196" width="4" height="201" rx="2" style="fill:var(--c-danger);fill-opacity:.85"/>
<text x="633.0" y="6216" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Courier doors</text>
<rect x="631.0" y="6224.0" width="157.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="709.5" y="6238.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">8 NICs · 400 Gb/s</text>
<text x="633.0" y="6262" text-anchor="start" style="fill:var(--c-text);font-size:11px">Every kitchen has its own</text>
<text x="633.0" y="6277" text-anchor="start" style="fill:var(--c-text);font-size:11px">courier door to the other</text>
<text x="633.0" y="6292" text-anchor="start" style="fill:var(--c-text);font-size:11px">branches (nodes). It is 9×</text>
<text x="633.0" y="6307" text-anchor="start" style="fill:var(--c-text);font-size:11px">narrower than the corridor;</text>
<text x="633.0" y="6322" text-anchor="start" style="fill:var(--c-text);font-size:11px">heavy work reaches it only</text>
<text x="633.0" y="6337" text-anchor="start" style="fill:var(--c-text);font-size:11px">at layer and step</text>
<text x="633.0" y="6352" text-anchor="start" style="fill:var(--c-text);font-size:11px">boundaries.</text>
<text x="633.0" y="6385" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">8 × 50 = 400 GB/s</text>
<rect x="36.0" y="6409.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="6426" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Golden rule: the chattiest work stays in the inner corridor. TP talks in every layer, so it never leaves the chassis; only the</text>
<text x="46" y="6441" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">sparser traffic at layer boundaries (PP) and at the end of a step (DP) reaches the courier doors.</text>
<line x1="420.0" y1="6466" x2="420.0" y2="6488" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="6481" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 32 nodes = 1 SU · × 4 SUs = SuperPOD</text>
<rect x="28" y="6491" width="784" height="2486" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="6517" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4 · Cluster: the DGX SuperPOD</text>
<text x="40" y="6537" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">4 SUs × 32 slots, one slot for UFM: 127 DGX, 1,016 H100s. This card draws the text&#x27;s “Scaling beyond the node: SuperPods and</text>
<text x="40" y="6554" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">rail-optimized fat-tree topologies” subsection; the port count and traffic paths of the fabric are in card 4b.</text>
<text x="40" y="6584" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Remember the hierarchy: from part to whole</text>
<rect x="40.0" y="6594" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="126.0" y="6608" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="126.0" y="6622" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">one GPU module</text>
<line x1="214.0" y1="6614" x2="234.0" y2="6614" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="236.0" y="6594" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="322.0" y="6608" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="322.0" y="6622" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8-GPU server</text>
<line x1="410.0" y1="6614" x2="430.0" y2="6614" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="432.0" y="6594" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="518.0" y="6608" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="518.0" y="6622" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 DGX = 256 GPUs</text>
<line x1="606.0" y1="6614" x2="626.0" y2="6614" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="628.0" y="6594" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="714.0" y="6608" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="714.0" y="6622" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 SUs = 127 DGX</text>
<text x="40" y="6654" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">H100 SXM5 modules plug not into the motherboard but into the HGX baseboard that carries 8 GPUs and 4 NVSwitches; the DGX H100&#x27;s two</text>
<text x="40" y="6669" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">CPUs are Intel Xeon 8480C. A SuperPOD is built from 32-DGX units (SUs); the reference design has 4 SUs.</text>
<text x="40" y="6701" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">What are leaf and spine?</text>
<rect x="40" y="6713" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="48.0" y="6721" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="162.0" y="6736.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Leaf switch</text>
<text x="52" y="6763" text-anchor="start" style="fill:var(--c-text);font-size:12px">The lower tier, where servers plug</text>
<text x="52" y="6780" text-anchor="start" style="fill:var(--c-text);font-size:12px">in directly; the leaves are the</text>
<text x="52" y="6797" text-anchor="start" style="fill:var(--c-text);font-size:12px">tips of the tree, nearest to the</text>
<text x="52" y="6814" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPUs.</text>
<text x="52" y="6837" text-anchor="start" style="fill:var(--c-text);font-size:12px">In the SuperPOD: 8 per SU, one per</text>
<text x="52" y="6854" text-anchor="start" style="fill:var(--c-text);font-size:12px">rail. 32 of its 64 ports go down to</text>
<text x="52" y="6871" text-anchor="start" style="fill:var(--c-text);font-size:12px">DGX nodes and 32 go up to spines.</text>
<rect x="296" y="6713" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="304.0" y="6721" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="418.0" y="6736.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Spine switch</text>
<text x="308" y="6763" text-anchor="start" style="fill:var(--c-text);font-size:12px">The upper tier that joins the</text>
<text x="308" y="6780" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaves to each other; no server</text>
<text x="308" y="6797" text-anchor="start" style="fill:var(--c-text);font-size:12px">ever plugs into a spine.</text>
<text x="308" y="6820" text-anchor="start" style="fill:var(--c-text);font-size:12px">In the SuperPOD: 16 in all, each</text>
<text x="308" y="6837" text-anchor="start" style="fill:var(--c-text);font-size:12px">linked to all 32 leaves with two</text>
<text x="308" y="6854" text-anchor="start" style="fill:var(--c-text);font-size:12px">lines apiece, which fills its</text>
<text x="308" y="6871" text-anchor="start" style="fill:var(--c-text);font-size:12px">64 ports.</text>
<text x="552" y="6727" text-anchor="start" style="fill:var(--c-text);font-size:12px">The rule of the two tiers is simple:</text>
<text x="552" y="6744" text-anchor="start" style="fill:var(--c-text);font-size:12px">every leaf connects to every spine. Any</text>
<text x="552" y="6761" text-anchor="start" style="fill:var(--c-text);font-size:12px">two servers therefore meet in at most</text>
<text x="552" y="6778" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf → spine → leaf, and the path does</text>
<text x="552" y="6795" text-anchor="start" style="fill:var(--c-text);font-size:12px">not get longer as the cluster grows. To</text>
<text x="552" y="6812" text-anchor="start" style="fill:var(--c-text);font-size:12px">add servers you add leaves; to add</text>
<text x="552" y="6829" text-anchor="start" style="fill:var(--c-text);font-size:12px">bandwidth you add spines. The pattern</text>
<text x="552" y="6846" text-anchor="start" style="fill:var(--c-text);font-size:12px">comes from telephone exchanges (Clos,</text>
<text x="552" y="6863" text-anchor="start" style="fill:var(--c-text);font-size:12px">1953); when the links get thicker toward</text>
<text x="552" y="6880" text-anchor="start" style="fill:var(--c-text);font-size:12px">the top it is called a fat-tree</text>
<text x="552" y="6897" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Leiserson, 1985).</text>
<text x="552" y="6919" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Analogy: a leaf is the local exchange of a</text>
<text x="552" y="6934" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">neighbourhood, a spine the trunk line</text>
<text x="552" y="6949" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">between cities.</text>
<text x="40" y="6987" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">What does what?</text>
<text x="168" y="7013" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Job</text>
<text x="384" y="7013" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Holds</text>
<text x="600" y="7013" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Links and parallelism</text>
<line x1="40.0" y1="7022" x2="800.0" y2="7022" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="7035" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="7051" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="168" y="7051" text-anchor="start" style="fill:var(--c-text);font-size:12px">does the model&#x27;s maths; the</text>
<text x="168" y="7068" text-anchor="start" style="fill:var(--c-text);font-size:12px">Tensor Cores in its 132 SMs run</text>
<text x="168" y="7085" text-anchor="start" style="fill:var(--c-text);font-size:12px">the matrix multiplications.</text>
<text x="384" y="7051" text-anchor="start" style="fill:var(--c-text);font-size:12px">this GPU&#x27;s share of the weights,</text>
<text x="384" y="7068" text-anchor="start" style="fill:var(--c-text);font-size:12px">activations and optimizer state,</text>
<text x="384" y="7085" text-anchor="start" style="fill:var(--c-text);font-size:12px">in 80 GB of HBM3.</text>
<text x="600" y="7051" text-anchor="start" style="fill:var(--c-text);font-size:12px">18 NVLinks reach the other</text>
<text x="600" y="7068" text-anchor="start" style="fill:var(--c-text);font-size:12px">7 GPUs in the chassis, PCIe</text>
<text x="600" y="7085" text-anchor="start" style="fill:var(--c-text);font-size:12px">reaches the CPU and its own NIC;</text>
<text x="600" y="7102" text-anchor="start" style="fill:var(--c-text);font-size:12px">every GPU has a TP, a PP and a</text>
<text x="600" y="7119" text-anchor="start" style="fill:var(--c-text);font-size:12px">DP coordinate.</text>
<line x1="40.0" y1="7129" x2="800.0" y2="7129" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="7143" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="7159" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="168" y="7159" text-anchor="start" style="fill:var(--c-text);font-size:12px">joins 8 GPUs into one NVLink</text>
<text x="168" y="7176" text-anchor="start" style="fill:var(--c-text);font-size:12px">domain; two Xeon CPUs run the</text>
<text x="168" y="7193" text-anchor="start" style="fill:var(--c-text);font-size:12px">OS, the data loader and the</text>
<text x="168" y="7210" text-anchor="start" style="fill:var(--c-text);font-size:12px">processes that send work to the</text>
<text x="168" y="7227" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPUs.</text>
<text x="384" y="7159" text-anchor="start" style="fill:var(--c-text);font-size:12px">on the HGX baseboard 8 GPUs and</text>
<text x="384" y="7176" text-anchor="start" style="fill:var(--c-text);font-size:12px">4 NVSwitches, plus 2 TB of RAM,</text>
<text x="384" y="7193" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 × 3.84 TB of NVMe cache and 8</text>
<text x="384" y="7210" text-anchor="start" style="fill:var(--c-text);font-size:12px">+ 2 network cards.</text>
<text x="600" y="7159" text-anchor="start" style="fill:var(--c-text);font-size:12px">tensor parallelism (TP 8) stays</text>
<text x="600" y="7176" text-anchor="start" style="fill:var(--c-text);font-size:12px">inside this chassis; the chassis</text>
<text x="600" y="7193" text-anchor="start" style="fill:var(--c-text);font-size:12px">reaches the network over 8</text>
<text x="600" y="7210" text-anchor="start" style="fill:var(--c-text);font-size:12px">rails, one per GPU.</text>
<line x1="40.0" y1="7237" x2="800.0" y2="7237" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="7251" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="7267" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="168" y="7267" text-anchor="start" style="fill:var(--c-text);font-size:12px">the repeating unit of the build</text>
<text x="168" y="7284" text-anchor="start" style="fill:var(--c-text);font-size:12px">and of the network: 32 DGX</text>
<text x="168" y="7301" text-anchor="start" style="fill:var(--c-text);font-size:12px">(8 racks × 4) and the 8 leaf</text>
<text x="168" y="7318" text-anchor="start" style="fill:var(--c-text);font-size:12px">switches that join them.</text>
<text x="384" y="7267" text-anchor="start" style="fill:var(--c-text);font-size:12px">256 GPUs; each rail is one leaf,</text>
<text x="384" y="7284" text-anchor="start" style="fill:var(--c-text);font-size:12px">so Leaf-xx-0i gathers NIC i of</text>
<text x="384" y="7301" text-anchor="start" style="fill:var(--c-text);font-size:12px">every node.</text>
<text x="600" y="7267" text-anchor="start" style="fill:var(--c-text);font-size:12px">same-numbered GPUs meet in one</text>
<text x="600" y="7284" text-anchor="start" style="fill:var(--c-text);font-size:12px">hop; pipeline and data-parallel</text>
<text x="600" y="7301" text-anchor="start" style="fill:var(--c-text);font-size:12px">traffic between nodes flows here</text>
<text x="600" y="7318" text-anchor="start" style="fill:var(--c-text);font-size:12px">first, rail by rail.</text>
<line x1="40.0" y1="7328" x2="800.0" y2="7328" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="7342" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="7358" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="168" y="7358" text-anchor="start" style="fill:var(--c-text);font-size:12px">joins the SUs into one</text>
<text x="168" y="7375" text-anchor="start" style="fill:var(--c-text);font-size:12px">non-blocking network through the</text>
<text x="168" y="7392" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine layer and adds the</text>
<text x="168" y="7409" text-anchor="start" style="fill:var(--c-text);font-size:12px">separate storage and management</text>
<text x="168" y="7426" text-anchor="start" style="fill:var(--c-text);font-size:12px">networks, UFM, and software such</text>
<text x="168" y="7443" text-anchor="start" style="fill:var(--c-text);font-size:12px">as Base Command and Slurm.</text>
<text x="384" y="7358" text-anchor="start" style="fill:var(--c-text);font-size:12px">127 DGX = 1,016 GPUs; a single</text>
<text x="384" y="7375" text-anchor="start" style="fill:var(--c-text);font-size:12px">training job can span the whole</text>
<text x="384" y="7392" text-anchor="start" style="fill:var(--c-text);font-size:12px">cluster.</text>
<text x="600" y="7358" text-anchor="start" style="fill:var(--c-text);font-size:12px">data parallelism spans the</text>
<text x="600" y="7375" text-anchor="start" style="fill:var(--c-text);font-size:12px">cluster and cross-rail traffic</text>
<text x="600" y="7392" text-anchor="start" style="fill:var(--c-text);font-size:12px">turns through a spine; Slurm</text>
<text x="600" y="7409" text-anchor="start" style="fill:var(--c-text);font-size:12px">places the job on nodes and</text>
<text x="600" y="7426" text-anchor="start" style="fill:var(--c-text);font-size:12px">takes failed nodes out of</text>
<text x="600" y="7443" text-anchor="start" style="fill:var(--c-text);font-size:12px">service.</text>
<rect x="40.0" y="7461" width="760.0" height="67" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="52" y="7481" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Why 127, not 128?</tspan> On paper 4 SUs × 32 = 128 DGX. In NVIDIA&#x27;s deployment pattern, however, a pair of UFM appliances takes</text>
<text x="52" y="7498" text-anchor="start" style="fill:var(--c-text);font-size:12px">one DGX&#x27;s place (C-UFM 1 and 2 in SU 4 in NVIDIA&#x27;s diagram), so a full SuperPOD holds at most 127 DGX, i.e. 1,016 GPUs.</text>
<text x="52" y="7515" text-anchor="start" style="fill:var(--c-text);font-size:12px">The cable table shows it too: 127 × 8 + 4 UFM = 1,020 cables. Smaller builds start from a single SU (31 to 127 DGX).</text>
<rect x="36" y="7540" width="768" height="392" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="214.0" y="7554" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="252.0" y="7569" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-01</text>
<rect x="318.0" y="7554" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="7569" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-02</text>
<rect x="530.0" y="7554" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="568.0" y="7569" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-16</text>
<text x="462" y="7569" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-weight:700">⋯</text>
<text x="620" y="7569" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 × QM9700</text>
<rect x="46" y="7610" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="52.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="83.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-01</text>
<rect x="120.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="151.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-02</text>
<rect x="206.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-08</text>
<text x="194" y="7637" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="89.0" y1="7696" x2="83.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="7696" x2="151.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="7696" x2="237.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="54" y="7696" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="89" y="7712" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-001</text>
<text x="89" y="7727" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="175.0" y1="7696" x2="83.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="7696" x2="151.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="7696" x2="237.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="140" y="7696" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="175" y="7712" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-032</text>
<text x="175" y="7727" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="132" y="7719" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="56" y="7774" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 1</text>
<text x="98" y="7774" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-001 … 032 · 8 leaves</text>
<rect x="290" y="7610" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="296.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="327.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-01</text>
<rect x="364.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="395.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-02</text>
<rect x="450.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="481.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-08</text>
<text x="438" y="7637" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="333.0" y1="7696" x2="327.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="7696" x2="395.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="7696" x2="481.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="298" y="7696" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="333" y="7712" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-033</text>
<text x="333" y="7727" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="419.0" y1="7696" x2="327.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="7696" x2="395.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="7696" x2="481.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="384" y="7696" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="419" y="7712" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-064</text>
<text x="419" y="7727" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="376" y="7719" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="300" y="7774" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 2</text>
<text x="342" y="7774" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-033 … 064 · 8 leaves</text>
<rect x="574" y="7610" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="580.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="611.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-01</text>
<rect x="648.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="679.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-02</text>
<rect x="734.0" y="7622" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="765.0" y="7636" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-08</text>
<text x="722" y="7637" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="617.0" y1="7696" x2="611.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="7696" x2="679.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="7696" x2="765.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="582" y="7696" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="617" y="7712" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-097</text>
<text x="617" y="7727" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="703.0" y1="7696" x2="611.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="7696" x2="679.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="7696" x2="765.0" y2="7642" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="668" y="7696" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="703" y="7712" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-127</text>
<text x="703" y="7727" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="660" y="7719" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="584" y="7774" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 4</text>
<text x="626" y="7774" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-097 … 127 + C-UFM</text>
<text x="545" y="7702" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">⋯</text>
<text x="545" y="7722" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SU 3</text>
<path d="M746 7711H743V7642" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="7702" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="7715" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 1</text>
<path d="M746 7735H740V7642" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="7726" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="7739" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 2</text>
<line x1="252.0" y1="7576" x2="83.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="151.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="237.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="327.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="395.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="481.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="611.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="679.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7576" x2="765.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="83.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="151.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="237.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="327.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="395.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="481.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="611.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="679.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7576" x2="765.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="83.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="151.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="237.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="327.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="395.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="481.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="611.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="679.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7576" x2="765.0" y2="7622" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="48.0" y1="7808" x2="70.0" y2="7808" style="stroke:var(--c-danger);stroke-width:2.4"/>
<text x="76" y="7812" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine–leaf: 2 × 400 Gb/s</text>
<line x1="300.0" y1="7808" x2="322.0" y2="7808" style="stroke:var(--c-warn);stroke-width:2.4"/>
<text x="328" y="7812" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX–leaf: 400 Gb/s</text>
<line x1="552.0" y1="7808" x2="574.0" y2="7808" style="stroke:var(--c-accent);stroke-width:2.4"/>
<text x="580" y="7812" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">UFM–leaf: 200 Gb/s</text>
<text x="48" y="7836" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Line colours follow NVIDIA&#x27;s diagram and mark the cable type here (not the poster&#x27;s pillar colours). Names follow it too:</text>
<text x="48" y="7851" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Leaf-SU-rail, so Leaf-04-08 is rail 8 of SU 4 and NIC 0 to 7 of every DGX go to Leaf-xx-01 to 08. SU 3 (H100-065 … 096) repeats the</text>
<text x="48" y="7866" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">same layout.</text>
<rect x="44.0" y="7880" width="368.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="54" y="7896" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">storage fabric (own InfiniBand)</text>
<text x="54" y="7912" text-anchor="start" style="fill:var(--c-text);font-size:11px">16 switches · ≥ 40 GB/s per node</text>
<rect x="428.0" y="7880" width="368.0" height="40" rx="5" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="438.0" y="7896" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">management (Ethernet)</text>
<text x="438.0" y="7912" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">in-band 100 GbE · OOB 1 GbE</text>
<rect x="40.0" y="7944" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="7966" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">127</text>
<text x="50.0" y="7983" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX nodes</text>
<rect x="234.0" y="7944" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="244.0" y="7966" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1,016</text>
<text x="244.0" y="7983" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs</text>
<rect x="428.0" y="7944" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="438.0" y="7966" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">32 + 16</text>
<text x="438.0" y="7983" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">leaf + spine</text>
<rect x="622.0" y="7944" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="632.0" y="7966" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">4</text>
<text x="632.0" y="7983" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">separate networks</text>
<text x="40" y="8014" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">127 DGX (UFM pair in SU 4)</text>
<text x="420" y="8014" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">within a rail: one hop to 31 nodes</text>
<text x="40" y="8030" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">compute: 32 leaf + 16 spine</text>
<text x="420" y="8030" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">every leaf to every spine, 2 × 400 Gb/s</text>
<text x="40" y="8046" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1:1 fat-tree · SHARPv3</text>
<text x="420" y="8046" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 networks: compute, storage, 2 mgmt</text>
<text x="40" y="8090" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Rails: why does every node&#x27;s NIC i go to leaf i?</text>
<text x="40" y="8120" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">Top-of-rack (ToR) layout</text>
<rect x="96.0" y="8130.0" width="120.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="8143.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="75.0" y1="8172.0" x2="156.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="156.0" y1="8172.0" x2="156.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="237.0" y1="8172.0" x2="156.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="47.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="55.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="63.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="71.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="79.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="87.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="95.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="103.0" y1="8264.0" x2="75.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="128.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="136.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="144.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="152.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="160.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="168.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="176.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="184.0" y1="8264.0" x2="156.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="209.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="217.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="225.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="233.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="241.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="249.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="257.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="265.0" y1="8264.0" x2="237.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<polyline points="71.0,8264.0 75.0,8190.0 75.0,8172.0 156.0,8148.0 156.0,8172.0 156.0,8190.0 152.0,8264.0" style="fill:none;stroke:var(--c-danger);stroke-width:2.4"/>
<rect x="51.0" y="8172.0" width="48.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="75.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf A</text>
<rect x="132.0" y="8172.0" width="48.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf B</text>
<rect x="213.0" y="8172.0" width="48.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf C</text>
<rect x="40" y="8238" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="75" y="8254" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX A</text>
<circle cx="47" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="55" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="63" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="71" cy="8264" r="2.6" style="fill:var(--c-danger);fill-opacity:1"/>
<circle cx="79" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="87" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="95" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="103" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="121" y="8238" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="156" y="8254" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX B</text>
<circle cx="128" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="136" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="144" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="152" cy="8264" r="2.6" style="fill:var(--c-danger);fill-opacity:1"/>
<circle cx="160" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="168" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="176" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="184" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="202" y="8238" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="237" y="8254" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX C</text>
<circle cx="209" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="217" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="225" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="233" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="241" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="249" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="257" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="265" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<text x="40" y="8294" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">GPU 3 (A) → GPU 3 (B):</text>
<text x="40" y="8309" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">leaf → spine → leaf, 3 hops</text>
<text x="288" y="8120" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">Rail layout (NVIDIA)</text>
<rect x="344.0" y="8130.0" width="120.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="404.0" y="8143.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="301.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="330.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="359.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="388.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="417.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="446.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="475.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="504.0" y1="8172.0" x2="404.0" y2="8148.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="295.0" y1="8264.0" x2="301.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="303.0" y1="8264.0" x2="330.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="311.0" y1="8264.0" x2="359.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="319.0" y1="8264.0" x2="388.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="327.0" y1="8264.0" x2="417.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="335.0" y1="8264.0" x2="446.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="343.0" y1="8264.0" x2="475.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="351.0" y1="8264.0" x2="504.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="376.0" y1="8264.0" x2="301.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="384.0" y1="8264.0" x2="330.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="392.0" y1="8264.0" x2="359.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="400.0" y1="8264.0" x2="388.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="408.0" y1="8264.0" x2="417.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="416.0" y1="8264.0" x2="446.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="424.0" y1="8264.0" x2="475.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="432.0" y1="8264.0" x2="504.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="457.0" y1="8264.0" x2="301.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="465.0" y1="8264.0" x2="330.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="473.0" y1="8264.0" x2="359.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="481.0" y1="8264.0" x2="388.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="489.0" y1="8264.0" x2="417.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="497.0" y1="8264.0" x2="446.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="505.0" y1="8264.0" x2="475.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="513.0" y1="8264.0" x2="504.0" y2="8190.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<polyline points="319.0,8264.0 388.0,8190.0 400.0,8264.0" style="fill:none;stroke:var(--c-success);stroke-width:2.4"/>
<rect x="288.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="301.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L0</text>
<rect x="317.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="330.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L1</text>
<rect x="346.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="359.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2</text>
<rect x="375.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="388.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L3</text>
<rect x="404.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="417.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L4</text>
<rect x="433.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="446.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L5</text>
<rect x="462.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="475.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L6</text>
<rect x="491.0" y="8172.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="504.0" y="8185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L7</text>
<rect x="288" y="8238" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="323" y="8254" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX A</text>
<circle cx="295" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="303" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="311" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="319" cy="8264" r="2.6" style="fill:var(--c-success);fill-opacity:1"/>
<circle cx="327" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="335" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="343" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="351" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="369" y="8238" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="404" y="8254" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX B</text>
<circle cx="376" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="384" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="392" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="400" cy="8264" r="2.6" style="fill:var(--c-success);fill-opacity:1"/>
<circle cx="408" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="416" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="424" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="432" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="450" y="8238" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="485" y="8254" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX C</text>
<circle cx="457" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="465" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="473" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="481" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="489" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="497" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="505" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="513" cy="8264" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<text x="288" y="8294" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">GPU 3 (A) → GPU 3 (B): one leaf, 1 hop</text>
<text x="40" y="8331" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The drawing shows 3 nodes per SU and 8 NICs per node; the green and red lines are the</text>
<text x="40" y="8346" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">same message&#x27;s path.</text>
<text x="540" y="8114" text-anchor="start" style="fill:var(--c-text);font-size:12px">The text&#x27;s “rail-optimized” wiring: the</text>
<text x="540" y="8131" text-anchor="start" style="fill:var(--c-text);font-size:12px">NIC of GPU i in every node plugs into the</text>
<text x="540" y="8148" text-anchor="start" style="fill:var(--c-text);font-size:12px">same leaf i. Because TP 8 stays inside the</text>
<text x="540" y="8165" text-anchor="start" style="fill:var(--c-text);font-size:12px">chassis, GPU i in every node holds the</text>
<text x="540" y="8182" text-anchor="start" style="fill:var(--c-text);font-size:12px">same slice of the model, so DP gradients</text>
<text x="540" y="8199" text-anchor="start" style="fill:var(--c-text);font-size:12px">and PP activations also flow between</text>
<text x="540" y="8216" text-anchor="start" style="fill:var(--c-text);font-size:12px">same-numbered GPUs. With rails that</text>
<text x="540" y="8233" text-anchor="start" style="fill:var(--c-text);font-size:12px">traffic crosses a single leaf inside an SU</text>
<text x="540" y="8250" text-anchor="start" style="fill:var(--c-text);font-size:12px">and never climbs to the spine; in a</text>
<text x="540" y="8267" text-anchor="start" style="fill:var(--c-text);font-size:12px">classic top-of-rack layout the same</text>
<text x="540" y="8284" text-anchor="start" style="fill:var(--c-text);font-size:12px">message goes leaf → spine → leaf. The</text>
<text x="540" y="8301" text-anchor="start" style="fill:var(--c-text);font-size:12px">price is cross-rail traffic (GPU 3 → GPU</text>
<text x="540" y="8318" text-anchor="start" style="fill:var(--c-text);font-size:12px">5): it crosses a spine, or NCCL&#x27;s PXN</text>
<text x="540" y="8335" text-anchor="start" style="fill:var(--c-text);font-size:12px">first moves it to the right rail over</text>
<text x="540" y="8352" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVLink (4b).</text>
<line x1="530.0" y1="8102.0" x2="530.0" y2="8352.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="8392" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">The scale ladder: from an SM to 16,384 GPUs</text>
<text x="40" y="8422" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">level</text>
<text x="252" y="8422" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPUs</text>
<text x="268" y="8422" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">BF16 compute (log)</text>
<text x="676" y="8422" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">HBM</text>
<text x="690" y="8422" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">link to the next level</text>
<line x1="40.0" y1="8429.0" x2="800.0" y2="8429.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="8440.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="8454.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SM</text>
<text x="252" y="8454" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1/132</text>
<rect x="268" y="8442" width="29.8" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="303.8" y="8454" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">7.5 TFLOP/s</text>
<text x="676" y="8454" text-anchor="end" style="fill:var(--c-text);font-size:11px">512 KiB SRAM</text>
<text x="690" y="8454" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">L2 · HBM (shared)</text>
<rect x="40.0" y="8466.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="8480.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100</text>
<text x="146" y="8480" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 132 SMs</text>
<text x="252" y="8480" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1</text>
<rect x="268" y="8468" width="101.9" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="375.9" y="8480" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">990 TFLOP/s</text>
<text x="676" y="8480" text-anchor="end" style="fill:var(--c-text);font-size:11px">80 GB</text>
<text x="690" y="8480" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink 450 GB/s</text>
<rect x="40.0" y="8492.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="8506.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="146" y="8506" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 8</text>
<text x="252" y="8506" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8</text>
<rect x="268" y="8494" width="132.6" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="406.6" y="8506" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">7.9 PFLOP/s</text>
<text x="676" y="8506" text-anchor="end" style="fill:var(--c-text);font-size:11px">640 GB</text>
<text x="690" y="8506" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">network 400 GB/s</text>
<rect x="40.0" y="8518.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="8532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="146" y="8532" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 32</text>
<text x="252" y="8532" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">256</text>
<rect x="268" y="8520" width="183.7" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="457.7" y="8532" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">253 PFLOP/s</text>
<text x="676" y="8532" text-anchor="end" style="fill:var(--c-text);font-size:11px">20.5 TB</text>
<text x="690" y="8532" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">to spine 12.8 TB/s</text>
<rect x="40.0" y="8544.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="8558.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SuperPOD</text>
<text x="146" y="8558" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 4 − 1</text>
<text x="252" y="8558" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1,016</text>
<rect x="268" y="8546" width="204.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="478.1" y="8558" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">1.0 EFLOP/s</text>
<text x="676" y="8558" text-anchor="end" style="fill:var(--c-text);font-size:11px">81 TB</text>
<text x="690" y="8558" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine 51.2 TB/s</text>
<rect x="40.0" y="8570.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="8584.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">64 SUs</text>
<text x="146" y="8584" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 16</text>
<text x="252" y="8584" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16,384</text>
<rect x="268" y="8572" width="245.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="519.1" y="8584" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16.2 EFLOP/s</text>
<text x="676" y="8584" text-anchor="end" style="fill:var(--c-text);font-size:11px">1.31 PB</text>
<text x="690" y="8584" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">core tier</text>
<text x="40" y="8610" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Compute is the BF16 dense peak (990 TFLOP/s × number of GPUs); real training runs at about 40% MFU (E). 64 SUs is the scale Parts B and</text>
<text x="40" y="8625" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">C think in: 2,048 DGX = 16,384 H100s. Each level multiplies compute N times while the road out narrows per GPU:</text>
<text x="40" y="8640" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">3.35 TB/s → 450 GB/s → 50 GB/s.</text>
<text x="40" y="8680" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Mental model: the restaurant chain</text>
<text x="40" y="8698" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The analogy from the text, continued: if a node is a restaurant with eight kitchens, a SuperPOD is a chain.</text>
<rect x="40.0" y="8710" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="8710" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="8730" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Restaurant chain</text>
<rect x="52.0" y="8738.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="8752.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SuperPOD · 127 DGX</text>
<text x="54.0" y="8776" text-anchor="start" style="fill:var(--c-text);font-size:11px">32 restaurants make a</text>
<text x="54.0" y="8791" text-anchor="start" style="fill:var(--c-text);font-size:11px">neighbourhood (SU), and the</text>
<text x="54.0" y="8806" text-anchor="start" style="fill:var(--c-text);font-size:11px">chain has 4. One slot goes</text>
<text x="54.0" y="8821" text-anchor="start" style="fill:var(--c-text);font-size:11px">to head office (UFM), so</text>
<text x="54.0" y="8836" text-anchor="start" style="fill:var(--c-text);font-size:11px">127 restaurants run, not</text>
<text x="54.0" y="8851" text-anchor="start" style="fill:var(--c-text);font-size:11px">128.</text>
<text x="54.0" y="8899" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">127 DGX · 1,016 GPUs</text>
<rect x="233.0" y="8710" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="8710" width="4" height="201" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="247.0" y="8730" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Local exchange</text>
<rect x="245.0" y="8738.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.5" y="8752.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf · 32 of them</text>
<text x="247.0" y="8776" text-anchor="start" style="fill:var(--c-text);font-size:11px">Each neighbourhood has 8</text>
<text x="247.0" y="8791" text-anchor="start" style="fill:var(--c-text);font-size:11px">exchanges, one per rail:</text>
<text x="247.0" y="8806" text-anchor="start" style="fill:var(--c-text);font-size:11px">kitchen 3 of every</text>
<text x="247.0" y="8821" text-anchor="start" style="fill:var(--c-text);font-size:11px">restaurant is wired to the</text>
<text x="247.0" y="8836" text-anchor="start" style="fill:var(--c-text);font-size:11px">same exchange.</text>
<text x="247.0" y="8899" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">8 per SU · 1 per rail</text>
<rect x="426.0" y="8710" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="8710" width="4" height="201" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="440.0" y="8730" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Trunk line</text>
<rect x="438.0" y="8738.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="516.5" y="8752.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine · 16 of them</text>
<text x="440.0" y="8776" text-anchor="start" style="fill:var(--c-text);font-size:11px">Joins the neighbourhoods;</text>
<text x="440.0" y="8791" text-anchor="start" style="fill:var(--c-text);font-size:11px">every exchange reaches</text>
<text x="440.0" y="8806" text-anchor="start" style="fill:var(--c-text);font-size:11px">every trunk over two</text>
<text x="440.0" y="8821" text-anchor="start" style="fill:var(--c-text);font-size:11px">cables. As much can go up</text>
<text x="440.0" y="8836" text-anchor="start" style="fill:var(--c-text);font-size:11px">as comes in, so it never</text>
<text x="440.0" y="8851" text-anchor="start" style="fill:var(--c-text);font-size:11px">clogs (1:1).</text>
<text x="440.0" y="8899" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">2 × 400 Gb/s · 51.2 TB/s</text>
<rect x="619.0" y="8710" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="8710" width="4" height="201" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="633.0" y="8730" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Depot and head office</text>
<rect x="631.0" y="8738.0" width="157.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="709.5" y="8752.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">storage + management</text>
<text x="633.0" y="8776" text-anchor="start" style="fill:var(--c-text);font-size:11px">Ingredients (data,</text>
<text x="633.0" y="8791" text-anchor="start" style="fill:var(--c-text);font-size:11px">checkpoints) come from the</text>
<text x="633.0" y="8806" text-anchor="start" style="fill:var(--c-text);font-size:11px">central depot on a road of</text>
<text x="633.0" y="8821" text-anchor="start" style="fill:var(--c-text);font-size:11px">their own; head office</text>
<text x="633.0" y="8836" text-anchor="start" style="fill:var(--c-text);font-size:11px">(UFM, Base Command) watches</text>
<text x="633.0" y="8851" text-anchor="start" style="fill:var(--c-text);font-size:11px">the lines and takes a</text>
<text x="633.0" y="8866" text-anchor="start" style="fill:var(--c-text);font-size:11px">failed restaurant out.</text>
<text x="633.0" y="8899" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">≥ 40 GB/s/node · UFM</text>
<rect x="36.0" y="8923.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="8940" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Golden rule: same-numbered kitchens share an exchange, so the most frequent meeting (the DP AllReduce) ends in one hop inside the</text>
<text x="46" y="8955" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">neighbourhood; only cross-rail and cross-neighbourhood traffic takes the trunk line.</text>
<line x1="420.0" y1="8980" x2="420.0" y2="9002" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="8995" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">a closer look at the SuperPOD&#x27;s network</text>
<rect x="28" y="9005" width="784" height="1926" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="9031" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4b · A closer look at the SuperPOD</text>
<text x="40" y="9051" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Let&#x27;s read the same 127 DGX as a network: why doesn&#x27;t it clog, which road does traffic take, how long does an AllReduce</text>
<text x="40" y="9068" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">take, and how far does the fabric grow? This card draws the text&#x27;s “Scaling beyond the node” and “Collective communication:</text>
<text x="40" y="9085" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">From ring algorithms to in-network SHARP” subsections.</text>
<text x="40" y="9115" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Why doesn&#x27;t it clog (1:1)? Count one leaf&#x27;s 64 ports</text>
<text x="40" y="9141" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Leaf i (rail i): 32 ports down, 32 ports up, 2 cables to every spine</text>
<text x="40" y="9168" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16 spines</text>
<line x1="110.5" y1="9173.0" x2="110.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="123.5" y1="9173.0" x2="123.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="136.5" y1="9173.0" x2="136.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="149.5" y1="9173.0" x2="149.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="162.5" y1="9173.0" x2="162.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="175.5" y1="9173.0" x2="175.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="188.5" y1="9173.0" x2="188.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="201.5" y1="9173.0" x2="201.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="214.5" y1="9173.0" x2="214.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="227.5" y1="9173.0" x2="227.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="240.5" y1="9173.0" x2="240.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="253.5" y1="9173.0" x2="253.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="266.5" y1="9173.0" x2="266.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="279.5" y1="9173.0" x2="279.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="292.5" y1="9173.0" x2="292.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="305.5" y1="9173.0" x2="305.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="318.5" y1="9173.0" x2="318.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="331.5" y1="9173.0" x2="331.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="344.5" y1="9173.0" x2="344.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="357.5" y1="9173.0" x2="357.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="370.5" y1="9173.0" x2="370.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="383.5" y1="9173.0" x2="383.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="396.5" y1="9173.0" x2="396.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="409.5" y1="9173.0" x2="409.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="422.5" y1="9173.0" x2="422.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="435.5" y1="9173.0" x2="435.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="448.5" y1="9173.0" x2="448.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="461.5" y1="9173.0" x2="461.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="474.5" y1="9173.0" x2="474.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="487.5" y1="9173.0" x2="487.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="500.5" y1="9173.0" x2="500.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="513.5" y1="9173.0" x2="513.5" y2="9211.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="106.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="117.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="132.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="143.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="158.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="169.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="184.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="195.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="210.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="221.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="236.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="247.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="262.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="273.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<rect x="288.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="299.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<rect x="314.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="325.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">9</text>
<rect x="340.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="351.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">10</text>
<rect x="366.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="377.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">11</text>
<rect x="392.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="403.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">12</text>
<rect x="418.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="429.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">13</text>
<rect x="444.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="455.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">14</text>
<rect x="470.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="481.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">15</text>
<rect x="496.0" y="9155.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="9168.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16</text>
<line x1="110.5" y1="9249.0" x2="110.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="123.5" y1="9249.0" x2="123.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="136.5" y1="9249.0" x2="136.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="149.5" y1="9249.0" x2="149.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="162.5" y1="9249.0" x2="162.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="175.5" y1="9249.0" x2="175.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="188.5" y1="9249.0" x2="188.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="201.5" y1="9249.0" x2="201.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="214.5" y1="9249.0" x2="214.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="227.5" y1="9249.0" x2="227.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="240.5" y1="9249.0" x2="240.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="253.5" y1="9249.0" x2="253.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="266.5" y1="9249.0" x2="266.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="279.5" y1="9249.0" x2="279.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="292.5" y1="9249.0" x2="292.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="305.5" y1="9249.0" x2="305.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="318.5" y1="9249.0" x2="318.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="331.5" y1="9249.0" x2="331.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="344.5" y1="9249.0" x2="344.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="357.5" y1="9249.0" x2="357.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="370.5" y1="9249.0" x2="370.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="383.5" y1="9249.0" x2="383.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="396.5" y1="9249.0" x2="396.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="409.5" y1="9249.0" x2="409.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="422.5" y1="9249.0" x2="422.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="435.5" y1="9249.0" x2="435.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="448.5" y1="9249.0" x2="448.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="461.5" y1="9249.0" x2="461.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="474.5" y1="9249.0" x2="474.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="487.5" y1="9249.0" x2="487.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="500.5" y1="9249.0" x2="500.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="513.5" y1="9249.0" x2="513.5" y2="9277.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="104.0" y="9207.0" width="416.0" height="46" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="106.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="106.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="119.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="119.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="132.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="132.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="145.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="145.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="158.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="158.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="171.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="171.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="184.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="184.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="197.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="197.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="210.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="210.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="223.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="223.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="236.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="236.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="249.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="262.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="275.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="275.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="288.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="288.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="301.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="301.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="314.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="314.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="327.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="327.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="340.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="340.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="353.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="353.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="366.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="366.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="379.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="379.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="392.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="405.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="405.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="418.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="418.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="431.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="431.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="444.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="444.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="457.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="457.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="470.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="470.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="483.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="483.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="496.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="496.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="509.0" y="9211" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="509.0" y="9241" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="312" y="9234" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">QM9700 · 64 × 400 Gb/s</text>
<text x="40" y="9234" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">leaf i</text>
<rect x="105.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="118.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="131.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="144.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="157.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="170.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="183.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="196.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="209.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="222.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="235.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="248.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="261.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="274.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="287.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="300.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="313.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="326.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="339.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="352.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="365.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="378.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="391.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="404.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="417.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="430.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="443.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="456.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="469.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="482.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="495.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="508.0" y="9277.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="9290" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">32 DGX</text>
<text x="104" y="9311" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX 1</text>
<text x="520" y="9311" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">DGX 32</text>
<text x="312" y="9311" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">NIC i of each · 400 Gb/s</text>
<rect x="40.0" y="9325.0" width="480.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="9340.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32 leaves × 32 uplinks = 1,024 cables = 16 spines × 64 ports</text>
<text x="40" y="9369" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Cable counts (NVIDIA&#x27;s table): 1,020 from nodes to leaves (127 × 8 + 4 UFM), 1,024</text>
<text x="40" y="9384" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">from leaves to spines. Smaller builds follow the same rule: 1 SU = 8 leaves +</text>
<text x="40" y="9399" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 spines, 2 SUs = 16 leaves + 8 spines.</text>
<text x="540" y="9141" text-anchor="start" style="fill:var(--c-text);font-size:12px">32 of each leaf&#x27;s 64 ports go down to the</text>
<text x="540" y="9158" text-anchor="start" style="fill:var(--c-text);font-size:12px">nodes and 32 go up to the spines. Whatever</text>
<text x="540" y="9175" text-anchor="start" style="fill:var(--c-text);font-size:12px">traffic enters from below can leave</text>
<text x="540" y="9192" text-anchor="start" style="fill:var(--c-text);font-size:12px">upward, so there is no oversubscription:</text>
<text x="540" y="9209" text-anchor="start" style="fill:var(--c-text);font-size:12px">the ratio is 1:1. Every leaf reaches every</text>
<text x="540" y="9226" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine, so any two GPUs meet in at most</text>
<text x="540" y="9243" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf → spine → leaf.</text>
<rect x="540.0" y="9259.0" width="260.0" height="173" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="9279" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 51.2 TB/s come from?</text>
<text x="552" y="9298" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16 × 64 × 50 GB/s = 51.2 TB/s</text>
<text x="552" y="9314" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1,016 × 50 GB/s = 50.8 TB/s</text>
<text x="552" y="9333" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">400 Gb/s ÷ 8 = 50 GB/s per port, one way.</text>
<text x="552" y="9348" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">What the spine layer can carry is no less</text>
<text x="552" y="9363" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">than what all the GPUs can send. Don&#x27;t mix</text>
<text x="552" y="9378" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">them up: the “51.2 Tb/s” in NVIDIA&#x27;s</text>
<text x="552" y="9393" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">QM9700 manual is one switch, both</text>
<text x="552" y="9408" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">directions, in bits; the 51.2 TB/s here is</text>
<text x="552" y="9423" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 spines, one direction, in bytes.</text>
<line x1="530.0" y1="9127.0" x2="530.0" y2="9432.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="9472" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Which road does traffic take? Four routes, four prices</text>
<text x="40" y="9498" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Same rail, same SU · DP and PP</text>
<rect x="455.6" y="9486.0" width="64.4" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="487.8" y="9499.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">1 switch</text>
<rect x="40.0" y="9510.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9526.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9510.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9526.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="126.0" y1="9522.0" x2="137.0" y2="9522.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="238.0" y="9510.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="9526.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · B</text>
<line x1="225.0" y1="9522.0" x2="236.0" y2="9522.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9564" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Same rail, another SU · DP across SUs</text>
<rect x="443.5" y="9552.0" width="76.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="481.8" y="9565.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">3 switches</text>
<rect x="40.0" y="9576.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9592.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9576.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9592.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="126.0" y1="9588.0" x2="137.0" y2="9588.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="238.0" y="9576.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="9592.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="225.0" y1="9588.0" x2="236.0" y2="9588.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="337.0" y="9576.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="379.0" y="9592.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3 · SU 2</text>
<line x1="324.0" y1="9588.0" x2="335.0" y2="9588.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="436.0" y="9576.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="478.0" y="9592.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · C</text>
<line x1="423.0" y1="9588.0" x2="434.0" y2="9588.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9630" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Different rail, direct · MoE AllToAll</text>
<rect x="443.5" y="9618.0" width="76.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="481.8" y="9631.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">3 switches</text>
<rect x="40.0" y="9642.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9658.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9642.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9658.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="126.0" y1="9654.0" x2="137.0" y2="9654.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="238.0" y="9642.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="9658.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="225.0" y1="9654.0" x2="236.0" y2="9654.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="337.0" y="9642.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="379.0" y="9658.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="324.0" y1="9654.0" x2="335.0" y2="9654.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="436.0" y="9642.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="478.0" y="9658.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<line x1="423.0" y1="9654.0" x2="434.0" y2="9654.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9696" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Different rail, with PXN</text>
<rect x="401.1" y="9684.0" width="118.9" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="460.6" y="9697.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink + 1 switch</text>
<rect x="40.0" y="9708.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9724.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9708.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.24;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9724.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">NVSwitch</text>
<line x1="126.0" y1="9720.0" x2="137.0" y2="9720.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<rect x="238.0" y="9708.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="9724.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · A</text>
<line x1="225.0" y1="9720.0" x2="236.0" y2="9720.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<rect x="337.0" y="9708.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="379.0" y="9724.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="324.0" y1="9720.0" x2="335.0" y2="9720.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="436.0" y="9708.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="478.0" y="9724.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<line x1="423.0" y1="9720.0" x2="434.0" y2="9720.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9754" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">A and B share an SU, C sits in another. Purple arrow: NVLink (inside the chassis);</text>
<text x="40" y="9769" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">grey: InfiniBand (400 Gb/s).</text>
<text x="540" y="9498" text-anchor="start" style="fill:var(--c-text);font-size:12px">According to NVIDIA, traffic within a rail</text>
<text x="540" y="9515" text-anchor="start" style="fill:var(--c-text);font-size:12px">is always one hop from the other 31 nodes</text>
<text x="540" y="9532" text-anchor="start" style="fill:var(--c-text);font-size:12px">of an SU, while traffic between SUs or</text>
<text x="540" y="9549" text-anchor="start" style="fill:var(--c-text);font-size:12px">between rails crosses the spine layer. PXN</text>
<text x="540" y="9566" text-anchor="start" style="fill:var(--c-text);font-size:12px">in NCCL 2.12 first moves the data over</text>
<text x="540" y="9583" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitch to the GPU on the destination&#x27;s</text>
<text x="540" y="9600" text-anchor="start" style="fill:var(--c-text);font-size:12px">rail, then sends it without leaving that</text>
<text x="540" y="9617" text-anchor="start" style="fill:var(--c-text);font-size:12px">rail. The eight GPUs of a node thus gather</text>
<text x="540" y="9634" text-anchor="start" style="fill:var(--c-text);font-size:12px">their messages for the same destination on</text>
<text x="540" y="9651" text-anchor="start" style="fill:var(--c-text);font-size:12px">one GPU and send them together; NVIDIA</text>
<text x="540" y="9668" text-anchor="start" style="fill:var(--c-text);font-size:12px">reports this doubles all2all performance.</text>
<rect x="540.0" y="9684.0" width="260.0" height="119" rx="5" style="fill:var(--c-success);fill-opacity:.08;stroke:var(--c-success);stroke-width:1.2"/>
<text x="552" y="9703" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-success);font-weight:700">A real example:</tspan> DeepSeek-V3 (2,048 H800s)</text>
<text x="552" y="9718" text-anchor="start" style="fill:var(--c-text);font-size:11px">does the mirror image: a token first</text>
<text x="552" y="9733" text-anchor="start" style="fill:var(--c-text);font-size:11px">travels over IB to the GPU with the same</text>
<text x="552" y="9748" text-anchor="start" style="fill:var(--c-text);font-size:11px">in-node index on the target node, on its</text>
<text x="552" y="9763" text-anchor="start" style="fill:var(--c-text);font-size:11px">own rail, and is then forwarded over</text>
<text x="552" y="9778" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink to the expert&#x27;s GPU. Each token</text>
<text x="552" y="9793" text-anchor="start" style="fill:var(--c-text);font-size:11px">goes to at most 4 nodes.</text>
<line x1="530.0" y1="9484.0" x2="530.0" y2="9803.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="9843" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">How long does an AllReduce take on this fabric? Ring vs SHARP</text>
<text x="40" y="9869" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Example: Llama 3 70B, TP 8 × DP 127. Per GPU 70 billion ÷ 8 = 8.75 billion</text>
<text x="40" y="9886" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">parameters, a BF16 gradient of B = 17.5 GB; the DP AllReduce runs among the</text>
<text x="40" y="9903" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">127 GPUs of rail i.</text>
<text x="40" y="9931" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">Ring: chunks travel from GPU to GPU</text>
<line x1="163.6" y1="9953.5" x2="180.4" y2="9958.9" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="193.6" y1="9968.2" x2="204.0" y2="9982.4" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="209.3" y1="9997.7" x2="209.3" y2="10015.3" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="204.6" y1="10030.8" x2="194.2" y2="10045.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="181.3" y1="10054.8" x2="164.6" y2="10060.2" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="148.4" y1="10060.5" x2="131.6" y2="10055.1" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="118.4" y1="10045.8" x2="108.0" y2="10031.6" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="102.7" y1="10016.3" x2="102.7" y2="9998.7" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="107.4" y1="9983.2" x2="118.4" y2="9968.2" style="stroke:var(--c-accent);stroke-width:1.4;stroke-dasharray:4 3;stroke-opacity:.7"/>
<line x1="130.7" y1="9959.2" x2="147.4" y2="9953.8" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<circle cx="156.0" cy="9951.0" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="188.9" cy="9961.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="209.3" cy="9989.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="209.3" cy="10024.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="188.9" cy="10052.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="156.0" cy="10063.0" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="123.1" cy="10052.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="102.7" cy="10024.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="102.7" cy="9989.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="123.1" cy="9961.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="156" y="10005" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">rail i</text>
<text x="156" y="10020" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">127 GPUs</text>
<text x="40" y="10091" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 × 126 steps; each step passes a chunk</text>
<text x="40" y="10106" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">of B/127 ≈ 138 MB to the neighbour.</text>
<text x="288" y="9931" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">SHARP: the switch does the sum</text>
<line x1="318.0" y1="9991.0" x2="383.0" y2="9969.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="302.0" y1="10047.0" x2="311.6" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="318.0" y1="10047.0" x2="318.0" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="334.0" y1="10047.0" x2="324.4" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="378.0" y1="9991.0" x2="397.0" y2="9969.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="362.0" y1="10047.0" x2="371.6" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="378.0" y1="10047.0" x2="378.0" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="394.0" y1="10047.0" x2="384.4" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="438.0" y1="9991.0" x2="411.0" y2="9969.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="422.0" y1="10047.0" x2="431.6" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="438.0" y1="10047.0" x2="438.0" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="454.0" y1="10047.0" x2="444.4" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="498.0" y1="9991.0" x2="425.0" y2="9969.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="482.0" y1="10047.0" x2="491.6" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="498.0" y1="10047.0" x2="498.0" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="514.0" y1="10047.0" x2="504.4" y2="10021.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="364.0" y="9947.0" width="80.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="404.0" y="9961.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Σ spine</text>
<rect x="292.0" y="9995.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="318.0" y="10010.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 1</text>
<circle cx="302" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="318" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="334" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="352.0" y="9995.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="378.0" y="10010.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 2</text>
<circle cx="362" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="378" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="394" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="412.0" y="9995.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="438.0" y="10010.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 3</text>
<circle cx="422" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="438" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="454" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="472.0" y="9995.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="498.0" y="10010.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 4</text>
<circle cx="482" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="498" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="514" cy="10053" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="520" y="10075" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">↓ the sum to everyone</text>
<text x="288" y="10091" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Every GPU sends B up once; leaves add up</text>
<text x="288" y="10106" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">their SU, the spine adds up the four</text>
<text x="288" y="10121" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SUs, and the result is copied back down</text>
<text x="288" y="10136" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">the same way.</text>
<line x1="280.0" y1="9919.0" x2="280.0" y2="10136.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="10154.0" width="480.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="160" y="10169" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">T ring ≈ 2 · (N−1)/N · B/W</text>
<text x="400" y="10169" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">T SHARP ≈ B/W</text>
<line x1="280.0" y1="10158.0" x2="280.0" y2="10172.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10202" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Data each GPU puts on the wire, and time (W = 50 GB/s)</text>
<text x="40" y="10226" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Ring</text>
<rect x="104" y="10214" width="260.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="370.0" y="10226" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">34.7 GB · 0.69 s</text>
<text x="40" y="10250" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">SHARP</text>
<rect x="104" y="10238" width="131.0" height="16" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="241.0" y="10250" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">17.5 GB · 0.35 s</text>
<text x="540" y="9869" text-anchor="start" style="fill:var(--c-text);font-size:12px">The text&#x27;s formulas: a ring AllReduce does</text>
<text x="540" y="9886" text-anchor="start" style="fill:var(--c-text);font-size:12px">a ReduceScatter, then an AllGather; every</text>
<text x="540" y="9903" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU puts 2 · (N−1)/N · B bytes on the</text>
<text x="540" y="9920" text-anchor="start" style="fill:var(--c-text);font-size:12px">wire, so the time is ≈ 2B/W. With SHARP</text>
<text x="540" y="9937" text-anchor="start" style="fill:var(--c-text);font-size:12px">the Quantum-2 switch ASICs do the sum: a</text>
<text x="540" y="9954" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU sends its gradient up once and gets</text>
<text x="540" y="9971" text-anchor="start" style="fill:var(--c-text);font-size:12px">the total back once, so traffic halves and</text>
<text x="540" y="9988" text-anchor="start" style="fill:var(--c-text);font-size:12px">the time drops to ≈ B/W. The rail layout</text>
<text x="540" y="10005" text-anchor="start" style="fill:var(--c-text);font-size:12px">fits this tree naturally: the GPU i&#x27;s of</text>
<text x="540" y="10022" text-anchor="start" style="fill:var(--c-text);font-size:12px">an SU add up in leaf i, and the four SUs&#x27;</text>
<text x="540" y="10039" text-anchor="start" style="fill:var(--c-text);font-size:12px">partial sums meet in a spine.</text>
<text x="540" y="10061" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">In real training, gradients are split into</text>
<text x="540" y="10076" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">buckets and overlapped with the backward pass,</text>
<text x="540" y="10091" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">so these times don&#x27;t add straight onto the</text>
<text x="540" y="10106" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">step. Meta&#x27;s 405B cluster used RoCE, so the</text>
<text x="540" y="10121" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">SHARP step belongs to the SuperPOD scenario</text>
<text x="540" y="10136" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(text, section 5).</text>
<line x1="530.0" y1="9855.0" x2="530.0" y2="10254.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10294" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">How far does the fabric grow? A third tier</text>
<text x="40" y="10324" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">SUs</text>
<text x="132" y="10324" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">DGX</text>
<text x="200" y="10324" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPUs</text>
<text x="256" y="10324" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">leaf</text>
<text x="312" y="10324" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">spine</text>
<text x="364" y="10324" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">core</text>
<text x="424" y="10324" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">tiers</text>
<text x="520" y="10324" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">longest path</text>
<line x1="40.0" y1="10331.0" x2="520.0" y2="10331.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="10352" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="132" y="10352" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">31</text>
<text x="200" y="10352" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">248</text>
<text x="256" y="10352" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="312" y="10352" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="364" y="10352" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">–</text>
<text x="424" y="10352" text-anchor="end" style="fill:var(--c-text);font-size:11px">2</text>
<text x="520" y="10352" text-anchor="end" style="fill:var(--c-text);font-size:11px">3 switches</text>
<rect x="36.0" y="10360.0" width="488.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="10374" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="132" y="10374" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">127</text>
<text x="200" y="10374" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1,016</text>
<text x="256" y="10374" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32</text>
<text x="312" y="10374" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="364" y="10374" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">–</text>
<text x="424" y="10374" text-anchor="end" style="fill:var(--c-text);font-size:11px">2</text>
<text x="520" y="10374" text-anchor="end" style="fill:var(--c-text);font-size:11px">3 switches</text>
<text x="40" y="10396" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="132" y="10396" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="200" y="10396" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">2,048</text>
<text x="256" y="10396" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="312" y="10396" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32</text>
<text x="364" y="10396" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">–</text>
<text x="424" y="10396" text-anchor="end" style="fill:var(--c-text);font-size:11px">2</text>
<text x="520" y="10396" text-anchor="end" style="fill:var(--c-text);font-size:11px">3 switches</text>
<line x1="40.0" y1="10406.0" x2="520.0" y2="10406.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="40" y="10420" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">the two-tier ceiling: 2,048 GPUs</text>
<text x="40" y="10440" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16</text>
<text x="132" y="10440" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">512</text>
<text x="200" y="10440" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">4,096</text>
<text x="256" y="10440" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">128</text>
<text x="312" y="10440" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">128</text>
<text x="364" y="10440" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="424" y="10440" text-anchor="end" style="fill:var(--c-text);font-size:11px">3</text>
<text x="520" y="10440" text-anchor="end" style="fill:var(--c-text);font-size:11px">5 switches</text>
<text x="40" y="10462" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">32</text>
<text x="132" y="10462" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1,024</text>
<text x="200" y="10462" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8,192</text>
<text x="256" y="10462" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="312" y="10462" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="364" y="10462" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-family:var(--font-mono)">128</text>
<text x="424" y="10462" text-anchor="end" style="fill:var(--c-text);font-size:11px">3</text>
<text x="520" y="10462" text-anchor="end" style="fill:var(--c-text);font-size:11px">5 switches</text>
<text x="40" y="10484" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">64</text>
<text x="132" y="10484" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2,048</text>
<text x="200" y="10484" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16,384</text>
<text x="256" y="10484" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">512</text>
<text x="312" y="10484" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">512</text>
<text x="364" y="10484" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="424" y="10484" text-anchor="end" style="fill:var(--c-text);font-size:11px">3</text>
<text x="520" y="10484" text-anchor="end" style="fill:var(--c-text);font-size:11px">5 switches</text>
<line x1="40.0" y1="10494.0" x2="520.0" y2="10494.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="10514" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The 1 and 4 SU rows come from NVIDIA&#x27;s SuperPOD table (one slot goes to UFM); 8 SUs</text>
<text x="40" y="10529" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">and up from its larger-build table, which counts 32 DGX per SU. That table labels the</text>
<text x="40" y="10544" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">last row “56 SU”; 2,048 ÷ 32 = 64.</text>
<rect x="540.0" y="10308.0" width="260.0" height="189" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="10328" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 2,048 come from?</text>
<text x="552" y="10347" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">64 ports = 32 down + 32 up</text>
<text x="552" y="10363" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2 tiers: 64 leaves × 32 = 2,048</text>
<text x="552" y="10379" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3 tiers: 64 × 32 × 32 = 65,536</text>
<text x="552" y="10398" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">A spine&#x27;s 64 ports reach at most</text>
<text x="552" y="10413" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 leaves, and each leaf carries 32 GPUs.</text>
<text x="552" y="10428" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 SUs (64 leaves) hit that ceiling</text>
<text x="552" y="10443" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">exactly; at 16 SUs a core tier goes above</text>
<text x="552" y="10458" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">the spines, the longest path grows to 5</text>
<text x="552" y="10473" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">switches (leaf, spine, core, spine, leaf),</text>
<text x="552" y="10488" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">and each GPU needs one more cable.</text>
<text x="540" y="10519" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">For comparison: Meta&#x27;s Llama 3 cluster (RoCE,</text>
<text x="540" y="10534" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">24K GPUs) also has three tiers but is not</text>
<text x="540" y="10549" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">non-blocking end to end: 3,072-GPU pods are</text>
<text x="540" y="10564" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">1:1 inside, 1:7 between pods.</text>
<line x1="530.0" y1="10306.0" x2="530.0" y2="10564.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10604" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Mental model: how the chain talks</text>
<text x="40" y="10622" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The restaurant-chain analogy, continued: this time we look at who talks to whom, on which line, and how.</text>
<rect x="40.0" y="10634" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="10634" width="4" height="216" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="10654" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Same-numbered kitchens</text>
<rect x="52.0" y="10662.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="10676.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">rail · leaf i</text>
<text x="54.0" y="10700" text-anchor="start" style="fill:var(--c-text);font-size:11px">Kitchen 3 of every</text>
<text x="54.0" y="10715" text-anchor="start" style="fill:var(--c-text);font-size:11px">restaurant is wired to the</text>
<text x="54.0" y="10730" text-anchor="start" style="fill:var(--c-text);font-size:11px">same exchange. They all</text>
<text x="54.0" y="10745" text-anchor="start" style="fill:var(--c-text);font-size:11px">cook the same recipe (the</text>
<text x="54.0" y="10760" text-anchor="start" style="fill:var(--c-text);font-size:11px">same slice of the model),</text>
<text x="54.0" y="10775" text-anchor="start" style="fill:var(--c-text);font-size:11px">so they talk most, and</text>
<text x="54.0" y="10790" text-anchor="start" style="fill:var(--c-text);font-size:11px">their calls never leave the</text>
<text x="54.0" y="10805" text-anchor="start" style="fill:var(--c-text);font-size:11px">neighbourhood.</text>
<text x="54.0" y="10838" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">1 switch · DP, PP</text>
<rect x="233.0" y="10634" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="10634" width="4" height="216" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="247.0" y="10654" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">The inner corridor</text>
<rect x="245.0" y="10662.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.5" y="10676.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PXN · NVSwitch</text>
<text x="247.0" y="10700" text-anchor="start" style="fill:var(--c-text);font-size:11px">A note for kitchen 5 first</text>
<text x="247.0" y="10715" text-anchor="start" style="fill:var(--c-text);font-size:11px">walks down the restaurant&#x27;s</text>
<text x="247.0" y="10730" text-anchor="start" style="fill:var(--c-text);font-size:11px">own corridor to its kitchen</text>
<text x="247.0" y="10745" text-anchor="start" style="fill:var(--c-text);font-size:11px">5, then goes to the right</text>
<text x="247.0" y="10760" text-anchor="start" style="fill:var(--c-text);font-size:11px">exchange; it never takes</text>
<text x="247.0" y="10775" text-anchor="start" style="fill:var(--c-text);font-size:11px">the trunk line.</text>
<text x="247.0" y="10838" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">NVLink + 1 switch</text>
<rect x="426.0" y="10634" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="10634" width="4" height="216" rx="2" style="fill:var(--c-success);fill-opacity:.85"/>
<text x="440.0" y="10654" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">The vote count</text>
<rect x="438.0" y="10662.0" width="157.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="516.5" y="10676.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SHARP · switch ASIC</text>
<text x="440.0" y="10700" text-anchor="start" style="fill:var(--c-text);font-size:11px">Instead of passing votes</text>
<text x="440.0" y="10715" text-anchor="start" style="fill:var(--c-text);font-size:11px">(gradients) hand to hand,</text>
<text x="440.0" y="10730" text-anchor="start" style="fill:var(--c-text);font-size:11px">everyone hands theirs to</text>
<text x="440.0" y="10745" text-anchor="start" style="fill:var(--c-text);font-size:11px">the exchange once; it</text>
<text x="440.0" y="10760" text-anchor="start" style="fill:var(--c-text);font-size:11px">counts and sends the result</text>
<text x="440.0" y="10775" text-anchor="start" style="fill:var(--c-text);font-size:11px">back once. Half the paper</text>
<text x="440.0" y="10790" text-anchor="start" style="fill:var(--c-text);font-size:11px">travels.</text>
<text x="440.0" y="10838" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">B instead of 2B</text>
<rect x="619.0" y="10634" width="181.0" height="216" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="10634" width="4" height="216" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="633.0" y="10654" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">National backbone</text>
<rect x="631.0" y="10662.0" width="157.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="709.5" y="10676.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">core · 3rd tier</text>
<text x="633.0" y="10700" text-anchor="start" style="fill:var(--c-text);font-size:11px">Once the chain outgrows 8</text>
<text x="633.0" y="10715" text-anchor="start" style="fill:var(--c-text);font-size:11px">neighbourhoods, a backbone</text>
<text x="633.0" y="10730" text-anchor="start" style="fill:var(--c-text);font-size:11px">goes above the trunk lines.</text>
<text x="633.0" y="10745" text-anchor="start" style="fill:var(--c-text);font-size:11px">The farthest order now</text>
<text x="633.0" y="10760" text-anchor="start" style="fill:var(--c-text);font-size:11px">passes 5 exchanges, not 3.</text>
<text x="633.0" y="10838" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">beyond 2,048 GPUs</text>
<rect x="36.0" y="10862.0" width="768.0" height="59" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="10879" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Golden rule: put traffic on the widest road. The chassis first (NVLink, 450 GB/s), then the rail (one leaf, 50 GB/s), the spine and</text>
<text x="46" y="10894" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">core last. TP stays in the chassis, DP and PP flow along the rails, PXN turns MoE&#x27;s AllToAll onto a rail, and SHARP sums the</text>
<text x="46" y="10909" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">AllReduce in the switch.</text>
<line x1="420.0" y1="10934" x2="420.0" y2="10956" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="10949" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">the building blocks that keep it running</text>
<rect x="28" y="10959" width="784" height="2291" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10985" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">SuperPOD&#x27;s building blocks: why is each one there?</text>
<text x="40" y="11005" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">NVIDIA&#x27;s 4-SU reference design (127 DGX); counts, power and roles come from NVIDIA&#x27;s documentation. The stripe colour shows</text>
<text x="40" y="11022" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">which pillar a part serves; grey is management and infrastructure. The card ties the text&#x27;s storage subsections in section 4</text>
<text x="40" y="11039" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">and its “Cluster MTBF” subsection to the SuperPOD&#x27;s parts.</text>
<text x="40" y="11069" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Four cables per DGX: why four separate networks?</text>
<rect x="40" y="11085" width="132" height="218" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="106" y="11105" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">DGX H100</text>
<rect x="48.0" y="11117.0" width="116.0" height="34" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="106.0" y="11131" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8 × CX-7</text>
<text x="106.0" y="11145" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × 400 Gb/s IB</text>
<line x1="166.0" y1="11134.0" x2="300.0" y2="11134.0" style="stroke:var(--c-accent-2);stroke-width:4"/>
<text x="214" y="11127" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">400 GB/s</text>
<rect x="300.0" y="11117.0" width="220.0" height="34" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="410.0" y="11131" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Compute fabric</text>
<text x="410.0" y="11145" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">48 × QM9700 · 1:1</text>
<rect x="48.0" y="11163.0" width="116.0" height="34" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="106.0" y="11177" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2 CX-7 ports</text>
<text x="106.0" y="11191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2 × 400 Gb/s IB</text>
<line x1="166.0" y1="11180.0" x2="300.0" y2="11180.0" style="stroke:var(--c-warn);stroke-width:3"/>
<text x="214" y="11173" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">100 GB/s</text>
<rect x="300.0" y="11163.0" width="220.0" height="34" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="410.0" y="11177" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Storage fabric</text>
<text x="410.0" y="11191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 × QM9700 · ~4:3</text>
<rect x="48.0" y="11209.0" width="116.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="106.0" y="11223" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2 CX-7 ports</text>
<text x="106.0" y="11237" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2 × 100 GbE</text>
<line x1="166.0" y1="11226.0" x2="300.0" y2="11226.0" style="stroke:var(--c-text-mute);stroke-width:2"/>
<text x="214" y="11219" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">25 GB/s</text>
<rect x="300.0" y="11209.0" width="220.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="410.0" y="11223" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">In-band management</text>
<text x="410.0" y="11237" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × SN4600C · Ethernet</text>
<rect x="48.0" y="11255.0" width="116.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="106.0" y="11269" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">BMC</text>
<text x="106.0" y="11283" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 × 1 GbE copper</text>
<line x1="166.0" y1="11272.0" x2="300.0" y2="11272.0" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="214" y="11265" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0.125 GB/s</text>
<rect x="300.0" y="11255.0" width="220.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="410.0" y="11269" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Out-of-band mgmt</text>
<text x="410.0" y="11283" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × SN2201 · Ethernet</text>
<text x="40" y="11325" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Cables (NVIDIA&#x27;s table): compute 2,040 NDR (1,016 node + 1,024 leaf–spine), storage 2</text>
<text x="40" y="11340" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per node (254), in-band 254 × 100 GbE, OOB 127 × 1 GbE. The storage links carry</text>
<text x="40" y="11355" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">100 GB/s, but NVIDIA&#x27;s target is ≥ 40 GB/s per node.</text>
<text x="540" y="11095" text-anchor="start" style="fill:var(--c-text);font-size:12px">The compute fabric carries only GPU-to-GPU</text>
<text x="540" y="11112" text-anchor="start" style="fill:var(--c-text);font-size:12px">traffic: steps 4, 5 and 6 of part B</text>
<text x="540" y="11129" text-anchor="start" style="fill:var(--c-text);font-size:12px">(activations, the backward pass, the</text>
<text x="540" y="11146" text-anchor="start" style="fill:var(--c-text);font-size:12px">gradient AllReduce). If a checkpoint write</text>
<text x="540" y="11163" text-anchor="start" style="fill:var(--c-text);font-size:12px">or a container pull shared those cables,</text>
<text x="540" y="11180" text-anchor="start" style="fill:var(--c-text);font-size:12px">it would slow the AllReduce. So the</text>
<text x="540" y="11197" text-anchor="start" style="fill:var(--c-text);font-size:12px">storage fabric handles data reads and</text>
<text x="540" y="11214" text-anchor="start" style="fill:var(--c-text);font-size:12px">checkpoint writes (steps 1 and 8), the</text>
<text x="540" y="11231" text-anchor="start" style="fill:var(--c-text);font-size:12px">in-band network the OS, user logins and</text>
<text x="540" y="11248" text-anchor="start" style="fill:var(--c-text);font-size:12px">container pulls, and the out-of-band</text>
<text x="540" y="11265" text-anchor="start" style="fill:var(--c-text);font-size:12px">network power and console access through</text>
<text x="540" y="11282" text-anchor="start" style="fill:var(--c-text);font-size:12px">the BMC, each on its own wires. The four</text>
<text x="540" y="11299" text-anchor="start" style="fill:var(--c-text);font-size:12px">links carry 400, 100, 25 and 0.125 GB/s;</text>
<text x="540" y="11316" text-anchor="start" style="fill:var(--c-text);font-size:12px">each is sized to its own traffic.</text>
<line x1="530.0" y1="11081.0" x2="530.0" y2="11355.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="11395" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Hardware: what, how many, and why?</text>
<rect x="36" y="11409" width="248" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="11409" width="4" height="158" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="48" y="11428" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">DGX H100 system</text>
<rect x="237.8" y="11415.0" width="38.2" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="256.9" y="11428.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×127</text>
<text x="48" y="11445" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 H100 · 640 GB HBM3 · 10.2 kW</text>
<text x="48" y="11463" text-anchor="start" style="fill:var(--c-text);font-size:11px">All of the model&#x27;s maths happens here;</text>
<text x="48" y="11478" text-anchor="start" style="fill:var(--c-text);font-size:11px">every other part of the cluster exists</text>
<text x="48" y="11493" text-anchor="start" style="fill:var(--c-text);font-size:11px">to feed these boxes.</text>
<line x1="48.0" y1="11541.0" x2="272.0" y2="11541.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="11557" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">4 SUs × 32 slots − 1 (UFM) = 127</text>
<rect x="296" y="11409" width="248" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="11409" width="4" height="158" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="308" y="11428" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Compute fabric · QM9700</text>
<rect x="503.9" y="11415.0" width="32.2" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="519.9" y="11428.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×48</text>
<text x="308" y="11445" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 × 400 Gb/s ports · rail-optimised</text>
<text x="308" y="11459" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">fat-tree</text>
<text x="308" y="11477" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gradient and activation traffic must</text>
<text x="308" y="11492" text-anchor="start" style="fill:var(--c-text);font-size:11px">cross nodes without jamming, so every</text>
<text x="308" y="11507" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU gets its own 400 Gb/s rail (cards 4</text>
<text x="308" y="11522" text-anchor="start" style="fill:var(--c-text);font-size:11px">and 4b).</text>
<line x1="308.0" y1="11541.0" x2="532.0" y2="11541.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="308" y="11557" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">32 leaves (4 SUs × 8 rails) + 16 spines</text>
<rect x="556" y="11409" width="248" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="11409" width="4" height="158" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="568" y="11428" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Storage fabric · QM9700</text>
<rect x="763.9" y="11415.0" width="32.2" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="779.9" y="11428.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×16</text>
<text x="568" y="11445" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">≥ 40 GB/s per node · ~4:3</text>
<text x="568" y="11463" text-anchor="start" style="fill:var(--c-text);font-size:11px">Data reads and checkpoint writes stay</text>
<text x="568" y="11478" text-anchor="start" style="fill:var(--c-text);font-size:11px">off the compute fabric&#x27;s AllReduce</text>
<text x="568" y="11493" text-anchor="start" style="fill:var(--c-text);font-size:11px">traffic; its speed is sized to the</text>
<text x="568" y="11508" text-anchor="start" style="fill:var(--c-text);font-size:11px">storage array&#x27;s peak.</text>
<line x1="568.0" y1="11541.0" x2="792.0" y2="11541.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="568" y="11557" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">2 × 400 Gb/s per node</text>
<rect x="36" y="11579" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="11579" width="4" height="144" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="48" y="11598" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">UFM appliance</text>
<rect x="249.9" y="11585.0" width="26.1" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="262.9" y="11598.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×4</text>
<text x="48" y="11615" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">UFM 3.1 · InfiniBand telemetry</text>
<text x="48" y="11633" text-anchor="start" style="fill:var(--c-text);font-size:11px">Watches thousands of ports and cables</text>
<text x="48" y="11648" text-anchor="start" style="fill:var(--c-text);font-size:11px">and finds the bad link; in Llama 3&#x27;s</text>
<text x="48" y="11663" text-anchor="start" style="fill:var(--c-text);font-size:11px">run, 35 of 419 unexpected interruptions</text>
<text x="48" y="11678" text-anchor="start" style="fill:var(--c-text);font-size:11px">(8.4%) came from switches and cables.</text>
<line x1="48.0" y1="11697.0" x2="272.0" y2="11697.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="11713" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">a pair takes the 128th DGX slot</text>
<rect x="296" y="11579" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="11579" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="11598" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">In-band mgmt · SN4600C</text>
<rect x="509.9" y="11585.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="522.9" y="11598.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="308" y="11615" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 ports × 100 GbE (Ethernet)</text>
<text x="308" y="11633" text-anchor="start" style="fill:var(--c-text);font-size:11px">Provisioning, user logins, the file</text>
<text x="308" y="11648" text-anchor="start" style="fill:var(--c-text);font-size:11px">system and outside access (NGC, package</text>
<text x="308" y="11663" text-anchor="start" style="fill:var(--c-text);font-size:11px">repos) run here, leaving the compute</text>
<text x="308" y="11678" text-anchor="start" style="fill:var(--c-text);font-size:11px">fabric to GPU traffic.</text>
<line x1="308.0" y1="11697.0" x2="532.0" y2="11697.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="308" y="11713" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">2 × 100 GbE per node</text>
<rect x="556" y="11579" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="11579" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="568" y="11598" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Out-of-band mgmt · SN2201</text>
<rect x="769.9" y="11585.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="782.9" y="11598.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="568" y="11615" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">48 ports × 1 GbE copper</text>
<text x="568" y="11633" text-anchor="start" style="fill:var(--c-text);font-size:11px">Reaches each server&#x27;s BMC chip and the</text>
<text x="568" y="11648" text-anchor="start" style="fill:var(--c-text);font-size:11px">PDUs, so a node can be powered off and</text>
<text x="568" y="11663" text-anchor="start" style="fill:var(--c-text);font-size:11px">on remotely even when its OS hangs.</text>
<line x1="568.0" y1="11697.0" x2="792.0" y2="11697.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="568" y="11713" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">1 × 1 GbE per node</text>
<rect x="36" y="11735" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="11735" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="11754" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Management servers</text>
<rect x="249.9" y="11741.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="262.9" y="11754.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×5</text>
<text x="48" y="11771" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 sockets · 24+ cores · 384 GB RAM</text>
<text x="48" y="11789" text-anchor="start" style="fill:var(--c-text);font-size:11px">Cluster management (Base Command</text>
<text x="48" y="11804" text-anchor="start" style="fill:var(--c-text);font-size:11px">Manager) and scheduling (Slurm) run on</text>
<text x="48" y="11819" text-anchor="start" style="fill:var(--c-text);font-size:11px">separate machines, so the GPU nodes are</text>
<text x="48" y="11834" text-anchor="start" style="fill:var(--c-text);font-size:11px">left for training alone.</text>
<line x1="48.0" y1="11853.0" x2="272.0" y2="11853.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="11869" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">4 × HDR200 · 7.68 TB NVMe</text>
<rect x="296" y="11735" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="11735" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="11754" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Racks and power</text>
<rect x="473.6" y="11741.0" width="62.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="504.8" y="11754.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">38 racks</text>
<text x="308" y="11771" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">108 PDUs (96 + 12) · 4 DGX per rack</text>
<text x="308" y="11789" text-anchor="start" style="fill:var(--c-text);font-size:11px">4 DGX per rack = 40.8 kW; an SU&#x27;s 8 DGX</text>
<text x="308" y="11804" text-anchor="start" style="fill:var(--c-text);font-size:11px">racks draw 326.4 kW. Power and cooling</text>
<text x="308" y="11819" text-anchor="start" style="fill:var(--c-text);font-size:11px">are the cluster&#x27;s hidden fourth pillar.</text>
<line x1="308.0" y1="11853.0" x2="532.0" y2="11853.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="308" y="11869" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">≈ 1.39 MW in total (below)</text>
<text x="40" y="11919" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Power: where does 1.39 MW come from?</text>
<rect x="40.0" y="11937" width="446.1" height="22" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:.8"/>
<rect x="486.1" y="11937" width="22.7" height="22" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:.8"/>
<rect x="508.8" y="11937" width="7.6" height="22" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:.8"/>
<rect x="516.4" y="11937" width="1.3" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="517.7" y="11937" width="1.2" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="518.9" y="11937" width="0.8" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="519.7" y="11937" width="0.3" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<text x="50" y="11952" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100 · 92.9%</text>
<text x="40" y="11983" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">part</text>
<text x="300" y="11983" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">count × power</text>
<text x="400" y="11983" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">total</text>
<text x="460" y="11983" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">share</text>
<line x1="40.0" y1="11989.0" x2="520.0" y2="11989.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40" y="11996" width="10" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="56" y="12005" text-anchor="start" style="fill:var(--c-text);font-size:11px">DGX H100</text>
<text x="300" y="12005" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">127 × 10,200 W</text>
<text x="400" y="12005" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1,295.4 kW</text>
<text x="460" y="12005" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">92.9%</text>
<rect x="40" y="12014" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="56" y="12023" text-anchor="start" style="fill:var(--c-text);font-size:11px">QM9700 · compute</text>
<text x="300" y="12023" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">48 × 1,376 W</text>
<text x="400" y="12023" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">66.0 kW</text>
<text x="460" y="12023" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4.7%</text>
<rect x="40" y="12032" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.55;stroke:var(--c-warn);stroke-width:1"/>
<text x="56" y="12041" text-anchor="start" style="fill:var(--c-text);font-size:11px">QM9700 · storage</text>
<text x="300" y="12041" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16 × 1,376 W</text>
<text x="400" y="12041" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">22.0 kW</text>
<text x="460" y="12041" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.6%</text>
<rect x="40" y="12050" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12059" text-anchor="start" style="fill:var(--c-text);font-size:11px">SN4600C · in-band</text>
<text x="300" y="12059" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × 466 W</text>
<text x="400" y="12059" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3.7 kW</text>
<text x="460" y="12059" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0.3%</text>
<rect x="40" y="12068" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12077" text-anchor="start" style="fill:var(--c-text);font-size:11px">Management server</text>
<text x="300" y="12077" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">5 × 704 W</text>
<text x="400" y="12077" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3.5 kW</text>
<text x="460" y="12077" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0.3%</text>
<rect x="40" y="12086" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12095" text-anchor="start" style="fill:var(--c-text);font-size:11px">UFM 3.1</text>
<text x="300" y="12095" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4 × 600 W</text>
<text x="400" y="12095" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">2.4 kW</text>
<text x="460" y="12095" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0.2%</text>
<rect x="40" y="12104" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12113" text-anchor="start" style="fill:var(--c-text);font-size:11px">SN2201 · OOB</text>
<text x="300" y="12113" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × 98 W</text>
<text x="400" y="12113" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0.8 kW</text>
<text x="460" y="12113" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0.1%</text>
<line x1="40.0" y1="12121.0" x2="520.0" y2="12121.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="56" y="12135" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Total (average)</text>
<text x="400" y="12135" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1,393.9 kW</text>
<text x="40" y="12159" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Average (EAP) values from NVIDIA&#x27;s planning table; the table counts IT equipment</text>
<text x="40" y="12174" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">only, and cooling comes on top.</text>
<rect x="540.0" y="11933.0" width="260.0" height="174" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="11953" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 1.39 MW come from?</text>
<text x="552" y="11972" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">127 × 10.2 kW = 1.295 MW</text>
<text x="552" y="11988" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">+ network and mgmt = 98.5 kW</text>
<text x="552" y="12004" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">≈ 1.39 MW (peak 1.42 MW)</text>
<text x="552" y="12023" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX systems run at or near peak all the</text>
<text x="552" y="12038" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time under AI load, so NVIDIA gives them</text>
<text x="552" y="12053" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">the same average and peak (10.2 kW). 4 DGX</text>
<text x="552" y="12068" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per rack = 40.8 kW; an SU&#x27;s 8 DGX racks</text>
<text x="552" y="12083" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">come to 326.4 kW. 93 of every 100 W go to</text>
<text x="552" y="12098" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">the GPU boxes.</text>
<line x1="530.0" y1="11931.0" x2="530.0" y2="12174.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="12214" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">What does the storage fabric carry? Checkpoints, not tokens</text>
<text x="40" y="12240" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Reads: the token stream vs NVIDIA&#x27;s targets (4 SUs, log scale)</text>
<text x="40" y="12264" text-anchor="start" style="fill:var(--c-text);font-size:11px">token stream (70B)</text>
<rect x="168" y="12252" width="29.2" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="203.2" y="12264" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3.8 MB/s</text>
<text x="40" y="12288" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVIDIA “Good” · NLP</text>
<rect x="168" y="12276" width="238.9" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="412.9" y="12288" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">60 GB/s</text>
<text x="40" y="12312" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVIDIA “Best”</text>
<rect x="168" y="12300" width="284.9" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="458.9" y="12312" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">500 GB/s</text>
<line x1="168.0" y1="12328.0" x2="468.0" y2="12328.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="168.0" y1="12328.0" x2="168.0" y2="12332.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="168" y="12344" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1 MB/s</text>
<line x1="268.0" y1="12328.0" x2="268.0" y2="12332.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="268" y="12344" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100 MB/s</text>
<line x1="368.0" y1="12328.0" x2="368.0" y2="12332.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="368" y="12344" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10 GB/s</text>
<line x1="468.0" y1="12328.0" x2="468.0" y2="12332.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="468" y="12344" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1 TB/s</text>
<text x="40" y="12372" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Writes: how many seconds does a 1.12 TB checkpoint stop training?</text>
<text x="40" y="12396" text-anchor="start" style="fill:var(--c-text);font-size:11px">“Good” · 30 GB/s</text>
<rect x="168" y="12384" width="250.0" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="424.0" y="12396" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">37 s</text>
<text x="40" y="12420" text-anchor="start" style="fill:var(--c-text);font-size:11px">“Better” · 80 GB/s</text>
<rect x="168" y="12408" width="93.8" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="267.8" y="12420" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">14 s</text>
<text x="40" y="12444" text-anchor="start" style="fill:var(--c-text);font-size:11px">“Best” · 250 GB/s</text>
<rect x="168" y="12432" width="30.0" height="16" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="204.0" y="12444" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">4.5 s</text>
<text x="40" y="12470" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The read and write targets are NVIDIA&#x27;s “Good / Better / Best” guidelines for 4 SUs.</text>
<text x="40" y="12485" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">NLP sits in the lowest class, because a text dataset mostly fits in the node&#x27;s local</text>
<text x="40" y="12500" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">NVMe cache.</text>
<rect x="540.0" y="12228.0" width="260.0" height="144" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="12248" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 3.8 MB/s come from?</text>
<text x="552" y="12267" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1,016 × 990 TFLOP/s × 0.40</text>
<text x="552" y="12283" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">÷ (6 × 70 billion) ≈ 958k/s</text>
<text x="552" y="12299" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">× 4 bytes ≈ 3.8 MB/s</text>
<text x="552" y="12318" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">The text&#x27;s 6·N FLOPs per token, part E&#x27;s</text>
<text x="552" y="12333" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">~40% MFU and int32 token IDs (a 128K</text>
<text x="552" y="12348" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">vocabulary doesn&#x27;t fit in uint16). Even</text>
<text x="552" y="12363" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">405B on 16,384 GPUs streams only ~11 MB/s.</text>
<text x="540" y="12392" text-anchor="start" style="fill:var(--c-text);font-size:12px">A checkpoint, by contrast, is 70</text>
<text x="540" y="12409" text-anchor="start" style="fill:var(--c-text);font-size:12px">billion × 16 bytes ≈ 1.12 TB (the formula</text>
<text x="540" y="12426" text-anchor="start" style="fill:var(--c-text);font-size:12px">in section 4 of the text). NVIDIA notes it</text>
<text x="540" y="12443" text-anchor="start" style="fill:var(--c-text);font-size:12px">is written synchronously and blocks</text>
<text x="540" y="12460" text-anchor="start" style="fill:var(--c-text);font-size:12px">training, so the storage fabric is sized</text>
<text x="540" y="12477" text-anchor="start" style="fill:var(--c-text);font-size:12px">for that short burst. Part C&#x27;s I/O wall</text>
<text x="540" y="12494" text-anchor="start" style="fill:var(--c-text);font-size:12px">prices the pause with Young-Daly.</text>
<line x1="530.0" y1="12226.0" x2="530.0" y2="12500.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="12540" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Software: the journey of a training job</text>
<text x="40" y="12558" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Six pieces of software carry a job from submission to its first failure; the number gives the order and the stripe the pillar it</text>
<text x="40" y="12573" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">serves.</text>
<rect x="40.0" y="12587" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="12587" width="4" height="118" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<circle cx="62.0" cy="12605" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="62.0" y="12609.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="78.0" y="12609" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Submit the job</text>
<rect x="229.8" y="12617.0" width="38.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="248.9" y="12630.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">mgmt</text>
<text x="52.0" y="12629" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Base Command Platform</text>
<text x="52.0" y="12647" text-anchor="start" style="fill:var(--c-text);font-size:11px">The data scientist picks the job, the</text>
<text x="52.0" y="12662" text-anchor="start" style="fill:var(--c-text);font-size:11px">dataset and the container, without</text>
<text x="52.0" y="12677" text-anchor="start" style="fill:var(--c-text);font-size:11px">needing to know the hardware</text>
<text x="52.0" y="12692" text-anchor="start" style="fill:var(--c-text);font-size:11px">underneath.</text>
<rect x="302.0" y="12587" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="302.0" y="12587" width="4" height="118" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="324.0" cy="12605" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="324.0" y="12609.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="340.0" y="12609" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Allocate nodes</text>
<rect x="473.6" y="12617.0" width="56.4" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="501.8" y="12630.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<text x="314.0" y="12629" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">Slurm</text>
<text x="314.0" y="12647" text-anchor="start" style="fill:var(--c-text);font-size:11px">Decides which GPUs a multi-node job</text>
<text x="314.0" y="12662" text-anchor="start" style="fill:var(--c-text);font-size:11px">runs on and in what order; it can</text>
<text x="314.0" y="12677" text-anchor="start" style="fill:var(--c-text);font-size:11px">drain a failed node and requeue the</text>
<text x="314.0" y="12692" text-anchor="start" style="fill:var(--c-text);font-size:11px">job.</text>
<line x1="279.0" y1="12605.0" x2="299.0" y2="12605.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="564.0" y="12587" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="564.0" y="12587" width="4" height="118" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<circle cx="586.0" cy="12605" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="586.0" y="12609.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="602.0" y="12609" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Install one image</text>
<rect x="753.8" y="12617.0" width="38.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.9" y="12630.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">mgmt</text>
<text x="576.0" y="12629" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Base Command Manager</text>
<text x="576.0" y="12647" text-anchor="start" style="fill:var(--c-text);font-size:11px">Installs, updates and monitors one OS</text>
<text x="576.0" y="12662" text-anchor="start" style="fill:var(--c-text);font-size:11px">image on every node; managing 127</text>
<text x="576.0" y="12677" text-anchor="start" style="fill:var(--c-text);font-size:11px">machines by hand is not realistic.</text>
<line x1="541.0" y1="12605.0" x2="561.0" y2="12605.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<path d="M682.0 12705 V12717 H158.0 V12727" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="40.0" y="12729" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="12729" width="4" height="118" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="62.0" cy="12747" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="62.0" y="12751.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="78.0" y="12751" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Start one stack</text>
<rect x="211.7" y="12759.0" width="56.4" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="239.8" y="12772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">compute</text>
<text x="52.0" y="12771" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">NGC · AI Enterprise</text>
<text x="52.0" y="12789" text-anchor="start" style="fill:var(--c-text);font-size:11px">CUDA, NCCL and PyTorch arrive as</text>
<text x="52.0" y="12804" text-anchor="start" style="fill:var(--c-text);font-size:11px">matched containers; AI Enterprise</text>
<text x="52.0" y="12819" text-anchor="start" style="fill:var(--c-text);font-size:11px">adds supported tools and</text>
<text x="52.0" y="12834" text-anchor="start" style="fill:var(--c-text);font-size:11px">orchestration.</text>
<rect x="302.0" y="12729" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="302.0" y="12729" width="4" height="118" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="324.0" cy="12747" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="324.0" y="12751.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="340.0" y="12751" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Move the data</text>
<rect x="413.1" y="12759.0" width="116.9" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="471.6" y="12772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">network + storage</text>
<text x="314.0" y="12771" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">Magnum IO</text>
<text x="314.0" y="12789" text-anchor="start" style="fill:var(--c-text);font-size:11px">NCCL runs on the compute fabric (with</text>
<text x="314.0" y="12804" text-anchor="start" style="fill:var(--c-text);font-size:11px">SHARP), GPUDirect Storage on the</text>
<text x="314.0" y="12819" text-anchor="start" style="fill:var(--c-text);font-size:11px">storage fabric: steps 1, 3, 4, 6 and</text>
<text x="314.0" y="12834" text-anchor="start" style="fill:var(--c-text);font-size:11px">8 of part B.</text>
<line x1="279.0" y1="12747.0" x2="299.0" y2="12747.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="564.0" y="12729" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="564.0" y="12729" width="4" height="118" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="586.0" cy="12747" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="586.0" y="12751.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="602.0" y="12751" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Watch and repair</text>
<rect x="735.6" y="12759.0" width="56.4" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="763.8" y="12772.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">network</text>
<text x="576.0" y="12771" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">UFM</text>
<text x="576.0" y="12789" text-anchor="start" style="fill:var(--c-text);font-size:11px">Finds the bad port or cable; Slurm</text>
<text x="576.0" y="12804" text-anchor="start" style="fill:var(--c-text);font-size:11px">drains the node and the job resumes</text>
<text x="576.0" y="12819" text-anchor="start" style="fill:var(--c-text);font-size:11px">from the last checkpoint.</text>
<line x1="541.0" y1="12747.0" x2="561.0" y2="12747.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="40.0" y="12861.0" width="760.0" height="67" rx="5" style="fill:var(--c-success);fill-opacity:.08;stroke:var(--c-success);stroke-width:1.2"/>
<text x="52" y="12881" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-success);font-weight:700">Why it matters:</tspan> in a 54-day window of Llama 3 training there were 419 unexpected interruptions (about one every 3</text>
<text x="52" y="12898" text-anchor="start" style="fill:var(--c-text);font-size:12px">hours); only 3 needed significant manual intervention, automation handled the rest, and effective training time stayed</text>
<text x="52" y="12915" text-anchor="start" style="fill:var(--c-text);font-size:12px">above 90% (E).</text>
<text x="40" y="12968" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Mental model: the back of the restaurant</text>
<text x="40" y="12986" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">The restaurant-chain analogy, continued: the parts the customer never sees but everything depends on.</text>
<rect x="40.0" y="12998" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="12998" width="4" height="186" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="54.0" y="13018" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Supply line</text>
<rect x="52.0" y="13026.0" width="157.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="130.5" y="13040.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">storage fabric</text>
<text x="54.0" y="13064" text-anchor="start" style="fill:var(--c-text);font-size:11px">Ingredients (tokens) arrive</text>
<text x="54.0" y="13079" text-anchor="start" style="fill:var(--c-text);font-size:11px">in small, steady amounts: a</text>
<text x="54.0" y="13094" text-anchor="start" style="fill:var(--c-text);font-size:11px">few MB per second. The real</text>
<text x="54.0" y="13109" text-anchor="start" style="fill:var(--c-text);font-size:11px">load is moving the till</text>
<text x="54.0" y="13124" text-anchor="start" style="fill:var(--c-text);font-size:11px">(the checkpoint) at the end</text>
<text x="54.0" y="13139" text-anchor="start" style="fill:var(--c-text);font-size:11px">of the day.</text>
<text x="54.0" y="13172" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">1.12 TB · 4.5–37 s</text>
<rect x="233.0" y="12998" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="12998" width="4" height="186" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="247.0" y="13018" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Service corridor</text>
<rect x="245.0" y="13026.0" width="157.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="323.5" y="13040.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">in-band management</text>
<text x="247.0" y="13064" text-anchor="start" style="fill:var(--c-text);font-size:11px">Staff entry, recipe books</text>
<text x="247.0" y="13079" text-anchor="start" style="fill:var(--c-text);font-size:11px">(containers) and the rota</text>
<text x="247.0" y="13094" text-anchor="start" style="fill:var(--c-text);font-size:11px">use this corridor, so the</text>
<text x="247.0" y="13109" text-anchor="start" style="fill:var(--c-text);font-size:11px">dining room (compute</text>
<text x="247.0" y="13124" text-anchor="start" style="fill:var(--c-text);font-size:11px">fabric) stays clear.</text>
<text x="247.0" y="13172" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">2 × 100 GbE</text>
<rect x="426.0" y="12998" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="12998" width="4" height="186" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="440.0" y="13018" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Fuse box</text>
<rect x="438.0" y="13026.0" width="157.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="516.5" y="13040.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">OOB · BMC</text>
<text x="440.0" y="13064" text-anchor="start" style="fill:var(--c-text);font-size:11px">Even if a kitchen locks up,</text>
<text x="440.0" y="13079" text-anchor="start" style="fill:var(--c-text);font-size:11px">the fuse box is reachable</text>
<text x="440.0" y="13094" text-anchor="start" style="fill:var(--c-text);font-size:11px">by its own path; the</text>
<text x="440.0" y="13109" text-anchor="start" style="fill:var(--c-text);font-size:11px">restaurant can be switched</text>
<text x="440.0" y="13124" text-anchor="start" style="fill:var(--c-text);font-size:11px">off and on remotely.</text>
<text x="440.0" y="13172" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">1 GbE · 8 × SN2201</text>
<rect x="619.0" y="12998" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="12998" width="4" height="186" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="633.0" y="13018" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Shift manager</text>
<rect x="631.0" y="13026.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="709.5" y="13040.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Slurm · BCM · UFM</text>
<text x="633.0" y="13064" text-anchor="start" style="fill:var(--c-text);font-size:11px">Decides who works in which</text>
<text x="633.0" y="13079" text-anchor="start" style="fill:var(--c-text);font-size:11px">kitchen, closes a broken</text>
<text x="633.0" y="13094" text-anchor="start" style="fill:var(--c-text);font-size:11px">kitchen and restarts the</text>
<text x="633.0" y="13109" text-anchor="start" style="fill:var(--c-text);font-size:11px">work from the last save.</text>
<text x="633.0" y="13172" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">419 stops, 3 by hand</text>
<rect x="36.0" y="13196.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="13213" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Golden rule: 93% of the power goes to the DGX systems; every other part exists so that power isn&#x27;t wasted. Everything that would</text>
<text x="46" y="13228" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">make a GPU wait (a clogged fabric, a slow checkpoint, a manual repair) is what these parts prevent.</text>
<rect x="28.0" y="13262" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="13282" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: the further from the GPU, the narrower the road: HBM 3.35 TB/s → NVLink 450 GB/s → network 50 GB/s.</text>
<text x="420.0" y="13327" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">We know the parts. Now let&#x27;s watch them work together in one training step.</text>
<line x1="420.0" y1="13335" x2="420.0" y2="13355" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="13359" width="808" height="4282" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="13383" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="13387.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">B</text>
<text x="56" y="13388" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Follow one training step: where does data go?</text>
<text x="28" y="13409" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Llama 3 405B on a 64-SU H100 SuperPOD: TP 8 × PP 16 × DP 128 = 16,384 GPUs</text>
<text x="28" y="13427" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Follows the text&#x27;s “4D parallelism rooflines for LLM scaling” and “End-to-end cluster walkthrough” subsections; the layout comes from Table</text>
<text x="28" y="13442" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">4 of the Llama 3 report, the numbers from the text&#x27;s formulas.</text>
<rect x="28" y="13456" width="784" height="2855" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="13482" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4D parallelism: how do we split a model across thousands of GPUs?</text>
<text x="40" y="13502" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Problem first, then the fix. Each parallelism solves one problem, and the order runs inside out: TP inside the node, PP</text>
<text x="40" y="13519" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">across nodes, DP outermost. Two more dimensions cover MoE and very long context: EP and CP.</text>
<text x="40" y="13553" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">The problem: one GPU is both far too slow and far too small</text>
<rect x="36" y="13565" width="374" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="13591" text-anchor="start" style="fill:var(--c-danger);font-size:18px;font-weight:700">~3,000 years</text>
<text x="48" y="13611" text-anchor="start" style="fill:var(--c-text);font-size:12px">Training Llama 3 405B (3.8 × 10²⁵ FLOPs) on one H100 at</text>
<text x="48" y="13628" text-anchor="start" style="fill:var(--c-text);font-size:12px">400 TFLOP/s would take this long.</text>
<rect x="422" y="13565" width="374" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="434" y="13591" text-anchor="start" style="fill:var(--c-danger);font-size:18px;font-weight:700">6.5 TB &gt; 80 GB</text>
<text x="434" y="13611" text-anchor="start" style="fill:var(--c-text);font-size:12px">The model&#x27;s training state (16 bytes per parameter) is 81</text>
<text x="434" y="13628" text-anchor="start" style="fill:var(--c-text);font-size:12px">times one GPU&#x27;s memory.</text>
<text x="40" y="13659" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">The fix is to split the work across GPUs; but what you split decides how often the GPUs must talk.</text>
<text x="40" y="13699" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">The fix, step by step: splitting Llama 3 405B across 16,384 GPUs</text>
<text x="40" y="13717" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Each step solves one problem and leaves a new one. The table on the right shows what the step changed: the GPU count, the training</text>
<text x="40" y="13732" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">state per GPU (it must fit in 80 GB) and the training time.</text>
<rect x="36" y="13733" width="768" height="197" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="13733" width="4" height="197" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="58" cy="13755" r="11" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="58" y="13759.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">1</text>
<text x="78" y="13760" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Problem:</tspan> the model doesn&#x27;t fit on one</text>
<text x="78" y="13777" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU. The training state is 6,480 GB; the</text>
<text x="78" y="13794" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU has 80 GB.</text>
<rect x="78.0" y="13806.0" width="167.2" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="161.6" y="13820.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TP 8 · tensor parallelism</text>
<text x="78" y="13846" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Fix:</tspan> split every layer&#x27;s matrices across</text>
<text x="78" y="13863" text-anchor="start" style="fill:var(--c-text);font-size:12px">the node&#x27;s 8 GPUs. Each GPU holds and</text>
<text x="78" y="13880" text-anchor="start" style="fill:var(--c-text);font-size:12px">computes one eighth of every matrix; the</text>
<text x="78" y="13897" text-anchor="start" style="fill:var(--c-text);font-size:12px">partial results are summed in every</text>
<text x="78" y="13914" text-anchor="start" style="fill:var(--c-text);font-size:12px">layer.</text>
<text x="346" y="13757" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">W · one layer&#x27;s matrix, 8 slices</text>
<rect x="348.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="362.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<line x1="362.0" y1="13784.0" x2="362.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="379.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="393.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="393.0" y1="13784.0" x2="393.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="410.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.30;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="424.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<line x1="424.0" y1="13784.0" x2="424.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="441.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.30;stroke:var(--c-success);stroke-width:1.2"/>
<text x="455.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<line x1="455.0" y1="13784.0" x2="455.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="472.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="486.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<line x1="486.0" y1="13784.0" x2="486.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="503.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="517.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<line x1="517.0" y1="13784.0" x2="517.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="534.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.30;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="548.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<line x1="548.0" y1="13784.0" x2="548.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="565.0" y="13763.0" width="29.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.30;stroke:var(--c-success);stroke-width:1.2"/>
<text x="579.5" y="13777.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<line x1="579.0" y1="13784.0" x2="579.0" y2="13796.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="346" y="13797" width="250" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="350.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="363.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G0</text>
<rect x="381.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="394.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G1</text>
<rect x="412.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="425.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G2</text>
<rect x="443.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="456.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G3</text>
<rect x="474.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="487.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G4</text>
<rect x="505.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="518.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G5</text>
<rect x="536.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="549.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G6</text>
<rect x="567.0" y="13803.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="580.5" y="13816.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G7</text>
<text x="346" y="13845" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">slice i goes to GPU i of the node</text>
<text x="612" y="13755" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">before → after</text>
<text x="612" y="13773" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs</text>
<text x="796" y="13773" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1 → 8</text>
<text x="612" y="13790" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per GPU</text>
<text x="796" y="13790" text-anchor="end" style="fill:var(--c-danger);font-size:11px;font-weight:700;font-family:var(--font-mono)">6,480 → 810 GB</text>
<text x="612" y="13807" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time</text>
<text x="796" y="13807" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3,000 → 380 yrs</text>
<text x="612" y="13832" text-anchor="start" style="fill:var(--c-text);font-size:11px">Why TP first: it talks in every</text>
<text x="612" y="13847" text-anchor="start" style="fill:var(--c-text);font-size:11px">layer, so it needs the fastest</text>
<text x="612" y="13862" text-anchor="start" style="fill:var(--c-text);font-size:11px">road, NVLink inside the node. A</text>
<text x="612" y="13877" text-anchor="start" style="fill:var(--c-text);font-size:11px">node has 8 GPUs, so it stops at</text>
<text x="612" y="13892" text-anchor="start" style="fill:var(--c-text);font-size:11px">8.</text>
<line x1="600.0" y1="13745.0" x2="600.0" y2="13918.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="13931.0" x2="420.0" y2="13943.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="36" y="13944" width="768" height="190" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="13944" width="4" height="190" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="58" cy="13966" r="11" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="58" y="13970.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">2</text>
<text x="78" y="13971" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Problem:</tspan> it still doesn&#x27;t fit (810 GB),</text>
<text x="78" y="13988" text-anchor="start" style="fill:var(--c-text);font-size:12px">and TP can&#x27;t go past 8.</text>
<rect x="78.0" y="14000.0" width="185.4" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="170.7" y="14014.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PP 16 · pipeline parallelism</text>
<text x="78" y="14040" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Fix:</tspan> spread the model&#x27;s 126 layers in</text>
<text x="78" y="14057" text-anchor="start" style="fill:var(--c-text);font-size:12px">order over 16 nodes. Each node runs ~8</text>
<text x="78" y="14074" text-anchor="start" style="fill:var(--c-text);font-size:12px">layers and passes its output to the next</text>
<text x="78" y="14091" text-anchor="start" style="fill:var(--c-text);font-size:12px">node on a small tray (34 MB).</text>
<text x="346" y="13968" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">stage 0</text>
<text x="594" y="13968" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">stage 15</text>
<rect x="346.0" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="361.6" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="377.2" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="392.8" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="408.4" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="424.0" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="439.6" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="455.2" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="470.8" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="486.4" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="502.0" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="517.6" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="533.2" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="548.8" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="564.4" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="580.0" y="13974.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="346.0" y1="14010.0" x2="594.0" y2="14010.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="346" y="14030" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 nodes in order, ~8 layers each</text>
<text x="612" y="13966" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">before → after</text>
<text x="612" y="13984" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs</text>
<text x="796" y="13984" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8 → 128</text>
<text x="612" y="14001" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per GPU</text>
<text x="796" y="14001" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">810 → 50.6 GB</text>
<text x="612" y="14018" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time</text>
<text x="796" y="14018" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">380 → 24 yrs</text>
<text x="612" y="14043" text-anchor="start" style="fill:var(--c-text);font-size:11px">16 nodes × 8 GPUs = a 128-GPU</text>
<text x="612" y="14058" text-anchor="start" style="fill:var(--c-text);font-size:11px">chain. Where: across nodes, on</text>
<text x="612" y="14073" text-anchor="start" style="fill:var(--c-text);font-size:11px">the rail; the tray is small, so</text>
<text x="612" y="14088" text-anchor="start" style="fill:var(--c-text);font-size:11px">a slow road is enough. The cost</text>
<text x="612" y="14103" text-anchor="start" style="fill:var(--c-text);font-size:11px">is stages waiting while the line</text>
<text x="612" y="14118" text-anchor="start" style="fill:var(--c-text);font-size:11px">fills (the bubble).</text>
<line x1="600.0" y1="13956.0" x2="600.0" y2="14122.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="14135.0" x2="420.0" y2="14147.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="36" y="14148" width="768" height="180" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="14148" width="4" height="180" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="58" cy="14170" r="11" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="58" y="14174.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">3</text>
<text x="78" y="14175" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Problem:</tspan> the model fits, but training on</text>
<text x="78" y="14192" text-anchor="start" style="fill:var(--c-text);font-size:12px">one chain would take ~24 years.</text>
<rect x="78.0" y="14204.0" width="167.2" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="161.6" y="14218.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DP 128 · data parallelism</text>
<text x="78" y="14244" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Fix:</tspan> copy this 128-GPU chain 128 times.</text>
<text x="78" y="14261" text-anchor="start" style="fill:var(--c-text);font-size:12px">Each copy works on different data; at the</text>
<text x="78" y="14278" text-anchor="start" style="fill:var(--c-text);font-size:12px">end of the step the copies average what</text>
<text x="78" y="14295" text-anchor="start" style="fill:var(--c-text);font-size:12px">they learned (the gradients) and make the</text>
<text x="78" y="14312" text-anchor="start" style="fill:var(--c-text);font-size:12px">same update.</text>
<rect x="346.0" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent);fill-opacity:0.8"/>
<rect x="353.8" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14166" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="346.0" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="353.8" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14175" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="346.0" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="353.8" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14184" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="346.0" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="353.8" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14193" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<text x="346" y="14218" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">each square is a 128-GPU chain: 128 copies</text>
<text x="612" y="14170" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">before → after</text>
<text x="612" y="14188" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs</text>
<text x="796" y="14188" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">128 → 16,384</text>
<text x="612" y="14205" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per GPU</text>
<text x="796" y="14205" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">50.6 → 50.6 GB</text>
<text x="612" y="14222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time</text>
<text x="796" y="14222" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">24 yrs → 67 days</text>
<text x="612" y="14247" text-anchor="start" style="fill:var(--c-text);font-size:11px">Where: the whole cluster, rail +</text>
<text x="612" y="14262" text-anchor="start" style="fill:var(--c-text);font-size:11px">spine. They talk once a step, so</text>
<text x="612" y="14277" text-anchor="start" style="fill:var(--c-text);font-size:11px">they sit outermost. Memory</text>
<text x="612" y="14292" text-anchor="start" style="fill:var(--c-text);font-size:11px">doesn&#x27;t change: every copy holds</text>
<text x="612" y="14307" text-anchor="start" style="fill:var(--c-text);font-size:11px">the same state.</text>
<line x1="600.0" y1="14160.0" x2="600.0" y2="14316.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="14329.0" x2="420.0" y2="14341.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="36" y="14342" width="768" height="197" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="14342" width="4" height="197" rx="2" style="fill:var(--c-success);fill-opacity:.85"/>
<circle cx="58" cy="14364" r="11" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.3"/>
<text x="58" y="14368.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">4</text>
<text x="78" y="14369" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Problem:</tspan> each of the 128 copies keeps the</text>
<text x="78" y="14386" text-anchor="start" style="fill:var(--c-text);font-size:12px">same optimizer state separately; only</text>
<text x="78" y="14403" text-anchor="start" style="fill:var(--c-text);font-size:12px">~29 GB is left for activations.</text>
<rect x="78.0" y="14415.0" width="149.1" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="152.6" y="14429.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">FSDP · share the state</text>
<text x="78" y="14455" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-success);font-weight:700">Fix:</tspan> the copies split the optimizer state</text>
<text x="78" y="14472" text-anchor="start" style="fill:var(--c-text);font-size:12px">(12 bytes per parameter) 128 ways. Each</text>
<text x="78" y="14489" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU updates only its own share, and the</text>
<text x="78" y="14506" text-anchor="start" style="fill:var(--c-text);font-size:12px">updated weights are then sent to everyone</text>
<text x="78" y="14523" text-anchor="start" style="fill:var(--c-text);font-size:12px">(all-gather).</text>
<rect x="346" y="14356" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="375.0" y="14369" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">copy 0</text>
<rect x="350.0" y="14375.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="375.0" y="14387.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="350.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<rect x="362.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="375.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="387.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="410" y="14356" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="439.0" y="14369" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">copy 1</text>
<rect x="414.0" y="14375.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="439.0" y="14387.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="414.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="426.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<rect x="439.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="451.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="474" y="14356" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="503.0" y="14369" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">copy 2</text>
<rect x="478.0" y="14375.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="503.0" y="14387.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="478.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="490.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="503.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<rect x="515.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="538" y="14356" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="567.0" y="14369" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">copy 3</text>
<rect x="542.0" y="14375.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="567.0" y="14387.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="542.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="554.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="567.0" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="579.5" y="14396" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<text x="346" y="14432" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Each copy keeps the full model (weights +</text>
<text x="346" y="14446" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradients); the green square is its share of</text>
<text x="346" y="14460" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">the optimizer (¼ here, 1/128 in Llama 3).</text>
<text x="612" y="14364" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">before → after</text>
<text x="612" y="14382" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPUs</text>
<text x="796" y="14382" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16,384 → 16,384</text>
<text x="612" y="14399" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">per GPU</text>
<text x="796" y="14399" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">50.6 → 12.9 GB</text>
<text x="612" y="14416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time</text>
<text x="796" y="14416" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">67 → 67 days</text>
<text x="612" y="14441" text-anchor="start" style="fill:var(--c-text);font-size:11px">Llama 3 used exactly this</text>
<text x="612" y="14456" text-anchor="start" style="fill:var(--c-text);font-size:11px">layout: TP 8 × PP 16 × DP 128</text>
<text x="612" y="14471" text-anchor="start" style="fill:var(--c-text);font-size:11px">with FSDP. It accumulates</text>
<text x="612" y="14486" text-anchor="start" style="fill:var(--c-text);font-size:11px">gradients in FP32, so its real</text>
<text x="612" y="14501" text-anchor="start" style="fill:var(--c-text);font-size:11px">share is ~19 GB.</text>
<line x1="600.0" y1="14354.0" x2="600.0" y2="14527.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="14579" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Summary: what did each step fix?</text>
<text x="40" y="14603" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">step</text>
<text x="186" y="14603" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPUs</text>
<text x="214" y="14603" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">training state per GPU (log scale)</text>
<text x="800" y="14603" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">training time</text>
<line x1="40.0" y1="14610.0" x2="800.0" y2="14610.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="14634" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">start</text>
<text x="186" y="14634" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1</text>
<rect x="214" y="14621" width="243.9" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="463.9" y="14634" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">6,480 GB</text>
<text x="800" y="14634" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~3,000 years</text>
<text x="40" y="14660" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">+ TP 8</text>
<text x="186" y="14660" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8</text>
<rect x="214" y="14647" width="186.1" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="406.1" y="14660" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">810 GB</text>
<text x="800" y="14660" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~380 years</text>
<text x="40" y="14686" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">+ PP 16</text>
<text x="186" y="14686" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">128</text>
<rect x="214" y="14673" width="109.1" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="343.8" y="14686" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">50.6 GB</text>
<text x="800" y="14686" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~24 years</text>
<text x="40" y="14712" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">+ DP 128</text>
<text x="186" y="14712" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16,384</text>
<rect x="214" y="14699" width="109.1" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="343.8" y="14712" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">50.6 GB</text>
<text x="800" y="14712" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~67 days</text>
<text x="40" y="14738" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">+ FSDP</text>
<text x="186" y="14738" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16,384</text>
<rect x="214" y="14725" width="71.1" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="343.8" y="14738" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">12.9 GB</text>
<text x="800" y="14738" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~67 days</text>
<line x1="335.8" y1="14615.0" x2="335.8" y2="14747.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="339.8" y="14761" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">H100 · 80 GB</text>
<text x="40" y="14783" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">TP and PP shrink the memory, DP cuts the time, and FSDP removes the share the copies duplicate. Time is uninterrupted compute at 400</text>
<text x="40" y="14798" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">TFLOP/s; memory is 16 bytes per parameter (text, section 4), activations excluded.</text>
<text x="40" y="14838" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">What do the numbers stand for? Every GPU has an address</text>
<rect x="40.0" y="14850" width="244.0" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="14850" width="4" height="72" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="14876" text-anchor="start" style="fill:var(--c-accent);font-size:18px;font-weight:700">TP 8</text>
<text x="54.0" y="14896" text-anchor="start" style="fill:var(--c-text);font-size:12px">the GPUs in one node</text>
<text x="54.0" y="14913" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">rank 0–7 = the GPU&#x27;s number in the node</text>
<rect x="298.0" y="14850" width="244.0" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="298.0" y="14850" width="4" height="72" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="312.0" y="14876" text-anchor="start" style="fill:var(--c-accent-2);font-size:18px;font-weight:700">PP 16</text>
<text x="312.0" y="14896" text-anchor="start" style="fill:var(--c-text);font-size:12px">the nodes in one chain</text>
<text x="312.0" y="14913" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">rank 0–15 = the stage in the chain</text>
<rect x="556.0" y="14850" width="244.0" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556.0" y="14850" width="4" height="72" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="570.0" y="14876" text-anchor="start" style="fill:var(--c-accent-2);font-size:18px;font-weight:700">DP 128</text>
<text x="570.0" y="14896" text-anchor="start" style="fill:var(--c-text);font-size:12px">the copies of the chain</text>
<text x="570.0" y="14913" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">rank 0–127 = the copy&#x27;s number</text>
<rect x="40.0" y="14934.0" width="760.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="420.0" y="14950.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700;font-family:var(--font-mono)">8 GPUs × 16 nodes × 128 copies = 16,384 GPUs  ·  16 × 128 = 2,048 DGX</text>
<text x="40" y="14992" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Example address: [TP rank, PP rank, DP rank]</text>
<text x="40" y="15030" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">[</text>
<rect x="56.0" y="15006.0" width="30.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="71.0" y="15027.8" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700;font-family:var(--font-mono)">3</text>
<text x="89" y="15030" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">,</text>
<rect x="102.0" y="15006.0" width="30.0" height="32" rx="5" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="117.0" y="15027.8" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700;font-family:var(--font-mono)">5</text>
<text x="135" y="15030" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">,</text>
<rect x="148.0" y="15006.0" width="42.0" height="32" rx="5" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="169.0" y="15027.8" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700;font-family:var(--font-mono)">42</text>
<text x="193" y="15030" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">]</text>
<rect x="40" y="15052" width="10" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="56" y="15062" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP rank 3: GPU 3 of the node</text>
<rect x="40" y="15070" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="56" y="15080" text-anchor="start" style="fill:var(--c-text);font-size:12px">PP rank 5: stage 5 of the chain</text>
<rect x="40" y="15088" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="56" y="15098" text-anchor="start" style="fill:var(--c-text);font-size:12px">DP rank 42: copy 42</text>
<rect x="300.0" y="14994.0" width="500.0" height="50" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="310" y="15011" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">copy 42 · a 16-node chain (stages 0–15)</text>
<rect x="310.0" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.0" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<rect x="340.3" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="353.3" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="370.6" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="383.6" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="400.9" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="413.9" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="431.2" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="444.2" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="461.5" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="474.5" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">5</text>
<rect x="491.8" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="504.8" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="522.1" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="535.1" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<rect x="552.4" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="565.4" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<rect x="582.7" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="595.7" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">9</text>
<rect x="613.0" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="626.0" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">10</text>
<rect x="643.3" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="656.3" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">11</text>
<rect x="673.6" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="686.6" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">12</text>
<rect x="703.9" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="716.9" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">13</text>
<rect x="734.2" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="747.2" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">14</text>
<rect x="764.5" y="15019.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="777.5" y="15032.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">15</text>
<line x1="461.5" y1="15037.0" x2="420.0" y2="15074.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<line x1="487.5" y1="15037.0" x2="800.0" y2="15074.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="420" y="15074" width="380" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="430" y="15091" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">the stage-5 node · 8 GPUs</text>
<rect x="430.0" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="450.0" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="475.5" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="495.5" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="521.0" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="541.0" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="566.5" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.50;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="586.5" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 3</text>
<rect x="612.0" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="632.0" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="657.5" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="677.5" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="703.0" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="723.0" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="748.5" y="15099.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="768.5" y="15112.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<rect x="40.0" y="15142.0" width="760.0" height="65" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="52" y="15162" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Same TP rank, same rail:</tspan> the PP neighbour of [3, 5, 42] is [3, 6, 42] and a DP peer is [3, 5, 43]. All three are GPU 3</text>
<text x="52" y="15179" text-anchor="start" style="fill:var(--c-text);font-size:12px">and sit on rail 3, which is why PP and DP traffic flows along the rails (4). Ranks count from 0; Llama 3 writes the same</text>
<text x="52" y="15196" text-anchor="start" style="fill:var(--c-text);font-size:12px">address as a [TP, CP, PP, DP] vector.</text>
<text x="40" y="15251" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Who does what? All five dimensions in one table</text>
<text x="40" y="15271" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Rule: the more often it talks, the further inside it sits, on the fastest road.</text>
<text x="88" y="15293" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Splits what?</text>
<text x="214" y="15293" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">What does one GPU do?</text>
<text x="392" y="15293" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Where?</text>
<text x="520" y="15293" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">How often does it talk?</text>
<text x="648" y="15293" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Why is it needed?</text>
<line x1="40.0" y1="15300.0" x2="800.0" y2="15300.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="15311.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="59.0" y="15324.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TP</text>
<text x="88" y="15324" text-anchor="start" style="fill:var(--c-text);font-size:11px">a layer&#x27;s matrices</text>
<text x="214" y="15324" text-anchor="start" style="fill:var(--c-text);font-size:11px">computes 1/8 of every layer</text>
<text x="392" y="15324" text-anchor="start" style="fill:var(--c-text);font-size:11px">inside a node ·</text>
<text x="392" y="15339" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink</text>
<text x="520" y="15324" text-anchor="start" style="fill:var(--c-text);font-size:11px">every layer</text>
<text x="648" y="15324" text-anchor="start" style="fill:var(--c-text);font-size:11px">even one layer doesn&#x27;t</text>
<text x="648" y="15339" text-anchor="start" style="fill:var(--c-text);font-size:11px">fit</text>
<rect x="40.0" y="15353.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="59.0" y="15366.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PP</text>
<text x="88" y="15366" text-anchor="start" style="fill:var(--c-text);font-size:11px">the layers</text>
<text x="214" y="15366" text-anchor="start" style="fill:var(--c-text);font-size:11px">runs ~8 layers and passes</text>
<text x="214" y="15381" text-anchor="start" style="fill:var(--c-text);font-size:11px">the result on</text>
<text x="392" y="15366" text-anchor="start" style="fill:var(--c-text);font-size:11px">across nodes · rail</text>
<text x="520" y="15366" text-anchor="start" style="fill:var(--c-text);font-size:11px">every micro-batch</text>
<text x="648" y="15366" text-anchor="start" style="fill:var(--c-text);font-size:11px">the model doesn&#x27;t fit in</text>
<text x="648" y="15381" text-anchor="start" style="fill:var(--c-text);font-size:11px">a node</text>
<rect x="40.0" y="15395.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="59.0" y="15408.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DP</text>
<text x="88" y="15408" text-anchor="start" style="fill:var(--c-text);font-size:11px">the data</text>
<text x="214" y="15408" text-anchor="start" style="fill:var(--c-text);font-size:11px">runs the whole chain on</text>
<text x="214" y="15423" text-anchor="start" style="fill:var(--c-text);font-size:11px">other data</text>
<text x="392" y="15408" text-anchor="start" style="fill:var(--c-text);font-size:11px">whole cluster ·</text>
<text x="392" y="15423" text-anchor="start" style="fill:var(--c-text);font-size:11px">rail + spine</text>
<text x="520" y="15408" text-anchor="start" style="fill:var(--c-text);font-size:11px">once a step</text>
<text x="648" y="15408" text-anchor="start" style="fill:var(--c-text);font-size:11px">one chain is too slow</text>
<line x1="40.0" y1="15433.0" x2="800.0" y2="15433.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="800" y="15446" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">when needed</text>
<rect x="40.0" y="15457.0" width="38.0" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="59.0" y="15470.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">EP</text>
<text x="88" y="15470" text-anchor="start" style="fill:var(--c-text);font-size:11px">MoE experts</text>
<text x="214" y="15470" text-anchor="start" style="fill:var(--c-text);font-size:11px">holds a few experts and</text>
<text x="214" y="15485" text-anchor="start" style="fill:var(--c-text);font-size:11px">handles incoming tokens</text>
<text x="392" y="15470" text-anchor="start" style="fill:var(--c-text);font-size:11px">across nodes</text>
<text x="520" y="15470" text-anchor="start" style="fill:var(--c-text);font-size:11px">every MoE layer</text>
<text x="648" y="15470" text-anchor="start" style="fill:var(--c-text);font-size:11px">the experts don&#x27;t fit</text>
<text x="648" y="15485" text-anchor="start" style="fill:var(--c-text);font-size:11px">(MoE only)</text>
<rect x="40.0" y="15499.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="59.0" y="15512.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CP</text>
<text x="88" y="15512" text-anchor="start" style="fill:var(--c-text);font-size:11px">a long sequence</text>
<text x="214" y="15512" text-anchor="start" style="fill:var(--c-text);font-size:11px">handles one piece of the</text>
<text x="214" y="15527" text-anchor="start" style="fill:var(--c-text);font-size:11px">sequence</text>
<text x="392" y="15512" text-anchor="start" style="fill:var(--c-text);font-size:11px">node + network</text>
<text x="520" y="15512" text-anchor="start" style="fill:var(--c-text);font-size:11px">every attention</text>
<text x="520" y="15527" text-anchor="start" style="fill:var(--c-text);font-size:11px">layer</text>
<text x="648" y="15512" text-anchor="start" style="fill:var(--c-text);font-size:11px">a long sequence&#x27;s</text>
<text x="648" y="15527" text-anchor="start" style="fill:var(--c-text);font-size:11px">activations don&#x27;t fit</text>
<text x="40" y="15553" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Sources count the fourth dimension of 4D as EP or CP: MoE models split their experts with EP (DeepSeek-V3: 256 experts, EP 64), and</text>
<text x="40" y="15568" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Llama 3 splits the sequence with CP at 128K context (CP 16).</text>
<text x="40" y="15608" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">How does it work? Four pictures: a small example and its Llama 3 version</text>
<rect x="36" y="15622" width="374" height="317" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="422" y="15622" width="374" height="317" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="15643" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">DP · data parallelism</text>
<rect x="48.0" y="15656" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="88.8" y="15671" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">copy 0</text>
<rect x="54.0" y="15678.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.8" y="15691.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">full model</text>
<rect x="54.0" y="15701.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="88.8" y="15714.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">data 1/4</text>
<line x1="88.8" y1="15726.0" x2="88.8" y2="15740.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="137.5" y="15656" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="178.2" y="15671" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">copy 1</text>
<rect x="143.5" y="15678.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="178.2" y="15691.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">full model</text>
<rect x="143.5" y="15701.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="178.2" y="15714.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">data 2/4</text>
<line x1="178.2" y1="15726.0" x2="178.2" y2="15740.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="227.0" y="15656" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="267.8" y="15671" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">copy 2</text>
<rect x="233.0" y="15678.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="267.8" y="15691.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">full model</text>
<rect x="233.0" y="15701.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="267.8" y="15714.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">data 3/4</text>
<line x1="267.8" y1="15726.0" x2="267.8" y2="15740.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="316.5" y="15656" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="357.2" y="15671" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">copy 3</text>
<rect x="322.5" y="15678.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="357.2" y="15691.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">full model</text>
<rect x="322.5" y="15701.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="357.2" y="15714.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">data 4/4</text>
<line x1="357.2" y1="15726.0" x2="357.2" y2="15740.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="88.8" y1="15740.0" x2="357.2" y2="15740.0" marker-start="url(#bp-gen)" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<text x="223.0" y="15757" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">gradient AllReduce · once a step</text>
<text x="48" y="15780" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">In the picture:</tspan> 4 copies hold the same model and each</text>
<text x="48" y="15795" text-anchor="start" style="fill:var(--c-text);font-size:11px">processes a quarter of the data; at the end of the step the</text>
<text x="48" y="15810" text-anchor="start" style="fill:var(--c-text);font-size:11px">gradients are averaged and everyone makes the same update.</text>
<text x="48" y="15830" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">In Llama 3:</tspan> the model doesn&#x27;t fit on one GPU, so a copy is not</text>
<text x="48" y="15845" text-anchor="start" style="fill:var(--c-text);font-size:11px">one GPU but a 128-GPU chain (TP 8 × PP 16). DP 128 means 128</text>
<text x="48" y="15860" text-anchor="start" style="fill:var(--c-text);font-size:11px">of these chains. GPUs at the same address average their</text>
<text x="48" y="15875" text-anchor="start" style="fill:var(--c-text);font-size:11px">gradients: [3, 5, 0] … [3, 5, 127], each for its own</text>
<text x="48" y="15890" text-anchor="start" style="fill:var(--c-text);font-size:11px">3.16-billion-parameter share.</text>
<text x="48" y="15910" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Cost:</tspan> every copy holds the whole model (FSDP lightens this);</text>
<text x="48" y="15925" text-anchor="start" style="fill:var(--c-text);font-size:11px">traffic once a step, gradient-sized.</text>
<text x="434" y="15643" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">TP · tensor parallelism</text>
<rect x="434.0" y="15656.0" width="34.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="451.0" y="15702.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700;font-family:var(--font-mono)">X</text>
<line x1="470.0" y1="15665.0" x2="482.0" y2="15665.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15656.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="528.0" y="15669.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in column 0</text>
<line x1="574.0" y1="15665.0" x2="584.0" y2="15665.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15656.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="630.0" y="15669.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out row 0</text>
<line x1="674.0" y1="15665.0" x2="690.0" y2="15698.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="470.0" y1="15687.0" x2="482.0" y2="15687.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15678.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="528.0" y="15691.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in column 1</text>
<line x1="574.0" y1="15687.0" x2="584.0" y2="15687.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15678.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="630.0" y="15691.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out row 1</text>
<line x1="674.0" y1="15687.0" x2="690.0" y2="15698.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="470.0" y1="15709.0" x2="482.0" y2="15709.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15700.0" width="88.0" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="528.0" y="15713.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in column 2</text>
<line x1="574.0" y1="15709.0" x2="584.0" y2="15709.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15700.0" width="88.0" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="630.0" y="15713.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out row 2</text>
<line x1="674.0" y1="15709.0" x2="690.0" y2="15698.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="470.0" y1="15731.0" x2="482.0" y2="15731.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15722.0" width="88.0" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="528.0" y="15735.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in column 3</text>
<line x1="574.0" y1="15731.0" x2="584.0" y2="15731.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15722.0" width="88.0" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="630.0" y="15735.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out row 3</text>
<line x1="674.0" y1="15731.0" x2="690.0" y2="15698.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<circle cx="702" cy="15698" r="12" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="702" y="15703" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Σ</text>
<line x1="715.0" y1="15698.0" x2="736.0" y2="15698.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<rect x="738.0" y="15688.0" width="30.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="753.0" y="15702.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700;font-family:var(--font-mono)">Y</text>
<text x="434" y="15760" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Each row is one GPU; Σ = AllReduce.</text>
<text x="434" y="15780" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">In the picture:</tspan> 4 GPUs share one layer. X is the same</text>
<text x="434" y="15795" text-anchor="start" style="fill:var(--c-text);font-size:11px">everywhere; each GPU holds a column slice of W_in and a row</text>
<text x="434" y="15810" text-anchor="start" style="fill:var(--c-text);font-size:11px">slice of W_out, and the partial outputs are added with Σ</text>
<text x="434" y="15825" text-anchor="start" style="fill:var(--c-text);font-size:11px">(AllReduce).</text>
<text x="434" y="15845" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">In Llama 3:</tspan> TP 8: the node&#x27;s 8 GPUs share every layer. Of</text>
<text x="434" y="15860" text-anchor="start" style="fill:var(--c-text);font-size:11px">405B&#x27;s 16,384 × 53,248 W_in, each GPU holds a 16,384 × 6,656</text>
<text x="434" y="15875" text-anchor="start" style="fill:var(--c-text);font-size:11px">slice; in attention, 16 of the 128 heads land on each GPU.</text>
<text x="434" y="15895" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Cost:</tspan> 2 sums per layer forward and 2 backward (~1 ms each for</text>
<text x="434" y="15910" text-anchor="start" style="fill:var(--c-text);font-size:11px">405B), so NVLink only.</text>
<rect x="36" y="15951" width="374" height="342" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="422" y="15951" width="374" height="342" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="15972" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">PP · pipeline parallelism</text>
<text x="48" y="15997" text-anchor="start" style="fill:var(--c-text);font-size:11px">stage 0</text>
<rect x="107.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="116.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="127.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="136.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="147.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="156.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="167.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="187.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="207.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="227.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="247.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="267.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="287.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="307.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="327.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="336.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="347.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="367.0" y="15985.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="376.0" y="15997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="48" y="16017" text-anchor="start" style="fill:var(--c-text);font-size:11px">stage 1</text>
<rect x="107.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="127.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="136.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="147.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="156.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="167.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="187.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="196.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="207.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="227.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="247.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="267.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="287.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="296.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="307.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="327.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="336.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="347.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="16017.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="367.0" y="16005.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="48" y="16037" text-anchor="start" style="fill:var(--c-text);font-size:11px">stage 2</text>
<rect x="107.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="127.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="147.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="156.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="167.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="187.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="196.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="207.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="216.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="227.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="247.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="267.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="276.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="287.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="296.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="307.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="327.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="336.0" y="16037.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="347.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="367.0" y="16025.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="48" y="16057" text-anchor="start" style="fill:var(--c-text);font-size:11px">stage 3</text>
<rect x="107.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="127.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="147.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="167.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="187.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="196.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="207.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="216.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="227.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="247.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="256.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="267.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="276.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="287.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="296.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="307.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="16057.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="327.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="347.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="367.0" y="16045.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="48.0" y="16069.0" width="14.0" height="12" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="68" y="16079" text-anchor="start" style="fill:var(--c-text);font-size:11px">forward</text>
<rect x="124.3" y="16069.0" width="14.0" height="12" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="144.3" y="16079" text-anchor="start" style="fill:var(--c-text);font-size:11px">backward</text>
<rect x="206.8" y="16069.0" width="14.0" height="12" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="226.8" y="16079" text-anchor="start" style="fill:var(--c-text);font-size:11px">bubble</text>
<text x="398" y="16079" text-anchor="end" style="fill:var(--c-danger);font-size:11px;font-weight:700">P = 4, M = 4 → 3/7 ≈ 43%</text>
<text x="48" y="16103" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent-2);font-weight:700">In the picture:</tspan> 4 stages (nodes) and 4 micro-batches. Stage 0</text>
<text x="48" y="16118" text-anchor="start" style="fill:var(--c-text);font-size:11px">finishes micro-batch 1 and hands it to stage 1; once the</text>
<text x="48" y="16133" text-anchor="start" style="fill:var(--c-text);font-size:11px">forward passes are done, the backward pass flows back.</text>
<text x="48" y="16153" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent-2);font-weight:700">In Llama 3:</tspan> PP 16: 16 nodes with ~8 layers each, and each copy</text>
<text x="48" y="16168" text-anchor="start" style="fill:var(--c-text);font-size:11px">processes 16 sequences (micro-batches) per step. In this</text>
<text x="48" y="16183" text-anchor="start" style="fill:var(--c-text);font-size:11px">simple schedule the bubble would be 48%; Llama 3&#x27;s interleaved</text>
<text x="48" y="16198" text-anchor="start" style="fill:var(--c-text);font-size:11px">schedule gives each node several smaller chunks of layers and</text>
<text x="48" y="16213" text-anchor="start" style="fill:var(--c-text);font-size:11px">shrinks the bubble by that number of chunks.</text>
<text x="48" y="16233" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent-2);font-weight:700">Cost:</tspan> idle slots while the line fills and drains (the bubble);</text>
<text x="48" y="16248" text-anchor="start" style="fill:var(--c-text);font-size:11px">it shrinks as micro-batches grow in number.</text>
<text x="434" y="15972" text-anchor="start" style="fill:var(--c-warn);font-size:13px;font-weight:700">EP · expert parallelism</text>
<line x1="448.0" y1="16015.0" x2="474.8" y2="16051.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="466.0" y1="16015.0" x2="564.2" y2="16051.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="484.0" y1="16015.0" x2="653.8" y2="16051.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="502.0" y1="16015.0" x2="743.2" y2="16051.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<line x1="537.5" y1="16015.0" x2="474.8" y2="16051.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="555.5" y1="16015.0" x2="564.2" y2="16051.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="573.5" y1="16015.0" x2="653.8" y2="16051.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="591.5" y1="16015.0" x2="743.2" y2="16051.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<line x1="627.0" y1="16015.0" x2="474.8" y2="16051.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="645.0" y1="16015.0" x2="564.2" y2="16051.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="663.0" y1="16015.0" x2="653.8" y2="16051.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="681.0" y1="16015.0" x2="743.2" y2="16051.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<line x1="716.5" y1="16015.0" x2="474.8" y2="16051.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="734.5" y1="16015.0" x2="564.2" y2="16051.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="752.5" y1="16015.0" x2="653.8" y2="16051.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="770.5" y1="16015.0" x2="743.2" y2="16051.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="434.0" y="15985" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="474.8" y="15998" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 0</text>
<circle cx="448.0" cy="16007" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="466.0" cy="16007" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="484.0" cy="16007" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="502.0" cy="16007" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="434.0" y="16051.0" width="81.5" height="30" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="474.8" y="16064" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 0</text>
<text x="474.8" y="16077" text-anchor="middle" style="fill:var(--c-text);font-size:11px">experts 0–1</text>
<rect x="523.5" y="15985" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="564.2" y="15998" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 1</text>
<circle cx="537.5" cy="16007" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="555.5" cy="16007" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="573.5" cy="16007" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="591.5" cy="16007" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="523.5" y="16051.0" width="81.5" height="30" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="564.2" y="16064" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 1</text>
<text x="564.2" y="16077" text-anchor="middle" style="fill:var(--c-text);font-size:11px">experts 2–3</text>
<rect x="613.0" y="15985" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="653.8" y="15998" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 2</text>
<circle cx="627.0" cy="16007" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="645.0" cy="16007" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="663.0" cy="16007" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="681.0" cy="16007" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="613.0" y="16051.0" width="81.5" height="30" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="653.8" y="16064" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 2</text>
<text x="653.8" y="16077" text-anchor="middle" style="fill:var(--c-text);font-size:11px">experts 4–5</text>
<rect x="702.5" y="15985" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="743.2" y="15998" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 3</text>
<circle cx="716.5" cy="16007" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="734.5" cy="16007" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="752.5" cy="16007" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="770.5" cy="16007" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="702.5" y="16051.0" width="81.5" height="30" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="743.2" y="16064" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 3</text>
<text x="743.2" y="16077" text-anchor="middle" style="fill:var(--c-text);font-size:11px">experts 6–7</text>
<text x="434" y="16099" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Colour shows the expert a token is headed to: every GPU sends</text>
<text x="434" y="16114" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">to every GPU (AllToAll).</text>
<text x="434" y="16134" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-warn);font-weight:700">In the picture:</tspan> 8 experts are spread over 4 GPUs (2 each).</text>
<text x="434" y="16149" text-anchor="start" style="fill:var(--c-text);font-size:11px">Tokens on each GPU travel by colour to their expert&#x27;s GPU</text>
<text x="434" y="16164" text-anchor="start" style="fill:var(--c-text);font-size:11px">(dispatch) and come back with the result (combine).</text>
<text x="434" y="16184" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-warn);font-weight:700">In DeepSeek-V3:</tspan> 256 experts per layer, EP 64: each of the</text>
<text x="434" y="16199" text-anchor="start" style="fill:var(--c-text);font-size:11px">64 GPUs on 8 nodes holds 4 experts; each token goes to 8</text>
<text x="434" y="16214" text-anchor="start" style="fill:var(--c-text);font-size:11px">experts on at most 4 nodes. Llama 3 is a dense model, so it</text>
<text x="434" y="16229" text-anchor="start" style="fill:var(--c-text);font-size:11px">uses no EP.</text>
<text x="434" y="16249" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-warn);font-weight:700">Cost:</tspan> in every MoE layer everyone sends to everyone</text>
<text x="434" y="16264" text-anchor="start" style="fill:var(--c-text);font-size:11px">(AllToAll), which loads the network; if one expert gets too</text>
<text x="434" y="16279" text-anchor="start" style="fill:var(--c-text);font-size:11px">many tokens, the others wait.</text>
<rect x="28" y="16323" width="784" height="863" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="16349" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">Where does the data go? Eight steps, four roads</text>
<text x="40" y="16369" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The layout is set; now let&#x27;s follow one training step on these 16,384 GPUs. The numbers on the map are the steps below;</text>
<text x="40" y="16386" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">figures are per GPU and per step (one sequence of 8,192 tokens per micro-batch).</text>
<rect x="290.0" y="16408" width="260.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="16423" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine → the other 127 DP replicas</text>
<rect x="48" y="16456" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="58" y="16472" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">node A · PP stage s</text>
<rect x="88.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="150.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="212.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="274.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="88.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="150.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="212.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="274.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="112.0" y1="16500" x2="112.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="112.0" y1="16526" x2="112.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="16500" x2="174.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="16526" x2="174.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="16500" x2="236.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="16526" x2="236.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="16500" x2="298.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="16526" x2="298.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="88.0" y="16508" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="205.0" y="16521" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<circle cx="64" cy="16490" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="64" cy="16517" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="64" y="16521" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="64" cy="16544" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<rect x="502" y="16456" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="512" y="16472" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">node B · stage s+1</text>
<rect x="530.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="592.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="654.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="716.0" y="16480" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="16494" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="530.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="592.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="654.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="716.0" y="16534" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="16548" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="554.0" y1="16500" x2="554.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="554.0" y1="16526" x2="554.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="16500" x2="616.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="16526" x2="616.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="16500" x2="678.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="16526" x2="678.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="16500" x2="740.0" y2="16508" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="16526" x2="740.0" y2="16534" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="530.0" y="16508" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="16521" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<rect x="396.0" y="16466" width="48.0" height="96" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="16524" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">leaf</text>
<circle cx="420.0" cy="16492" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="420.0" y="16496" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="420.0" cy="16546" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="420.0" y="16550" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<line x1="338.0" y1="16492" x2="396.0" y2="16492" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<line x1="444.0" y1="16492" x2="500.0" y2="16492" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="502.0" y1="16546" x2="444.0" y2="16546" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:4 3"/>
<line x1="396.0" y1="16546" x2="340.0" y2="16546" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<line x1="420.0" y1="16464" x2="420.0" y2="16432" marker-start="url(#bp-gen)" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<circle cx="438.0" cy="16448" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438.0" y="16452" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="48.0" y="16608" width="744.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="420.0" y="16624" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">storage system (e.g. Lustre · Weka)</text>
<text x="420.0" y="16640" text-anchor="middle" style="fill:var(--c-text);font-size:11px">separate storage fabric · ≥ 40 GB/s per node</text>
<line x1="108.0" y1="16606" x2="108.0" y2="16575" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<circle cx="126" cy="16590" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="126" y="16594" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<line x1="208.0" y1="16575" x2="208.0" y2="16606" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5;stroke-dasharray:5 4"/>
<circle cx="226" cy="16590" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="226" y="16594" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="647.0" y="16592" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(node B uses the same path)</text>
<line x1="40.0" y1="16662.0" x2="800.0" y2="16662.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="46.0" y1="16690.0" x2="46.0" y2="17064.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="46" cy="16690" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="46" y="16694.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="16695" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Read data</text>
<rect x="66.0" y="16704.0" width="171.3" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="151.7" y="16717.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">storage → HBM · 67 MB/step</text>
<text x="66" y="16740" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The step&#x27;s 16.8 million tokens (128 × 16</text>
<text x="66" y="16756" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sequences × 8,192) come to 67 MB as int32, about</text>
<text x="66" y="16772" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">11 MB/s. Only the first stage reads tokens; mmap or GDS</text>
<text x="66" y="16788" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">keep the CPU copy out of the way.</text>
<circle cx="46" cy="16820" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="46" y="16824.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="16825" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Forward pass</text>
<rect x="66.0" y="16834.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="151.7" y="16847.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Tensor Cores · ~2.1 s/step</text>
<text x="66" y="16870" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">3.16 billion parameters per GPU: each 8,192-token</text>
<text x="66" y="16886" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">micro-batch reads the weights once (6.3 GB, ~2 ms) but</text>
<text x="66" y="16902" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">does 52 TFLOP of work (~130 ms). Compute dominates.</text>
<circle cx="46" cy="16934" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="16938.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="16939" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">TP AllReduce</text>
<rect x="66.0" y="16948.0" width="135.0" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="133.5" y="16961.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink · ~500 × 1 ms</text>
<text x="66" y="16984" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Every layer sums a 268 MB activation across 8 GPUs</text>
<text x="66" y="17000" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">twice forward and twice backward: ~8 layers × 4 × 16</text>
<text x="66" y="17016" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">micro-batches ≈ 500 times, ~1 ms each. That is why TP</text>
<text x="66" y="17032" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">never leaves the chassis.</text>
<circle cx="46" cy="17064" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="17068.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="66" y="17069" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Send activations (PP)</text>
<rect x="66.0" y="17078.0" width="159.2" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="145.6" y="17091.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">rail · 34 MB/micro-batch</text>
<text x="66" y="17114" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">A stage&#x27;s output is 8,192 × 16,384 × 2 bytes = 268 MB;</text>
<text x="66" y="17130" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">each TP GPU sends its eighth over its own rail to the</text>
<text x="66" y="17146" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">same-numbered GPU of the next node: 34 MB, ~0.7 ms.</text>
<line x1="438.0" y1="16690.0" x2="438.0" y2="17080.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="438" cy="16690" r="10" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="438" y="16694.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="16695" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Backward pass</text>
<rect x="458.0" y="16704.0" width="189.5" height="19" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="552.7" y="16717.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">same links, reversed · ~4.1 s</text>
<text x="458" y="16740" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Gradients flow back the same way; steps 3 and 4 repeat</text>
<text x="458" y="16756" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">in reverse at ~2× the forward work. Gradients</text>
<text x="458" y="16772" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">accumulate in FP32 across micro-batches.</text>
<circle cx="438" cy="16804" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438" y="16808.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="16809" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Gradient sync (DP)</text>
<rect x="458.0" y="16818.0" width="177.4" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="546.7" y="16831.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">rail + spine · ~0.38 s/step</text>
<text x="458" y="16854" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">FSDP reduce-scatters the FP32 gradients of 128 copies</text>
<text x="458" y="16870" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">(12.7 GB) and all-gathers the updated BF16 weights</text>
<text x="458" y="16886" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">(6.3 GB): ~0.38 s at 50 GB/s, overlapped with the</text>
<text x="458" y="16902" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">backward pass. In layouts that use AllReduce, SHARP</text>
<text x="458" y="16918" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sums in the switch (4b).</text>
<circle cx="438" cy="16950" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="438" y="16954.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<text x="458" y="16955" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Optimizer step</text>
<rect x="458.0" y="16964.0" width="159.2" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="537.6" y="16977.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM · 25M params per GPU</text>
<text x="458" y="17000" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Each GPU updates only its 1/128 share: 24.7 million</text>
<text x="458" y="17016" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">parameters and 0.3 GB of AdamW state (FP32 master</text>
<text x="458" y="17032" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">weight, momentum, variance); it finishes in HBM in</text>
<text x="458" y="17048" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">under a millisecond.</text>
<circle cx="438" cy="17080" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="438" y="17084.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="458" y="17085" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Checkpoint</text>
<rect x="458.0" y="17094.0" width="171.3" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="543.6" y="17107.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">storage · every ~130 steps</text>
<text x="458" y="17130" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Every ~14 minutes 7.5 TB of state goes to disk (~460 MB</text>
<text x="458" y="17146" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">per GPU): ~31 s at 250 GB/s, ~5 min on a legacy 25 GB/s</text>
<text x="458" y="17162" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">path. After a failure, training resumes here.</text>
<line x1="420.0" y1="16674.0" x2="420.0" y2="17168.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="17198" width="784" height="369" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="17224" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">One step&#x27;s budget: how long does each road take?</text>
<text x="40" y="17244" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The eight steps&#x27; times side by side, per GPU and added up serially: where does the time go?</text>
<text x="40" y="17285" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">compute (400 TFLOP/s)</text>
<rect x="200.0" y="17272" width="86.7" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="206" y="17285" text-anchor="start" style="fill:var(--c-text);font-size:11px">forward 2.1</text>
<rect x="286.7" y="17272" width="173.3" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="292.7" y="17285" text-anchor="start" style="fill:var(--c-text);font-size:11px">backward 4.1</text>
<text x="466.0" y="17285" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">6.2 s</text>
<text x="40" y="17311" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">compute at peak (990)</text>
<rect x="200" y="17298" width="105.1" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:4 3"/>
<text x="311.1" y="17311" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">2.5 s</text>
<line x1="40.0" y1="17324.0" x2="520.0" y2="17324.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="17345" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink · TP AllReduce</text>
<rect x="200" y="17332" width="22.0" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="228.0" y="17345" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0.53 s</text>
<text x="40" y="17371" text-anchor="start" style="fill:var(--c-text);font-size:11px">network · DP sync</text>
<rect x="200" y="17358" width="15.8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="221.8" y="17371" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0.38 s</text>
<text x="40" y="17397" text-anchor="start" style="fill:var(--c-text);font-size:11px">storage · checkpoint share</text>
<rect x="200" y="17384" width="9.8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="215.8" y="17397" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0.23 s</text>
<text x="40" y="17423" text-anchor="start" style="fill:var(--c-text);font-size:11px">network · PP activations</text>
<rect x="200" y="17410" width="2.0" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="208" y="17423" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0.02 s</text>
<text x="40" y="17449" text-anchor="start" style="fill:var(--c-text);font-size:11px">storage · token reads</text>
<rect x="200" y="17436" width="2.0" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="208" y="17449" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">~0</text>
<text x="40" y="17472" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Communication times are added serially; in a real step most of it overlaps with</text>
<text x="40" y="17487" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">compute. Checkpoint share: 31 s ÷ ~130 steps. PP is at least 32 sends; an interleaved</text>
<text x="40" y="17502" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">schedule adds more, but it stays small.</text>
<rect x="540.0" y="17260.0" width="260.0" height="144" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="17280" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Where does 6.2 s come from?</text>
<text x="552" y="17299" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6 × 405 billion × 16.8M tokens</text>
<text x="552" y="17315" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">÷ (16,384 × 400 TFLOP/s)</text>
<text x="552" y="17331" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">≈ 6.2 s per step</text>
<text x="552" y="17350" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16.8M = 128 DP × 16</text>
<text x="552" y="17365" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sequences × 8,192 tokens (Table 4); that</text>
<text x="552" y="17380" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">is 2.7 million tokens per second. 400</text>
<text x="552" y="17395" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s is 41% of the H100&#x27;s 990 (MFU).</text>
<text x="540" y="17426" text-anchor="start" style="fill:var(--c-text);font-size:12px">Added serially, communication is ~0.9 s,</text>
<text x="540" y="17443" text-anchor="start" style="fill:var(--c-text);font-size:12px">and most of it hides behind compute: the</text>
<text x="540" y="17460" text-anchor="start" style="fill:var(--c-text);font-size:12px">DP sync overlaps the backward pass and PP</text>
<text x="540" y="17477" text-anchor="start" style="fill:var(--c-text);font-size:12px">sends run asynchronously. The same work</text>
<text x="540" y="17494" text-anchor="start" style="fill:var(--c-text);font-size:12px">would take 2.5 s at peak; the gap to 6.2 s</text>
<text x="540" y="17511" text-anchor="start" style="fill:var(--c-text);font-size:12px">is pipeline bubbles, exposed TP</text>
<text x="540" y="17528" text-anchor="start" style="fill:var(--c-text);font-size:12px">communication and kernel efficiency. Part</text>
<text x="540" y="17545" text-anchor="start" style="fill:var(--c-text);font-size:12px">C&#x27;s walls take them one by one.</text>
<line x1="530.0" y1="17260.0" x2="530.0" y2="17545.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="17579.0" width="784.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="17599" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: in one step data crosses four roads; whoever talks most sits on the fastest one (TP → NVLink,</text>
<text x="40" y="17616" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">PP → rail, DP → network, checkpoint → storage), and a good layout hides communication behind compute.</text>
<text x="420.0" y="17661" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">So where do these roads jam? Each pillar has a wall.</text>
<line x1="420.0" y1="17669" x2="420.0" y2="17689" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="17693" width="808" height="1041" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="17717" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="17721.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">C</text>
<text x="56" y="17722" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Three walls: symptom → cause → fix</text>
<text x="28" y="17743" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Curves are computed from the article&#x27;s formulas; red is the wall, green the fix.</text>
<rect x="28" y="17755" width="784" height="312" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="17779" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Memory wall</text>
<text x="800" y="17779" text-anchor="end" style="fill:var(--c-accent);font-size:11px;font-weight:700">pillar 1 · compute</text>
<line x1="80.0" y1="17799" x2="80.0" y2="17925" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="17925" x2="512.0" y2="17925" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="17806.2" x2="512.0" y2="17806.2" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="17810.2" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">990</text>
<text x="76" y="17929" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="40" y="17798.2" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s</text>
<line x1="377.9" y1="17806.2" x2="377.9" y2="17925" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<polyline points="80,17924.6 84.5,17924.6 89,17924.5 93.5,17924.5 98,17924.4 102.5,17924.4 107,17924.3 111.5,17924.3 116,17924.2 120.5,17924.1 125,17924 129.5,17924 134,17923.9 138.5,17923.8 143,17923.6 147.5,17923.5 152,17923.4 156.5,17923.2 161,17923.1 165.5,17922.9 170,17922.7 174.5,17922.5 179,17922.3 183.5,17922.1 188,17921.8 192.5,17921.5 197,17921.2 201.5,17920.8 206,17920.5 210.5,17920 215,17919.6 219.5,17919.1 224,17918.6 228.5,17918 233,17917.4 237.5,17916.7 242,17915.9 246.5,17915.1 251,17914.2 255.5,17913.3 260,17912.2 264.5,17911 269,17909.8 273.5,17908.4 278,17906.9 282.5,17905.3 287,17903.5 291.5,17901.6 296,17899.5 300.5,17897.2 305,17894.8 309.5,17892.1 314,17889.1 318.5,17885.9 323,17882.4 327.5,17878.7 332,17874.6 336.5,17870.1 341,17865.2 345.5,17860 350,17854.2 354.5,17848 359,17841.3 363.5,17834 368,17826.1 372.5,17817.5 377,17808.2 381.5,17806.2 386,17806.2 390.5,17806.2 395,17806.2 399.5,17806.2 404,17806.2 408.5,17806.2 413,17806.2 417.5,17806.2 422,17806.2 426.5,17806.2 431,17806.2 435.5,17806.2 440,17806.2 444.5,17806.2 449,17806.2 453.5,17806.2 458,17806.2 462.5,17806.2 467,17806.2 471.5,17806.2 476,17806.2 480.5,17806.2 485,17806.2 489.5,17806.2 494,17806.2 498.5,17806.2 503,17806.2 507.5,17806.2 512,17806.2" style="fill:none;stroke:var(--c-accent);stroke-width:2.2"/>
<text x="80.0" y="17939" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="152.0" y="17939" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="224.0" y="17939" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="296.0" y="17939" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="368.0" y="17939" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="440.0" y="17939" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1,024</text>
<text x="512.0" y="17939" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4,096</text>
<text x="512" y="17955" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">batch B (tokens, log scale)</text>
<text x="150" y="17822.2" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">memory-bound</text>
<text x="470" y="17907" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">compute-bound</text>
<circle cx="80.0" cy="17924.6" r="4" style="fill:var(--c-danger)"/>
<text x="88.0" y="17915" text-anchor="start" style="fill:var(--c-text);font-size:11px">B = 1: 3.3 TFLOP/s (0.3%)</text>
<circle cx="332.0" cy="17874.6" r="4" style="fill:var(--c-accent)"/>
<text x="324.0" y="17868.6" text-anchor="end" style="fill:var(--c-text);font-size:11px">B = 128: 420 TFLOP/s (42%)</text>
<circle cx="377.9" cy="17806.2" r="4" style="fill:var(--c-success)"/>
<text x="385.9" y="17822.2" text-anchor="start" style="fill:var(--c-text);font-size:11px">B ≈ 310: ceiling 990</text>
<text x="40" y="17973" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP (D = 8,192, F = 28,672), H100 BF16: speed = min(990, 3.35·α)</text>
<text x="40" y="17989" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">How to read it: as the batch grows, every byte read does more work; the curve hits</text>
<text x="40" y="18004" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">the ceiling near B ≈ 310.</text>
<text x="552" y="17809" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Symptom</text>
<text x="552" y="17827" text-anchor="start" style="fill:var(--c-text);font-size:12px">The GPU looks busy, yet only 1–2% of its</text>
<text x="552" y="17844" text-anchor="start" style="fill:var(--c-text);font-size:12px">power is used.</text>
<text x="552" y="17872" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Cause</text>
<text x="552" y="17890" text-anchor="start" style="fill:var(--c-text);font-size:12px">Too little work per byte: while α = BDF</text>
<text x="552" y="17907" text-anchor="start" style="fill:var(--c-text);font-size:12px">/ (B(D+F) + DF) stays below the H100&#x27;s</text>
<text x="552" y="17924" text-anchor="start" style="fill:var(--c-text);font-size:12px">knee of 295, the Tensor Cores wait for</text>
<text x="552" y="17941" text-anchor="start" style="fill:var(--c-text);font-size:12px">data.</text>
<text x="552" y="17969" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Fix</text>
<rect x="552.0" y="17979" width="128.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="616.5" y="17993" text-anchor="middle" style="fill:var(--c-text);font-size:11px">batch ≥ ~310 tokens</text>
<rect x="552.0" y="18005" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="18019" text-anchor="middle" style="fill:var(--c-text);font-size:11px">fusion / FlashAttention</text>
<rect x="552.0" y="18031" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="18045" text-anchor="middle" style="fill:var(--c-text);font-size:11px">decode: 4–8 bit weights</text>
<line x1="540.0" y1="17795" x2="540.0" y2="18051" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="18077" width="784" height="321" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="18101" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Network wall</text>
<text x="800" y="18101" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">pillar 2 · network</text>
<line x1="184.0" y1="18119" x2="184.0" y2="18203" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="184" y="18217" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 GB/s</text>
<line x1="288.0" y1="18119" x2="288.0" y2="18203" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="288" y="18217" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 GB/s</text>
<line x1="392.0" y1="18119" x2="392.0" y2="18203" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="392" y="18217" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 TB/s</text>
<line x1="496.0" y1="18119" x2="496.0" y2="18203" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="496" y="18217" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 TB/s</text>
<text x="40" y="18137" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM (on chip)</text>
<rect x="184" y="18125" width="262.6" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="452.6" y="18137" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3.35 TB/s</text>
<text x="40" y="18163" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (in node)</text>
<rect x="184" y="18151" width="171.9" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="361.9" y="18163" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">450 GB/s</text>
<text x="420.7" y="18163" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷7.4</text>
<text x="40" y="18189" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (off node)</text>
<rect x="184" y="18177" width="72.7" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="262.7" y="18189" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">50 GB/s</text>
<text x="314.9" y="18189" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷9</text>
<rect x="40.0" y="18229" width="480.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="280.0" y="18243" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink: Y ≤ F / (C/W) = 28,672 / 2,200 ≈ 13 → TP = 8 fits ✓ (F: Llama 3 70B)</text>
<rect x="40.0" y="18255" width="480.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="18269" text-anchor="middle" style="fill:var(--c-text);font-size:11px">InfiniBand: Y ≤ 28,672 / 19,800 ≈ 1.45 → TP across nodes starves compute ✗</text>
<text x="40" y="18293" text-anchor="start" style="fill:var(--c-text);font-size:11px">PP = 16 stages, bubble = 15 / (15 + M): M = 16 → 48% idle · M = 128 → 10.5%</text>
<text x="40" y="18311" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">How to read it: each bar is one H100&#x27;s bandwidth on that link; leaving the node</text>
<text x="40" y="18326" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">shrinks it 9×.</text>
<text x="552" y="18131" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Symptom</text>
<text x="552" y="18149" text-anchor="start" style="fill:var(--c-text);font-size:12px">Adding nodes stretches the step; GPUs</text>
<text x="552" y="18166" text-anchor="start" style="fill:var(--c-text);font-size:12px">wait to talk.</text>
<text x="552" y="18194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Cause</text>
<text x="552" y="18212" text-anchor="start" style="fill:var(--c-text);font-size:12px">Data leaving the node squeezes through a</text>
<text x="552" y="18229" text-anchor="start" style="fill:var(--c-text);font-size:12px">road 9× narrower; jobs that talk every</text>
<text x="552" y="18246" text-anchor="start" style="fill:var(--c-text);font-size:12px">layer, like TP, choke on it.</text>
<text x="552" y="18274" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Fix</text>
<rect x="552.0" y="18284" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="18298" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TP inside the node</text>
<rect x="552.0" y="18310" width="147.1" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="625.5" y="18324" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PP and DP across nodes</text>
<rect x="552.0" y="18336" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="18350" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SHARP: 2·B/W → B/W</text>
<rect x="552.0" y="18362" width="135.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="619.5" y="18376" text-anchor="middle" style="fill:var(--c-text);font-size:11px">DualPipe: hide comms</text>
<line x1="540.0" y1="18117" x2="540.0" y2="18382" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="18408" width="784" height="269" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="18432" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">I/O wall</text>
<text x="800" y="18432" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:700">pillar 3 · storage</text>
<line x1="80.0" y1="18456" x2="80.0" y2="18578" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="18578" x2="512.0" y2="18578" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="18548" x2="512.0" y2="18548" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18552" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10%</text>
<line x1="80.0" y1="18518" x2="512.0" y2="18518" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18522" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">20%</text>
<line x1="80.0" y1="18488" x2="512.0" y2="18488" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18492" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">30%</text>
<line x1="80.0" y1="18458" x2="512.0" y2="18458" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18462" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">40%</text>
<polyline points="80,18578 512,18503" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/>
<polyline points="148.4,18458.3 149.6,18460 150.8,18461.6 152,18463.1 153.2,18464.6 154.4,18466 155.6,18467.4 156.8,18468.7 158,18469.9 159.2,18471.2 160.4,18472.3 161.6,18473.5 162.8,18474.6 164,18475.6 165.2,18476.7 166.4,18477.7 167.6,18478.6 168.8,18479.6 170,18480.5 171.2,18481.3 172.4,18482.2 173.6,18483 174.8,18483.8 176,18484.5 177.2,18485.3 178.4,18486 179.6,18486.7 180.8,18487.4 182,18488 183.2,18488.6 184.4,18489.3 185.6,18489.8 186.8,18490.4 188,18491 189.2,18491.5 190.4,18492.1 191.6,18492.6 192.8,18493.1 194,18493.5 195.2,18494 196.4,18494.5 197.6,18494.9 198.8,18495.3 200,18495.7 201.2,18496.1 202.4,18496.5 203.6,18496.9 204.8,18497.3 206,18497.6 207.2,18498 208.4,18498.3 209.6,18498.6 210.8,18498.9 212,18499.2 213.2,18499.5 214.4,18499.8 215.6,18500.1 216.8,18500.4 218,18500.6 219.2,18500.9 220.4,18501.1 221.6,18501.3 222.8,18501.6 224,18501.8 225.2,18502 226.4,18502.2 227.6,18502.4 228.8,18502.6 230,18502.8 231.2,18503 232.4,18503.2 233.6,18503.3 234.8,18503.5 236,18503.7 237.2,18503.8 238.4,18504 239.6,18504.1 240.8,18504.2 242,18504.4 243.2,18504.5 244.4,18504.6 245.6,18504.7 246.8,18504.8 248,18504.9 249.2,18505.1 250.4,18505.1 251.6,18505.2 252.8,18505.3 254,18505.4 255.2,18505.5 256.4,18505.6 257.6,18505.7 258.8,18505.7 260,18505.8 261.2,18505.9 262.4,18505.9 263.6,18506 264.8,18506 266,18506.1 267.2,18506.1 268.4,18506.2 269.6,18506.2 270.8,18506.2 272,18506.3 273.2,18506.3 274.4,18506.3 275.6,18506.3 276.8,18506.4 278,18506.4 279.2,18506.4 280.4,18506.4 281.6,18506.4 282.8,18506.4 284,18506.4 285.2,18506.4 286.4,18506.4 287.6,18506.4 288.8,18506.4 290,18506.4 291.2,18506.4 292.4,18506.4 293.6,18506.4 294.8,18506.4 296,18506.4 297.2,18506.3 298.4,18506.3 299.6,18506.3 300.8,18506.3 302,18506.2 303.2,18506.2 304.4,18506.2 305.6,18506.2 306.8,18506.1 308,18506.1 309.2,18506 310.4,18506 311.6,18506 312.8,18505.9 314,18505.9 315.2,18505.8 316.4,18505.8 317.6,18505.7 318.8,18505.7 320,18505.6 321.2,18505.6 322.4,18505.5 323.6,18505.4 324.8,18505.4 326,18505.3 327.2,18505.3 328.4,18505.2 329.6,18505.1 330.8,18505.1 332,18505 333.2,18504.9 334.4,18504.9 335.6,18504.8 336.8,18504.7 338,18504.6 339.2,18504.6 340.4,18504.5 341.6,18504.4 342.8,18504.3 344,18504.2 345.2,18504.2 346.4,18504.1 347.6,18504 348.8,18503.9 350,18503.8 351.2,18503.7 352.4,18503.6 353.6,18503.6 354.8,18503.5 356,18503.4 357.2,18503.3 358.4,18503.2 359.6,18503.1 360.8,18503 362,18502.9 363.2,18502.8 364.4,18502.7 365.6,18502.6 366.8,18502.5 368,18502.4 369.2,18502.3 370.4,18502.2 371.6,18502.1 372.8,18502 374,18501.9 375.2,18501.8 376.4,18501.7 377.6,18501.6 378.8,18501.5 380,18501.3 381.2,18501.2 382.4,18501.1 383.6,18501 384.8,18500.9 386,18500.8 387.2,18500.7 388.4,18500.6 389.6,18500.4 390.8,18500.3 392,18500.2 393.2,18500.1 394.4,18500 395.6,18499.8 396.8,18499.7 398,18499.6 399.2,18499.5 400.4,18499.4 401.6,18499.2 402.8,18499.1 404,18499 405.2,18498.9 406.4,18498.7 407.6,18498.6 408.8,18498.5 410,18498.4 411.2,18498.2 412.4,18498.1 413.6,18498 414.8,18497.9 416,18497.7 417.2,18497.6 418.4,18497.5 419.6,18497.3 420.8,18497.2 422,18497.1 423.2,18496.9 424.4,18496.8 425.6,18496.7 426.8,18496.5 428,18496.4 429.2,18496.3 430.4,18496.1 431.6,18496 432.8,18495.9 434,18495.7 435.2,18495.6 436.4,18495.4 437.6,18495.3 438.8,18495.2 440,18495 441.2,18494.9 442.4,18494.7 443.6,18494.6 444.8,18494.5 446,18494.3 447.2,18494.2 448.4,18494 449.6,18493.9 450.8,18493.7 452,18493.6 453.2,18493.5 454.4,18493.3 455.6,18493.2 456.8,18493 458,18492.9 459.2,18492.7 460.4,18492.6 461.6,18492.4 462.8,18492.3 464,18492.1 465.2,18492 466.4,18491.8 467.6,18491.7 468.8,18491.5 470,18491.4 471.2,18491.2 472.4,18491.1 473.6,18490.9 474.8,18490.8 476,18490.6 477.2,18490.5 478.4,18490.3 479.6,18490.2 480.8,18490 482,18489.9 483.2,18489.7 484.4,18489.6 485.6,18489.4 486.8,18489.3 488,18489.1 489.2,18488.9 490.4,18488.8 491.6,18488.6 492.8,18488.5 494,18488.3 495.2,18488.2 496.4,18488 497.6,18487.8 498.8,18487.7 500,18487.5 501.2,18487.4 502.4,18487.2 503.6,18487.1 504.8,18486.9 506,18486.7 507.2,18486.6 508.4,18486.4 509.6,18486.3 510.8,18486.1 512,18485.9" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<line x1="286.1" y1="18506.4" x2="286.1" y2="18578" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="286.1" cy="18506.4" r="4" style="fill:var(--c-danger)"/>
<text x="286.1" y="18592" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700;font-family:var(--font-mono)">42.9</text>
<polyline points="87.2,18474.4 88.4,18488.8 89.6,18499.5 90.8,18507.9 92,18514.5 93.2,18519.9 94.4,18524.3 95.6,18528 96.8,18531.2 98,18533.9 99.2,18536.3 100.4,18538.3 101.6,18540.1 102.8,18541.7 104,18543.1 105.2,18544.4 106.4,18545.5 107.6,18546.5 108.8,18547.4 110,18548.2 111.2,18549 112.4,18549.6 113.6,18550.2 114.8,18550.8 116,18551.3 117.2,18551.7 118.4,18552.1 119.6,18552.5 120.8,18552.8 122,18553.2 123.2,18553.4 124.4,18553.7 125.6,18553.9 126.8,18554.1 128,18554.3 129.2,18554.5 130.4,18554.6 131.6,18554.8 132.8,18554.9 134,18555 135.2,18555.1 136.4,18555.1 137.6,18555.2 138.8,18555.3 140,18555.3 141.2,18555.3 142.4,18555.4 143.6,18555.4 144.8,18555.4 146,18555.4 147.2,18555.4 148.4,18555.3 149.6,18555.3 150.8,18555.3 152,18555.3 153.2,18555.2 154.4,18555.2 155.6,18555.1 156.8,18555.1 158,18555 159.2,18554.9 160.4,18554.9 161.6,18554.8 162.8,18554.7 164,18554.6 165.2,18554.6 166.4,18554.5 167.6,18554.4 168.8,18554.3 170,18554.2 171.2,18554.1 172.4,18554 173.6,18553.9 174.8,18553.8 176,18553.7 177.2,18553.5 178.4,18553.4 179.6,18553.3 180.8,18553.2 182,18553.1 183.2,18552.9 184.4,18552.8 185.6,18552.7 186.8,18552.6 188,18552.4 189.2,18552.3 190.4,18552.2 191.6,18552 192.8,18551.9 194,18551.7 195.2,18551.6 196.4,18551.5 197.6,18551.3 198.8,18551.2 200,18551 201.2,18550.9 202.4,18550.7 203.6,18550.6 204.8,18550.4 206,18550.3 207.2,18550.1 208.4,18550 209.6,18549.8 210.8,18549.7 212,18549.5 213.2,18549.3 214.4,18549.2 215.6,18549 216.8,18548.9 218,18548.7 219.2,18548.5 220.4,18548.4 221.6,18548.2 222.8,18548 224,18547.9 225.2,18547.7 226.4,18547.5 227.6,18547.4 228.8,18547.2 230,18547 231.2,18546.9 232.4,18546.7 233.6,18546.5 234.8,18546.4 236,18546.2 237.2,18546 238.4,18545.8 239.6,18545.7 240.8,18545.5 242,18545.3 243.2,18545.1 244.4,18545 245.6,18544.8 246.8,18544.6 248,18544.4 249.2,18544.3 250.4,18544.1 251.6,18543.9 252.8,18543.7 254,18543.6 255.2,18543.4 256.4,18543.2 257.6,18543 258.8,18542.8 260,18542.7 261.2,18542.5 262.4,18542.3 263.6,18542.1 264.8,18541.9 266,18541.7 267.2,18541.6 268.4,18541.4 269.6,18541.2 270.8,18541 272,18540.8 273.2,18540.6 274.4,18540.5 275.6,18540.3 276.8,18540.1 278,18539.9 279.2,18539.7 280.4,18539.5 281.6,18539.3 282.8,18539.2 284,18539 285.2,18538.8 286.4,18538.6 287.6,18538.4 288.8,18538.2 290,18538 291.2,18537.8 292.4,18537.7 293.6,18537.5 294.8,18537.3 296,18537.1 297.2,18536.9 298.4,18536.7 299.6,18536.5 300.8,18536.3 302,18536.1 303.2,18535.9 304.4,18535.8 305.6,18535.6 306.8,18535.4 308,18535.2 309.2,18535 310.4,18534.8 311.6,18534.6 312.8,18534.4 314,18534.2 315.2,18534 316.4,18533.8 317.6,18533.6 318.8,18533.5 320,18533.3 321.2,18533.1 322.4,18532.9 323.6,18532.7 324.8,18532.5 326,18532.3 327.2,18532.1 328.4,18531.9 329.6,18531.7 330.8,18531.5 332,18531.3 333.2,18531.1 334.4,18530.9 335.6,18530.7 336.8,18530.5 338,18530.4 339.2,18530.2 340.4,18530 341.6,18529.8 342.8,18529.6 344,18529.4 345.2,18529.2 346.4,18529 347.6,18528.8 348.8,18528.6 350,18528.4 351.2,18528.2 352.4,18528 353.6,18527.8 354.8,18527.6 356,18527.4 357.2,18527.2 358.4,18527 359.6,18526.8 360.8,18526.6 362,18526.4 363.2,18526.2 364.4,18526 365.6,18525.8 366.8,18525.6 368,18525.4 369.2,18525.2 370.4,18525 371.6,18524.8 372.8,18524.6 374,18524.5 375.2,18524.3 376.4,18524.1 377.6,18523.9 378.8,18523.7 380,18523.5 381.2,18523.3 382.4,18523.1 383.6,18522.9 384.8,18522.7 386,18522.5 387.2,18522.3 388.4,18522.1 389.6,18521.9 390.8,18521.7 392,18521.5 393.2,18521.3 394.4,18521.1 395.6,18520.9 396.8,18520.7 398,18520.5 399.2,18520.3 400.4,18520.1 401.6,18519.9 402.8,18519.7 404,18519.5 405.2,18519.3 406.4,18519.1 407.6,18518.9 408.8,18518.7 410,18518.5 411.2,18518.3 412.4,18518.1 413.6,18517.9 414.8,18517.7 416,18517.5 417.2,18517.3 418.4,18517.1 419.6,18516.9 420.8,18516.7 422,18516.5 423.2,18516.3 424.4,18516.1 425.6,18515.9 426.8,18515.7 428,18515.5 429.2,18515.3 430.4,18515.1 431.6,18514.9 432.8,18514.7 434,18514.5 435.2,18514.3 436.4,18514.1 437.6,18513.9 438.8,18513.7 440,18513.5 441.2,18513.3 442.4,18513 443.6,18512.8 444.8,18512.6 446,18512.4 447.2,18512.2 448.4,18512 449.6,18511.8 450.8,18511.6 452,18511.4 453.2,18511.2 454.4,18511 455.6,18510.8 456.8,18510.6 458,18510.4 459.2,18510.2 460.4,18510 461.6,18509.8 462.8,18509.6 464,18509.4 465.2,18509.2 466.4,18509 467.6,18508.8 468.8,18508.6 470,18508.4 471.2,18508.2 472.4,18508 473.6,18507.8 474.8,18507.6 476,18507.4 477.2,18507.2 478.4,18507 479.6,18506.8 480.8,18506.6 482,18506.4 483.2,18506.2 484.4,18506 485.6,18505.8 486.8,18505.6 488,18505.4 489.2,18505.2 490.4,18505 491.6,18504.8 492.8,18504.5 494,18504.3 495.2,18504.1 496.4,18503.9 497.6,18503.7 498.8,18503.5 500,18503.3 501.2,18503.1 502.4,18502.9 503.6,18502.7 504.8,18502.5 506,18502.3 507.2,18502.1 508.4,18501.9 509.6,18501.7 510.8,18501.5 512,18501.3" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<line x1="145.2" y1="18555.4" x2="145.2" y2="18578" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="145.2" cy="18555.4" r="4" style="fill:var(--c-success)"/>
<text x="145.2" y="18592" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">13.6</text>
<text x="80.0" y="18592" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="224.0" y="18592" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">30</text>
<text x="368.0" y="18592" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">60</text>
<text x="512.0" y="18592" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">90</text>
<text x="512" y="18608" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">checkpoint interval τ (minutes)</text>
<text x="40" y="18450" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">loss</text>
<rect x="300" y="18457" width="12" height="10" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1"/>
<text x="318" y="18466" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">A · 25 GB/s: at best 24% lost</text>
<rect x="300" y="18474" width="12" height="10" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1"/>
<text x="318" y="18483" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">B · 250 GB/s (GDS): at best 7.5%</text>
<text x="353.6" y="18546.5" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">rewind τ/2M (shared)</text>
<text x="40" y="18626" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">M = 3 h (on 16,384 H100s a failure every ~3 h) · loss = δ/τ + τ/2M</text>
<text x="40" y="18642" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">How to read it: frequent saves lose time to writing, rare saves lose it to rewinding;</text>
<text x="40" y="18657" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">the best interval is the bottom of the dip.</text>
<text x="552" y="18462" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Symptom</text>
<text x="552" y="18480" text-anchor="start" style="fill:var(--c-text);font-size:12px">The cluster stops on every save; a</text>
<text x="552" y="18497" text-anchor="start" style="fill:var(--c-text);font-size:12px">failure rewinds to the last one.</text>
<text x="552" y="18525" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Cause</text>
<text x="552" y="18543" text-anchor="start" style="fill:var(--c-text);font-size:12px">Writing 7.5 TB through the CPU page</text>
<text x="552" y="18560" text-anchor="start" style="fill:var(--c-text);font-size:12px">cache at 25 GB/s takes δ = 307 s, and</text>
<text x="552" y="18577" text-anchor="start" style="fill:var(--c-text);font-size:12px">the cluster waits the whole time.</text>
<text x="552" y="18605" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Fix</text>
<rect x="552.0" y="18615" width="116.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="610.4" y="18629" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPUDirect Storage</text>
<rect x="552.0" y="18641" width="135.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="619.5" y="18655" text-anchor="middle" style="fill:var(--c-text);font-size:11px">async / burst writes</text>
<rect x="693.0" y="18641" width="104.8" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="745.4" y="18655" text-anchor="middle" style="fill:var(--c-text);font-size:11px">pick τ = √(2δM)</text>
<line x1="540.0" y1="18448" x2="540.0" y2="18661" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="18687" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="18707" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: the three fixes share one idea: move less data, and keep computing while you move it.</text>
<text x="420.0" y="18754" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Let&#x27;s feel these gaps: push the same load through every road.</text>
<line x1="420.0" y1="18762" x2="420.0" y2="18782" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="18786" width="808" height="333" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="18810" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="18814.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">D</text>
<text x="56" y="18815" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Speed intuition: how long does 140 GB take?</text>
<text x="28" y="18836" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Llama 3 70B&#x27;s BF16 weights = 140 GB; the same load on every link (log scale).</text>
<rect x="28" y="18848" width="784" height="214" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="200.0" y1="18862" x2="200.0" y2="18986" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="200" y="19000" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 ms</text>
<line x1="292.0" y1="18862" x2="292.0" y2="18986" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="292" y="19000" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 ms</text>
<line x1="384.0" y1="18862" x2="384.0" y2="18986" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="384" y="19000" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 s</text>
<line x1="476.0" y1="18862" x2="476.0" y2="18986" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="476" y="19000" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 s</text>
<text x="40" y="18880" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3 (one H100)</text>
<rect x="200" y="18868" width="57.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="263.1" y="18880" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">42 ms</text>
<text x="40" y="18904" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (in node)</text>
<rect x="200" y="18892" width="137.3" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="343.3" y="18904" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0.31 s</text>
<text x="40" y="18928" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU path)</text>
<rect x="200" y="18916" width="215.3" height="16" rx="2" style="fill:var(--c-text-mute);fill-opacity:.45;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="421.3" y="18928" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2.2 s</text>
<text x="40" y="18952" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (per GPU)</text>
<rect x="200" y="18940" width="225.1" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="431.1" y="18952" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2.8 s</text>
<text x="40" y="18976" text-anchor="start" style="fill:var(--c-text);font-size:11px">legacy storage (cluster)</text>
<rect x="200" y="18964" width="252.8" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="458.8" y="18976" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5.6 s</text>
<text x="40" y="19028" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Longer bar = slower: each bar is the time to push the same 140 GB through that link.</text>
<rect x="544.0" y="18862" width="256.0" height="80" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554" y="18880" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Human scale: data that arrives from</text>
<text x="554" y="18896" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">HBM in 1 second</text>
<text x="554" y="18912" text-anchor="start" style="fill:var(--c-text);font-size:11px">takes 7.4 s over NVLink, 52 s over PCIe</text>
<text x="554" y="18928" text-anchor="start" style="fill:var(--c-text);font-size:11px">and 67 s over InfiniBand.</text>
<rect x="544.0" y="18952" width="256.0" height="96" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="554" y="18970" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Decode&#x27;s ceiling: every token reads</text>
<text x="554" y="18986" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">all the weights once</text>
<text x="554" y="19002" text-anchor="start" style="fill:var(--c-text);font-size:11px">on 8 H100s, 140 GB ÷</text>
<text x="554" y="19018" text-anchor="start" style="fill:var(--c-text);font-size:11px">(8 × 3.35 TB/s) ≈ 5.2 ms → at most</text>
<text x="554" y="19034" text-anchor="start" style="fill:var(--c-text);font-size:11px">~190 tokens/s at batch 1</text>
<line x1="532.0" y1="18862" x2="532.0" y2="19048" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="19074" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="19094" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Remember: milliseconds in HBM, seconds on the network. Good design keeps data in the fast layer.</text>
<text x="420.0" y="19139" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Last stop: what does all this look like in a real cluster?</text>
<line x1="420.0" y1="19147" x2="420.0" y2="19167" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="19171" width="808" height="310" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="19195" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="19199.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">E</text>
<text x="56" y="19200" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">A real cluster: Meta&#x27;s Llama 3 405B run</text>
<text x="28" y="19221" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">The same 16,384 H100s in Meta&#x27;s own RoCE cluster; numbers from the Llama 3 report.</text>
<rect x="28.0" y="19233" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="19241" width="237.3" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.7" y="19255" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Compute</text>
<text x="42.0" y="19279" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16,384 H100s</text>
<text x="42.0" y="19294" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TP 8 × PP 16 × DP 128</text>
<text x="42.0" y="19309" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16M tokens per step</text>
<text x="42.0" y="19324" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">400 TFLOP/s per GPU</text>
<text x="42.0" y="19339" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BF16 MFU 41%</text>
<rect x="293.33333333333337" y="19233" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="301.3" y="19241" width="237.3" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="19255" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Network</text>
<text x="307.3" y="19279" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">RoCE · 400 Gb/s/GPU</text>
<text x="307.3" y="19294" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod = 3,072 GPUs</text>
<text x="307.3" y="19309" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">full bisection in pod</text>
<text x="307.3" y="19324" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1:7 between 8 pods</text>
<text x="307.3" y="19339" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(smaller models: InfiniBand)</text>
<rect x="558.6666666666667" y="19233" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="566.7" y="19241" width="237.3" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="685.3" y="19255" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Storage</text>
<text x="572.7" y="19279" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Tectonic · 240 PB</text>
<text x="572.7" y="19294" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7,500 SSD servers</text>
<text x="572.7" y="19309" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 TB/s sustained</text>
<text x="572.7" y="19324" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7 TB/s peak</text>
<text x="572.7" y="19339" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ckpt: 1 MB–4 GB/GPU</text>
<text x="40" y="19375" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">MFU: 43% on 8,192 H100s, 41% on 16,384, 38% at 128K context; scale makes comms grow.</text>
<rect x="28.0" y="19389" width="386.0" height="80" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="38" y="19407" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">466 interruptions in 54 days, 419 unexpected → a failure</text>
<text x="38" y="19423" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">every ~3 hours</text>
<text x="38" y="19439" text-anchor="start" style="fill:var(--c-text);font-size:11px">78% hardware (GPU 30.1% · HBM3 17.2%); automation kept effective</text>
<text x="38" y="19455" text-anchor="start" style="fill:var(--c-text);font-size:11px">time &gt; 90%</text>
<rect x="426.0" y="19389" width="386.0" height="80" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="436.0" y="19407" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Lesson: the three pillars are designed together.</text>
<text x="436.0" y="19424" text-anchor="start" style="fill:var(--c-text);font-size:11px">The slowest link (HBM, network or disk) sets the whole cluster&#x27;s</text>
<text x="436.0" y="19440" text-anchor="start" style="fill:var(--c-text);font-size:11px">pace; MFU mirrors it.</text>
</svg>

*Figure 1.1: The three pillars at a glance. A: the SM → GPU → DGX node (with its system topology) → SuperPOD scale ladder, a closer look at the SuperPOD's network (port count, traffic paths, scale) and the 14 building blocks that keep the cluster running; B: the path of data through one training step; C: the three walls with computed curves; D: speed intuition with a 140 GB load; E: real numbers from Llama 3 405B. Sources: NVIDIA DGX H100 and DGX SuperPOD documentation, The Llama 3 Herd of Models (2024).*

### How the poster is organized

The poster organizes the entire hardware architecture into four logical panels, followed by Meta's empirical datacenter benchmark:

| Section | Focus | Hardware Components and Metrics Analyzed |
| :--- | :--- | :--- |
| **A: Scale Ladder** | Microscopic silicon to cluster | SM subpartitions, H100 chip layout, DGX node NUMA topology, SuperPOD fat-tree |
| **B: Execution Step** | Single training iteration | Step timeline, activation communication, gradient sync, and hardware execution |
| **C: Hardware Gaps** | Physical hardware constraints | HBM arithmetic intensity threshold, NVLink vs network limits, I/O direct transfer |
| **D: Latency Hierarchy** | 140 GB transit across interconnects | Latency benchmarks: HBM3 (42 ms) → NVLink (0.31 s) → PCIe (2.2 s) → InfiniBand (2.8 s) → Legacy I/O (5.6 s) |
| **E: Datacenter Ground Truth** | Meta Llama 3 405B infrastructure | Empirical parameters: 16,384 H100s, 400 TFLOP/s sustained, 41% MFU, RoCE fabric, Tectonic storage |

Part A zooms out across four hardware levels: 132 SMs form an H100 GPU, 8 H100s form a DGX node, 32 DGX nodes form a scalable unit (SU), and 4 SUs compose a 127-node DGX SuperPOD. Dedicated cards detail the fabric architecture (leaf-to-spine non-blocking wiring, in-network reduction) and physical infrastructure building blocks (dedicated compute/storage/management fabrics, 1.39 MW datacenter provisioning).

Most cards follow a consistent pedagogical anatomy:
- **Hardware schematic** with numbered references linking directly to the mathematical breakdown.
- **First-principles derivations**, such as $990 	ext{ TFLOP/s} = 528 	ext{ Tensor Cores} 	imes 1{,}024 	ext{ FLOP} 	imes 1.83 	ext{ GHz}$.
- **Mental models** that scale hierarchically: from line cooks in an SM to interconnected kitchens across a datacenter.
- **Architectural rules:** "chattiest tensor parallelism stays inside the NVLink corridor; scale-out traffic is aligned to independent network rails."

The three curves in Part C represent the physical rooflines: the arithmetic intensity boundary, the interconnect bandwidth cliff, and direct DMA storage throughput. Part E anchors these calculations to Meta's published datacenter metrics.

> [!TIP]
> **Part 2 of the Series: System Diagnosis**
> While this article teaches physical hardware anatomy, the diagnostic counterpart explores why these machines stall under real pre-training workloads, analyzing MFU losses, cluster crashes, and operational bottlenecks: [AI Infrastructure (2): Diagnosing an AI Supercomputer — Why 1,024 H100s Stall at 38% MFU](post.html?slug=from-gpus-to-ai-supercomputers).

---

## 2. Pillar I — Compute: inside the SM and the H100 chip

This section is the legend for levels 1 and 2 of Part A and for the memory wall in Part C: what an SM does in one cycle, why it keeps 64 warps in flight, and why a batch of about 300 tokens separates a waiting GPU from a working one.

### Silicon-level GPU anatomy: SMs, warp schedulers, and tensor cores

Calling a GPU "a more powerful processor" misses the point. A CPU spends its silicon on a few sophisticated, low-latency cores built for branch-heavy, irregular and latency-sensitive instruction streams. A GPU spends most of its silicon on execution units, so that it can run an enormous number of similar operations at the same time. That trade wins for dense matrix multiplication and stops paying off when the work is tiny or irregular, because coordination overhead then eats the parallelism.

To grasp why GPUs behave so differently from TPUs, consider a commercial kitchen analogy. A Google TPU v5p behaves like a high-end French brigade run by just two colossal master chefs (the two TensorCore units). Each chef handles enormous cutting boards (Matrix Multiply Units, or MXUs) capable of processing 128x128 matrices in lockstep; if you hand them an irregular task or small batches, their massive tools sit idle. A modern NVIDIA GPU (such as Hopper H100 or Blackwell B200) is instead an army of 132 to 148 short-order line cooks, termed **Streaming Multiprocessors (SMs)**. Each line cook operates independently, executing separate tasks on modest prep stations, yet all sharing a central pantry.

In hardware, an SM is broken down into four identical quadrants called **SM subpartitions** (SMSPs). Each subpartition contains the four parts numbered 1–4 on level 1 of the poster:
1. **A Tensor Core (TC):** a matrix multiply-accumulate engine. On an H100 SXM5 at its 1.83 GHz Tensor Core clock, each Tensor Core does 1,024 dense BF16 floating-point operations per cycle, the work of an 8×8×8 matrix product, and twice that (2,048) in FP8. On Blackwell B200, each Tensor Core does 2,048 BF16 FLOPs per cycle, reaching 2,250 TFLOP/s dense BF16.
2. **A Warp Scheduler and Dispatch Unit:** it issues one instruction per cycle to a group of 32 threads, collectively known as a **warp**, choosing among up to 16 resident warps. With four schedulers an SM keeps up to 64 warps in flight and switches between them every cycle to hide memory load latencies.
3. **32 FP32 CUDA Cores (ALUs)**, plus 16 INT32 and 16 FP64 units: general-purpose vector execution units that compute elementwise activations (GELU, SiLU), normalizations (RMSNorm), and additions. Across the 132 SMs of an H100, there are 16,896 CUDA cores running at ~1.98 GHz boost clock.
4. **16,384 32-bit Registers:** 64 KiB of register file per subpartition (256 KiB per SM, ~33 MB chip-wide), where every thread's variables live and from which every instruction reads its operands.

The four subpartitions share 256 KiB of L1 data cache and shared memory (SMEM), up to 228 KiB of which can be reserved as SMEM, and the **Tensor Memory Accelerator (TMA)** copies tiles from L2 or HBM into SMEM asynchronously. Multiplying the per-cycle work out gives the headline number:

$$C_{\text{peak}} = N_{\text{TC}} \times \text{FLOP/cycle} \times f = 528 \times 1{,}024 \times 1.83 \times 10^{9} \approx 989 \times 10^{12} \text{ FLOP/s}$$

Read this as: $N_{\text{TC}} = 132 \times 4 = 528$ Tensor Cores, each doing 1,024 BF16 FLOPs per cycle, at a clock $f$ of 1.83 GHz. That is the "990 TFLOP/s" used everywhere in this article; FP8 doubles it to ~1,979 TFLOP/s.

These figures belong to the SXM module; the PCIe version of the H100 has 114 SMs. Inside an SM the four kinds of parts have clearly separate jobs:

| Unit | Primary function | In the kitchen model |
| :--- | :--- | :--- |
| **Warp scheduler** | issues the next instruction of a warp whose operands are ready | the order rail in front of the cook |
| **CUDA core / ALU** | additions, multiplications, elementwise ops and general arithmetic | the general workbench |
| **Tensor Core** | fused matrix multiply-accumulate, $D = A \times B + C$, on small tiles | the knife and stove |
| **Registers / shared memory** | the operands in active use, at the lowest latency | the counter |

The linear projections and feed-forward layers of a Transformer are dense matrix multiplications, so almost all of the 990 TFLOP/s comes from the Tensor Cores. A CUDA core and a Tensor Core are different physical units, which is why `nvidia-smi` reporting 99% GPU utilization says nothing about whether the Tensor Cores are being used efficiently, or at all.

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

### SIMT execution dynamics: Thread, warp, and block hierarchy

Unlike the TPU's pure SIMD (Single Instruction, Multiple Data) execution, NVIDIA GPUs operate under the **SIMT (Single Instruction, Multiple Threads)** paradigm. While all 32 CUDA cores within a warp execute the same instruction in a given cycle, each thread maintains its own program counter and private register space.

The units nest. A **thread** is the smallest logical instance of a program: you write one program, for example a blur filter, and the GPU launches thousands of copies of it, one per pixel. Threads are grouped into **thread blocks**, and the hardware cuts each block into **warps** of 32 threads that are scheduled together. A block runs on a single SM, and one SM can host several blocks at once as long as its registers, shared memory and warp slots last. SIMT means that every thread keeps its own registers and state, while the fetch and dispatch of each instruction is shared by the whole warp: 32 clerks applying the same formula to 32 rows of a spreadsheet.

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

This flexibility introduces a perilous hardware penalty known as **warp divergence**. If your kernel contains a conditional branch (such as an `if/else` condition checking for MoE expert routing assignment or dynamic sequence padding), threads taking the branch cannot proceed independently. Instead, the warp scheduler executes the `if` path while masking out (stalling) the remaining cores, and then executes the `else` path while stalling the first set:

If 16 threads take branch A and 16 take branch B and both branches cost the same, each pass leaves half of the lanes idle and that warp's throughput halves. The 50% belongs to this equal-cost case and is not a universal rule: the real penalty depends on the length of each branch, on how much independent work the instructions contain and on how the compiler schedules them.

### Latency hiding and occupancy: why an SM keeps 64 warps

Divergence wastes lanes inside one warp; the bigger loss is a scheduler with no warp to issue at all. In the kitchen model, a cook who has sent for tomatoes from the pantry does not stand and wait: they chop onions for another order. The orders are warps, the counter in front of the cook is the register file and SMEM, and the pantry is HBM. Measured on Hopper, one load takes about 29 cycles from SMEM, 41 from L1, 263 from L2 and 479 from HBM, so a trip to HBM costs as much time as one Tensor Core needs for ~490 thousand FLOPs. Because every resident warp's registers already sit on the chip, switching to another warp costs nothing; there is no state to save and restore as in a CPU context switch.

How many warps does a scheduler need so that it is never idle? If each warp computes for $t_c$ cycles and then waits $t_w$ cycles for memory, the others must fill the wait:

$$N_{\text{warps}} \approx 1 + \frac{t_w}{t_c}$$

Read this as: one warp computing now, plus enough warps to cover the waiting time of that warp in units of its compute time. In the poster's toy timeline each warp computes for 2 units and waits 6, so $1 + 6/2 = 4$ warps keep the scheduler busy, while 2 warps leave it idle half the time.

The pool of warps is limited by the register file of item 4: 65,536 registers per SM, shared by every resident thread. A kernel that wants $R$ registers per thread fits

$$\text{warps per SM} = \min\left(64, \frac{65{,}536}{32 \cdot R}\right)$$

```text
registers/thread   warps per SM          occupancy
  32               65,536 / 1,024 = 64   100%
  64               65,536 / 2,048 = 32    50%
 128               65,536 / 4,096 = 16    25%
 255 → 256         65,536 / 8,192 =  8    12.5%
```

Occupancy in profiling tools is exactly this ratio. Few warps are not always bad, though. Hopper's fastest GEMM and FlashAttention-3 kernels deliberately run few warps with many registers and hide latency differently: producer warps start TMA copies from HBM into SMEM ahead of time while consumer warps compute on the Tensor Cores with `wgmma`. Either way the rule is the same: while one piece of work waits for memory, another must be ready to run.

Three ideas are often blurred at this point. **Concurrency** means that many tasks are in progress in the same window. **Parallel execution** means that different hardware engines, several SMs or Tensor Cores, execute in the very same cycle. **Latency hiding** means that a scheduler interleaves ready warps while others wait for memory. High occupancy only says that the scheduler has many candidates to choose from; it does not say that the Tensor Cores are producing useful FLOPs.

The warp counts in the table are upper bounds from the register budget alone. Shared memory per block, block size and allocation granularity cut them further, which is why the occupancy a profiler reports is usually lower, and the limits interact non-linearly. A kernel with 256-thread blocks that fits only one block per SM may hold more active warps with 128-thread blocks, yet that does not guarantee a faster kernel, because data reuse across the block can drop and scheduling contention can rise. Launching "more threads" without first profiling register spills, shared memory, divergence and memory access patterns often makes a kernel slower, not faster.

### The H100 chip: 132 SMs, L2 cache, and HBM3 stacks

Level 2 of the poster maps the SXM5 package. The GH100 die is 814 mm² with 80 billion transistors on TSMC 4N, built as 8 GPCs of 9 TPCs with 2 SMs each: 144 SM slots, of which the H100 SXM5 enables 132. A die this large rarely comes out flawless, so switching a few units off raises the yield; which six TPCs are off varies from chip to chip. The 50 MB L2 is split into two partitions, and the GPCs on each side sit closer to one of them. Five of the package's six HBM3 sites are active, each a 16 GB stack behind two 512-bit memory controllers, and the bandwidth follows from the bus width:

$$W_{\text{HBM}} = \frac{5 \times 1{,}024 \text{ bit} \times 5.23 \text{ Gb/s}}{8 \text{ bit/byte}} \approx 3.35 \text{ TB/s}$$

Read this as: five stacks with a 1,024-bit interface each give a 5,120-bit bus; at ~5.23 Gb/s per pin this moves 3.35 TB per second, about 670 GB/s per stack. The full GH100 has 144 SMs, 60 MB of L2 and six stacks; the board power limit of the SXM5 module is 700 W.

Computation is meaningless without the memory bandwidth required to feed the execution units. The GPU memory subsystem forms a steep pyramid, small and fast at the top, large and slow at the bottom:
- **Register File:** 256 KiB per SM (~33 MB chip-wide). Operands are read directly by the instruction, with no load at all.
- **Shared Memory (SMEM) / L1 Cache:** 256 KiB per SM (~33 MB chip-wide). Programmatically managed SRAM with ~29-cycle (~16 ns) SMEM and ~41-cycle L1 latency, critical for caching attention tiles in FlashAttention.
- **Tensor Memory (TMEM):** Introduced in Blackwell B200 (256 KiB per SM). Because B200 Tensor Cores produce matrix accumulators so massive they exceeded traditional register capacity, TMEM provides dedicated on-chip storage for matrix products.
- **Shared L2 Cache:** 50 MB on Hopper H100 (~263 cycles, roughly 5.5 TB/s), 126 MB on Blackwell B200.
- **High Bandwidth Memory (HBM3 / HBM3e):** 80 GB on H100 (~479 cycles, 3.35 TB/s), 141 GB on H200 (4.8 TB/s), and 192 GB on B200 (8.0 TB/s).

Going from L2 to HBM, capacity grows 1,600 times while latency almost doubles. The SRAM inside the SMs (~33 MB of registers plus ~33 MB of SMEM/L1) is larger in total than L2, which is why good kernels keep their working set in these top levels and go to HBM as rarely as possible.

Two different questions hide in the word "fast". **Bandwidth** is how many bytes per second a path carries; **latency** is how long the first byte of a request takes to arrive. An eight-lane highway 500 km long has high bandwidth and high latency at the same time. Streaming large contiguous tensors is governed by bandwidth; pointer chasing, irregular lookups and fine-grained dependencies are governed by latency. Physical distance is only part of it: HBM sits on the same package as the die, yet for a warp an operand in a register and one in HBM are worlds apart, because protocol overhead, arbitration, bank conflicts, cache hits and transfer width all add to the cost of "near".

Bandwidth also has to be earned by the access pattern. **Memory coalescing** means that the 32 threads of a warp request adjacent, aligned addresses, so the memory controller can serve the whole warp with a few wide 32- or 128-byte transactions; scattered or misaligned accesses waste bus cycles on bytes nobody uses. And a peak is not a sustained rate: decimal GB is not binary GiB, and no kernel streams the full 3.35 TB/s.

The balance between compute capability and memory throughput is formalized by the Roofline Model:

$$\alpha_{\text{crit}} = \frac{C_{\text{peak}}}{W_{\text{mem}}}$$

Read this as: $\alpha_{\text{crit}}$ is the critical arithmetic intensity in FLOPs per byte, $C_{\text{peak}}$ is peak arithmetic throughput in FLOPs per second, and $W_{\text{mem}}$ is memory bandwidth in bytes per second. A kernel with intensity $\alpha$ runs at $\min($C_{\text{peak}} / W_{\text{mem}} \approx 295$ FLOP/byte)$, the roof drawn on level 2 of the poster.

For an H100 SXM5 running BF16 Tensor Core operations:
$$C_{\text{peak}} = 990 \times 10^{12} \text{ FLOP/s}, \quad W_{\text{mem}} = 3.35 \times 10^{12} \text{ B/s}$$
$$\alpha_{\text{crit}} = \frac{990 \times 10^{12}}{3.35 \times 10^{12}} \approx 295.5 \text{ FLOP/byte}$$

This number is the machine balance of dense BF16 Tensor Core math against HBM, not a rule that every kernel must do 295 FLOPs per byte. Change the data type and the knee moves with the peak: FP8 Tensor Cores at ~1,979 TFLOP/s put it near $1{,}979 / 3.35 \approx 591$ FLOP/byte, while FP32 on the CUDA cores puts it far lower.

For a matrix multiplication multiplying an activation matrix $X \in \mathbb{R}^{B \times D}$ by a weight matrix $W \in \mathbb{R}^{D \times F}$ in 16-bit precision:
- **Total Math Operations:** $2 B D F \text{ FLOPs}$
- **Total Memory Transferred:** $2(BD + DF + BF) \text{ bytes}$
- **Operational Intensity:**

$$\alpha(B) = \frac{2 B D F}{2(BD + DF + BF)} = \frac{B D F}{B(D + F) + D F}$$

When the batch size $B$ is tiny ($B \to 1$, such as single-sequence auto-regressive decoding), the weight matrix dominates memory traffic ($DF \gg B(D+F)$), yielding:
$$\alpha(1) \approx \frac{D F}{D F} \cdot B \approx B = 1 \text{ FLOP/byte} \ll 295.5 \text{ FLOP/byte}$$

In this regime, the GPU is severely memory-bandwidth bound, utilizing less than 1% of its peak compute capability. To become compute-bound, the batch size must satisfy:

$$B \ge \frac{\alpha_{\text{crit}} \cdot D F}{D F - \alpha_{\text{crit}}(D + F)} \approx \alpha_{\text{crit}} \approx 295 \text{ tokens}$$

The poster's roofline and memory-wall curve use the Llama 3 70B MLP sizes, $D = 8{,}192$ and $F = 28{,}672$. Walking the batch up shows the three regimes:

```text
B = 1      α = 8,192·28,672 / (1·36,864 + 234.9M)     ≈ 1.0     → 3.35 × 1.0   ≈   3.3 TFLOP/s  (0.3%)
B = 8      α = 1.88G / (0.29M + 234.9M)               ≈ 8.0     → 3.35 × 8.0   ≈  26.8 TFLOP/s  (2.7%)
B = 32     α = 7.52G / (1.18M + 234.9M)               ≈ 31.8    → 3.35 × 31.8  ≈ 106.7 TFLOP/s  (11%)
B = 128    α = 30.06G / (4.72M + 234.9M)              ≈ 125.5   → 3.35 × 125.5 ≈ 420   TFLOP/s  (42%)
B = 310    α = 72.8G / (11.4M + 234.9M)               ≈ 295.6   → at the knee      ≈ 990   TFLOP/s
B = 512    α = 120.3G / (18.9M + 234.9M)              ≈ 473.9   → under the roof   = 990   TFLOP/s
B = 4,096  α = 962G / (151M + 234.9M)                 ≈ 2,493   → at the roof      = 990   TFLOP/s
```

On both Hopper H100 and Blackwell B200, a matrix multiply therefore needs a batch of roughly 280 to 300 tokens to cross the roofline knee into the compute-bound zone, about 310 for these Llama 3 70B shapes. For comparison, a BF16 elementwise addition does 1 FLOP per 6 bytes ($\alpha \approx 0.17$), which is why such operations are fused into neighbouring kernels rather than run alone.

These points come from an idealized data-movement model, not from profiled kernels: tiling, L2 residency, register reuse, activation layouts and quantization all shift the real inflection points. The model is still enough to sort workloads into three families:

| Workload | Weight or data reuse | Expected bottleneck |
| :--- | :--- | :--- |
| Large matrix-vector multiply (GEMV, $B = 1$) | near zero | HBM bandwidth |
| Batched matrix-matrix multiply (GEMM, large $B$) | high | Tensor Cores (compute) |
| Chains of elementwise kernels | low, fragmented | kernel launches and repeated HBM round trips |

Faster HBM helps all three, but the largest gains come from algorithms that fetch data into on-chip SRAM once and reuse it many times before writing it back, which is the idea behind kernel fusion and FlashAttention.

### Low-precision numerical formats: From FP32 to FP8 and NVFP4

To mitigate memory pressure and double arithmetic throughput, modern GPU generations have aggressively reduced numerical precision. Moving from FP32 (32 bits: 1 sign, 8 exponent, 23 mantissa) to 16-bit formats (FP16 and BF16) halved memory footprint and doubled FLOP rates.

The historical evolution of Tensor Core compute engines across architecture generations is summarized below. The last column applies the roofline formula to each generation, and it explains the memory wall of section 1: compute has grown faster than bandwidth, so the knee has moved up.

| Architecture Generation | Compute Engine | HBM Memory & Bandwidth | BF16 / FP16 TFLOP/s | FP8 / INT8 TFLOP/s | FP4 TFLOP/s | Knee $\alpha_{\text{crit}}$ (16-bit) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Volta (V100)** | 1st Gen Tensor Core | 32 GB @ 0.9 TB/s | 125 | — | — | ~139 |
| **Ampere (A100)** | 3rd Gen Tensor Core | 80 GB @ 2.0 TB/s | 312 | 624 | — | ~156 |
| **Hopper (H100)** | 4th Gen Tensor Core | 80 GB @ 3.35 TB/s | 990 | 1,980 | — | ~295 |
| **Hopper (H200)** | 4th Gen Tensor Core | 141 GB @ 4.8 TB/s | 990 | 1,980 | — | ~206 |
| **Blackwell (B200)** | 5th Gen Tensor Core | 192 GB @ 8.0 TB/s | 2,250 | 4,500 | 9,000 | ~281 |

All throughput figures are dense (without 2:4 sparsity). The DGX B200 system exposes 180 GB of the B200's HBM3e per GPU. H200 keeps H100's compute but adds 43% more bandwidth, which is why the same decode workload runs faster on it.

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

With Blackwell B200, NVIDIA went one step further with **NVFP4**, a block-scaled 4-bit format. Each element is an E2M1 number; every block of 16 elements shares one FP8 (E4M3) scale factor, and a per-tensor FP32 scale sits on top. The small blocks let the scale follow local outliers, which is what makes 4 bits usable at all. B200 runs these at 9,000 TFLOP/s dense, twice its FP8 rate and four times its BF16 rate, while every weight takes a quarter of the bytes it needs in BF16, which also helps decode, where bytes rather than FLOPs set the speed.

---

## 3. Pillar II — Network: from the NVLink chassis to the SuperPOD fabric

This section is the legend for levels 3, 4 and 4b of Part A, for Part B and for the network wall in Part C. Its single idea is the bandwidth staircase: per GPU, 3.35 TB/s in HBM, 450 GB/s on NVLink and 50 GB/s on the network. Every parallelism decision below is a decision about which step of the staircase a piece of traffic is allowed to use.

### Intra-node networking: NVLink and NVSwitch fabrics

When scaling beyond a single chip, standard PCIe buses (delivering a paltry 64 GB/s on PCIe Gen 5) become an intolerable choke point. NVIDIA bypasses PCIe through **NVLink** and **NVSwitch**.

Inside a standard DGX H100 or HGX H100 chassis, 8 GPUs are paired with 4 custom NVSwitch Gen 4 ASICs on the HGX baseboard. Each H100 chip features 18 NVLink 4 ports. Each port transmits at 25 GB/s in each direction (50 GB/s bidirectional), aggregating to:
$$W_{\text{GPU egress}} = 18 \times 25 \text{ GB/s} = 450 \text{ GB/s unidirectional } (900 \text{ GB/s bidirectional})$$

Each NVSwitch chip has 64 NVLink ports, so it carries $64 \times 25 = 1.6$ TB/s in each direction (3.2 TB/s both ways). The 4 NVSwitches together form a non-blocking crossbar fabric with a total switching capacity of:
$$4 \text{ switches} \times 1.6 \text{ TB/s} = 6.4 \text{ TB/s switching bandwidth (one way)}$$

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

The poster's link matrix counts how the 18 links are spread. Every GPU sends 5, 4, 4 and 5 NVLinks to NVSwitches 0 to 3, so the switches receive 40, 32, 32 and 40 links: $8 \times 18 = 144$ in total. Because every GPU's full 450 GB/s lands on the switches, all eight GPUs can send at full speed at once, and any GPU reaches any other in a single switch hop. NVSwitch can also do the summing itself (**NVLink SHARP**), so in an AllReduce the data travels to the switch only once.

With the Blackwell generation, NVIDIA fundamentally altered this equation with the **GB200 NVL72**. Instead of confining the NVLink domain to an 8-GPU chassis, liquid-cooled racks integrate 72 B200 GPUs interconnected by 18 NVSwitch 5 ASICs over copper backplanes:

In NVL72, all 72 GPUs share a single NVLink domain where any chip can access any other chip's memory at a staggering 900 GB/s unidirectional rate, expanding the single-node boundary by 9x.

### Inside the DGX H100: two NUMA halves and four roads

NVLink is only one of the roads out of a GPU. Level 3 of the poster redraws NVIDIA's DGX H100 system topology, and its six numbered questions are worth answering in prose, because each one turns into a configuration rule.

1. **Who feeds whom?** The node is two mirror-image halves. CPU 0 (a 56-core Xeon 8480C) feeds GPUs 0–3 and network cards 0–3 over PCIe Gen5; CPU 1 feeds GPUs 4–7 and cards 4–7. Each half has its own memory, a **NUMA** domain, and the two sockets are joined by UPI. The process that prepares data for GPU 5 should therefore run on CPU 1 and use CPU 1's memory.
2. **How does a GPU reach the network?** Right above every GPU sits its own single-port 400 Gb/s ConnectX-7 card (GPU $i$ ↔ NIC $i$). With **GPUDirect RDMA**, data goes from GPU memory straight to the NIC and never stops in CPU memory. Eight such cards give $8 \times 50 = 400$ GB/s out of the chassis.
3. **How do GPUs talk to each other?** Over the 18 NVLinks to the four NVSwitches, as counted above.
4. **Where does storage traffic go?** Two upper PCIe switches carry two dual-port ConnectX-7 cards (storage fabric and in-band management) and four U.2 NVMe drives each: 8 × 3.84 TB in total, used as a local RAID 0 cache. Reading data and writing checkpoints therefore use different cards and cables from gradient traffic.
5. **What joins the halves?** UPI carries the CPUs' own traffic. GPUs in different halves never talk through it; they use NVLink.
6. **Who manages the NVSwitches?** A PCIe control path from CPU 0 reaches the four NVSwitches through a third PCIe switch. **NVIDIA Fabric Manager** sets up the switches' routing tables over it, and no GPU data flows on it. If Fabric Manager is not running, the NVLink fabric never comes up and multi-GPU jobs fail to start, even though every GPU looks healthy on its own.

Put together, one GPU has four roads, and every step of the training loop in Part B picks one of them:

| Road (per GPU, one way) | Bandwidth | Traffic that uses it | How often |
| :--- | :--- | :--- | :--- |
| NVLink through NVSwitch | 450 GB/s | Tensor-parallel AllReduce | twice per layer, forward and backward |
| Compute NIC → leaf switch $i$ | 50 GB/s | Pipeline activations, data-parallel gradients | per micro-batch, per step |
| Storage CX-7 → storage fabric | shared, ≥ 40 GB/s per node | Reading data, writing checkpoints | per step, every ~14 minutes |
| PCIe Gen5 to the CPU | 64 GB/s | Data loader, host copies | always, from the same socket |

The tensor-parallel rule of the next subsections gives the reason for the first row: NVLink is nine times wider than the network, so the traffic that talks in every layer stays inside these eight GPUs, and the chassis wall is also the tensor-parallelism wall.

Two precisions about these roads. PCIe Gen5 ×16 runs 16 lanes at 32 GT/s with 128b/130b encoding, about 63 GB/s per direction before protocol overhead, so the 64 GB/s on the poster is the rounded line rate. And **GPUDirect RDMA** is not **GPUDirect Storage**: RDMA is the NIC ↔ HBM path that carries gradients and activations between nodes, while GDS is the storage ↔ HBM path opened in section 4. Storage systems often use RDMA transports underneath, but the two address different subsystems.

The NVSwitch domain is also where **scale-up** ends. Inside it, GPUs form one tightly coupled, high-bandwidth domain; growing beyond it means **scale-out**, connecting autonomous nodes over the compute fabric. For H100 systems that boundary is the 8-GPU chassis.

### Scaling beyond the node: SuperPods and rail-optimized fat-tree topologies

To train modern models with hundreds of billions of parameters, clusters must scale beyond a single node to thousands of GPUs. This scale-out tier is powered by **InfiniBand** (or high-performance RoCE v2 Ethernet).

In a 1,016-GPU DGX H100 SuperPod (127 nodes; a pair of UFM appliances takes the 128th slot):
- The cluster is partitioned into **Scalable Units (SUs)**, each containing 32 DGX nodes (256 GPUs).
- Each node houses 8 ConnectX-7 400 Gbps (NDR) NICs, providing $8 \times 50 \text{ GB/s} = 400 \text{ GB/s}$ of scale-out network egress per node (or 50 GB/s per GPU).
- Each SU contains 8 Leaf switches. Across 4 SUs, 16 central Spine switches connect the fabric into a non-blocking 1:1 **Fat-Tree topology** providing 51.2 TB/s of aggregate switching capacity.

A **leaf** is the lower-tier switch that servers plug into; a **spine** is the upper-tier switch that joins leaves and has no servers attached. Every leaf connects to every spine, so any two servers meet in at most leaf → spine → leaf, and the path does not get longer as the cluster grows: to add servers you add leaves, to add bandwidth you add spines. The pattern comes from telephone exchanges (Clos, 1953); when the links thicken toward the top it is called a fat-tree (Leiserson, 1985).

A **hop** is one pass through a switch ASIC. **Bisection bandwidth** is the capacity between two equal halves of the network under the worst possible split, and a non-blocking or full-bisection fabric guarantees that no permutation of traffic is throttled by the uplinks under balanced routing. In practice, hash collisions between flows, full switch buffers and short microbursts can still add tail latency, which is why traffic patterns are measured rather than assumed.

Crucially, modern clusters employ **Rail-Optimized wiring**. In this topology, GPU rank $i$ in every node is wired to the exact same Leaf switch $i$, the switch of rail $i$. When GPUs with the same index on nodes of the same SU run data-parallel or pipeline-parallel collectives, packets travel purely within their designated rail without crossing the spine switches, eliminating spine hops and preventing fabric congestion. In a classic top-of-rack layout, where all eight NICs of a node go to the same leaf, GPU 3 of node A would reach GPU 3 of node B through leaf → spine → leaf in three hops; with rails the same message takes one. It works because tensor parallelism stays inside the chassis: GPU $i$ of every node then holds the same slice of the model, so its pipeline and data-parallel partners are also GPU $i$ elsewhere.

<svg viewBox="0 0 560 440" role="img" aria-label="1,016-GPU DGX H100 SuperPod Fat-Tree InfiniBand network topology diagram." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sp-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="sp-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">SuperPod Fat-Tree Scaling Topology</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1,016 GPUs · Full Bisection Bandwidth</text>
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
<text x="480.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">31 Nodes (248 Chips)</text>
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

**Why "non-blocking" holds: count one leaf's ports.** Each leaf is a Quantum-2 QM9700 with 64 ports of 400 Gb/s. 32 ports go down, one to NIC $i$ of each of the 32 nodes in its SU, and 32 go up, two cables to each of the 16 spines. Whatever enters from below can leave upward, so the oversubscription ratio is 1:1. The same count at the top of the tree:

$$\underbrace{16 \times 64 \times 50 \text{ GB/s}}_{\text{spine capacity}} = 51.2 \text{ TB/s} \;\ge\; \underbrace{1{,}016 \times 50 \text{ GB/s}}_{\text{all GPUs sending}} = 50.8 \text{ TB/s}$$

Read this as: 16 spines with 64 ports of 50 GB/s each (400 Gb/s ÷ 8) can carry everything that all 1,016 GPUs can push into the fabric at once. The 32 leaves × 32 uplinks = 1,024 cables exactly fill 16 spines × 64 ports. Do not mix this up with the "51.2 Tb/s" in NVIDIA's QM9700 manual: that is one switch, both directions, in bits; the 51.2 TB/s here is 16 spines, one direction, in bytes.

**Why 127 nodes, not 128.** On paper 4 SUs × 32 = 128 DGX. In NVIDIA's deployment pattern a pair of UFM appliances, which watch the InfiniBand fabric, takes one DGX's place in SU 4, so a full SuperPOD holds at most 127 DGX: 1,016 GPUs. NVIDIA's cable table shows it too: 127 × 8 + 4 UFM ports = 1,020 node-to-leaf cables. Smaller builds start from a single SU, anywhere between 31 and 127 DGX.

**Four routes, four prices.** The rail layout makes some traffic cheap and some expensive, and card 4b of the poster prices each route:

| Route | Typical traffic | Switches crossed |
| :--- | :--- | :--- |
| Same rail, same SU | Data-parallel gradients, pipeline activations | 1 (leaf $i$) |
| Same rail, another SU | Data parallelism across SUs | 3 (leaf → spine → leaf) |
| Different rail, direct | MoE AllToAll | 3 (leaf → spine → leaf) |
| Different rail, with PXN | MoE AllToAll in NCCL ≥ 2.12 | NVLink hop + 1 switch |

PXN (PCIe × NVLink) in NCCL 2.12 first moves a cross-rail message over NVSwitch to the GPU in the same node that sits on the destination's rail, then sends it without leaving that rail. The eight GPUs of a node thus gather their messages for the same destination on one GPU and send them together; NVIDIA reports that this doubles all2all performance. DeepSeek-V3, trained on 2,048 H800s, does the mirror image: a token first travels over InfiniBand to the GPU with the same in-node index on the target node, staying on its own rail, and is then forwarded over NVLink to the expert's GPU.

**How far does the fabric grow?** A spine's 64 ports reach at most 64 leaves, and each leaf carries 32 GPUs, so two tiers stop at $64 \times 32 = 2{,}048$ GPUs: exactly 8 SUs. From 16 SUs on, NVIDIA adds a core tier above the spines, and the longest path grows from 3 switches (leaf, spine, leaf) to 5 (leaf, spine, core, spine, leaf):

| SUs | DGX | GPUs | Leaf | Spine | Core | Tiers | Longest path |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 31 | 248 | 8 | 4 | – | 2 | 3 switches |
| 4 | 127 | 1,016 | 32 | 16 | – | 2 | 3 switches |
| 8 | 256 | 2,048 | 64 | 32 | – | 2 | 3 switches |
| 16 | 512 | 4,096 | 128 | 128 | 64 | 3 | 5 switches |
| 64 | 2,048 | 16,384 | 512 | 512 | 256 | 3 | 5 switches |

The 1 and 4 SU rows come from NVIDIA's SuperPOD table, where one slot goes to UFM; the larger rows come from its scaling table, which counts 32 DGX per SU (that table labels the last row "56 SU", but $2{,}048 \div 32 = 64$). Three tiers reach up to $64 \times 32 \times 32 = 65{,}536$ GPUs. Meta's Llama 3 cluster, built on RoCE Ethernet with 24K GPUs, also has three tiers but is not non-blocking end to end: its 3,072-GPU pods are 1:1 inside and 1:7 between pods, which is why Meta made its parallelism layout and job scheduler aware of the network topology, keeping the heaviest traffic inside a pod.

The poster's scale ladder sums the zoom up: each level multiplies BF16 compute (7.5 TFLOP/s per SM, 990 per H100, 7.9 PFLOP/s per DGX, 253 per SU, 1.0 EFLOP/s per SuperPOD and 16.2 EFLOP/s for 64 SUs) while the road out narrows per GPU from 3.35 TB/s to 450 GB/s to 50 GB/s.

### In-network hardware acceleration: Mellanox SHARP and PXN

When scaling across thousands of GPUs, transferring massive model and gradient tensors creates extreme pressure on the network fabric. In traditional collective communication (such as the AllReduce primitive implemented in NCCL), data moves across logical GPU rings in two successive passes: first a ReduceScatter to form partial sums, followed by an AllGather to replicate the accumulated result. Across $N$ GPUs and network egress bandwidth $W$, a classic ring AllReduce transfers roughly $2B$ bytes per GPU:

$$T_{\text{AllReduce, Ring}} \approx 2 \cdot \frac{B}{W}$$

This mechanism forces all tensor slices to traverse the leaf and spine switch fabric twice. To eliminate this $2\times$ bandwidth penalty, modern InfiniBand fabrics employ **Mellanox SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)**. SHARP embeds floating-point reduction logic directly into the Quantum switch ASICs:

<svg viewBox="0 0 560 400" role="img" aria-label="Comparison between ring-based collective communication and Quantum switch SHARP in-network reduction." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="col-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="col-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="col-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="col-arr-suc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Collective Communication Primitives</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Ring AllGather vs SHARP Hardware AllReduce</text>
<rect x="16" y="48" width="256" height="276" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="68" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Classic Ring Collective</text>
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
<text x="144" y="270" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Duration: T ≈ 2B / W_egress</text>
<text x="144" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2(N-1) steps ring traversal</text>
<text x="144" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2x transfer for AllReduce (RS + AG)</text>
<rect x="288" y="48" width="256" height="276" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="68" text-anchor="middle" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">In-Network Reduction (SHARP)</text>
<text x="416" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Quantum-2 Hardware Summation</text>
<rect x="348" y="104" width="136" height="46" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="123" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Quantum Switch</text>
<text x="416" y="138" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10px;font-weight:600">In-ASIC FP Summation</text>
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
<text x="416" y="180" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">Aggregate and broadcast in 1 pass</text>
<text x="416" y="270" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Duration: T ≈ B / W_egress (1x)</text>
<text x="416" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Reduces wire traffic by 50%</text>
<text x="416" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Halves AllReduce latency</text>
<rect x="16" y="334" width="528" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="354" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Intra-node AllGather over NVLink: T ≈ B / 450 GB/s (H100).</text>
<text x="28" y="372" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">With SHARP enabled, AllReduce completes in single-pass duration instead of 2x (RS + AG).</text>
</svg>

Rather than passing tensors through an iterative bucket ring, each GPU streams its tensor slice up to the switch once. The switch ASIC accumulates vectors in-flight and multicasts the reduced tensor back down in a single pass:

$$T_{\text{AllReduce, SHARP}} \approx \frac{B}{W}$$

On large, bandwidth-bound transfers, SHARP cuts the wire traffic per GPU by 50% and brings AllReduce latency directly in half. The rail-optimized cabling maps naturally into this reduction tree: GPU $i$ of every server in an SU converges on leaf $i$, and partial sums across SUs meet at the spine switches.

**Cross-Rail Bridge: PXN (PCIe Acceleration / NVLink)**

In a rail-optimized cluster, every GPU connects directly to its designated leaf switch ($i$-th GPU $\rightarrow$ $i$-th leaf). However, whenever cross-rail communication is needed (for example, GPU 0 in node A communicating with GPU 3 in node B), a standard packet would have to climb from the leaf to a spine switch, traverse down to the destination rail's leaf, and finally enter node B. This spine detour injects unnecessary latency and congestion into the upper fabric.

NVIDIA NCCL's **PXN (PCIe Acceleration / NVLink)** mechanism resolves this bottleneck by leveraging the high-speed intra-node mesh: the source GPU routes the packet over intra-chassis 450 GB/s NVLink to the local GPU belonging to the target rail, which then transmits it directly over its dedicated NIC to the remote server in a single hop. The spine switches remain entirely bypassed.

This hierarchical interconnect topology—spanning PCIe, NVLink, and NUMA—can be verified on any HGX node using `nvidia-smi topo -m`:

```text
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NIC0  NIC1  CPU0  CPU1
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18  PIX   PXB   SYS   SYS
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18  PIX   PXB   SYS   SYS
GPU2    NV18  NV18   X    NV18  NV18  NV18  NV18  NV18  PXB   PIX   SYS   SYS
GPU3    NV18  NV18  NV18   X    NV18  NV18  NV18  NV18  PXB   PIX   SYS   SYS
GPU4    NV18  NV18  NV18  NV18   X    NV18  NV18  NV18  SYS   SYS   SYS   SYS
```

Here `NV18` represents 18 4th-generation NVLink connections between GPUs (900 GB/s bidirectional), `PIX` denotes direct traversal through a single PCIe switch, and `SYS` traverses NUMA CPU interconnects. This microsecond-level hardware clarity demonstrates why modern cluster fabrics coordinate communication paths across physical tiers rather than relying on generic network routing.

---

## 4. Pillar III — Storage: Direct DMA and the high-throughput datapath

In an AI supercomputer, the storage subsystem must satisfy two fundamentally different workloads: continuous dataset streaming without GPU starvation, and writing model checkpoints without freezing the compute fabric.

### Four networks per node: why storage gets its own fabric

Inspecting the rear of a DGX H100 reveals four physically isolated network fabrics rather than a shared set of cables. The compute network carries strictly GPU-to-GPU traffic: forward activations, backward passes, and gradient AllReduce operations. If checkpoint saves or OS container pulls shared these cables, compute traffic would immediately suffer severe latency jitter. Consequently, each traffic class is isolated onto dedicated links:

| Network | Per DGX | Switches (4 SU) | Per-Node Bandwidth | Primary Traffic |
| :--- | :--- | :--- | :--- | :--- |
| Compute network | 8 × 400 Gb/s InfiniBand | 48 × QM9700, 1:1 | 400 GB/s | Inter-node GPU traffic |
| Storage network | 2 × 400 Gb/s InfiniBand | 16 × QM9700, ~4:3 | 100 GB/s | Dataset streaming, checkpoint writes |
| In-band management | 2 × 100 GbE | 8 × SN4600C | 25 GB/s | OS, orchestration, cluster services |
| Out-of-band management | 1 × 1 GbE (BMC) | 8 × SN2201 | 0.125 GB/s | Hardware telemetry and power control |

Closest to the GPUs, each node also carries 8 × 3.84 TB U.2 NVMe drives configured as a local RAID 0 cache. Text pre-training datasets are frequently cached in this local NVMe tier. The shared external high-throughput storage cluster is primarily sized to absorb periodic bursts of multi-gigabyte checkpoint snapshots.

### The traditional I/O bottleneck: Host CPU page cache and bounce buffers

In conventional server architectures, a storage transfer traverses a multi-hop CPU detour:
1. **NVMe Read:** Data moves from local SSDs over PCIe to the host CPU root complex.
2. **Kernel Page Cache Copy:** The Linux kernel copies data into page cache buffers in system DDR5 RAM.
3. **User Space Bounce Buffer:** The application or I/O library performs a `memcpy` from kernel space to user space memory.
4. **Device Transmission:** The host CPU issues a second DMA command to stream data over PCIe Gen5 into GPU High Bandwidth Memory (HBM).

This legacy datapath imposes two severe penalties: data must traverse the PCIe bus twice, and gigabytes of memory copies saturate host CPU cores, turning them into memory-copy engines rather than data-preparation workers. As a result, I/O throughput drops to the speed of the slowest `memcpy` loop.

### GPUDirect Storage (GDS): Direct DMA from NVMe to GPU memory

Modern AI supercomputers eliminate this bottleneck using the **GPUDirect Storage (GDS)** architecture:

<svg viewBox="0 0 560 420" role="img" aria-label="Comparison between conventional CPU bounce buffer I/O and GPUDirect Storage direct DMA architecture." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
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
<text x="150" y="232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">I/O Step 1</text>
<line x1="144" y1="148" x2="144" y2="134" marker-end="url(#gds-arr-warn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="150" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">I/O Step 2</text>
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
<text x="28" y="364" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Checkpoint Challenge: Frontier models snapshot massive state periodically.</text>
<text x="28" y="382" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">With GDS, storage streams directly to GPU HBM bypassing system RAM,</text>
<text x="28" y="398" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">eliminating CPU bounce latency and preventing cluster training stalls.</text>
</svg>

GDS operates through the NVIDIA `cuFile` API and the `nvidia-fs.ko` kernel driver. It establishes a direct DMA (Direct Memory Access) datapath between local NVMe drives (or NVMe-oF network storage) and GPU HBM over the PCIe switch fabric:
- Data traverses the PCIe bus exactly once; host memory intermediate copies are entirely eliminated.
- Host CPU cores are freed from `memcpy` operations to focus on dataloading and token preprocessing.
- Latency jitter drops significantly, aligning the I/O datapath with physical PCIe Gen5 line rates (~64 GB/s per direction).

Within a DGX H100 system, storage NICs and local NVMe drives sit directly on the same PCIe switches as the GPUs in their respective NUMA half. This ensures that direct DMA transfers complete within the local PCIe switch without crossing inter-socket UPI interconnects. At the code level, direct DMA access is invoked via the `cuFile` API:

```cpp
// cuFile API: Direct DMA from NVMe to GPU HBM (CPU bounce buffer bypassed)
CUfileDescr_t cf_descr;
cf_descr.handle.fd = fd;
cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
cuFileHandleRegister(&cf_handle, &cf_descr);

// Direct read into device pointer without host RAM bounce
cuFileRead(cf_handle, dev_ptr, size, file_offset, 0);
cuFileHandleDeregister(cf_handle);
```

---

## 5. Synthesis: Hardware hierarchy and latency gaps

The architectural decisions across the three pillars place hardware under fundamentally contrasting operating regimes between pre-training and high-throughput inference serving:

### Two engineering lenses: Training and inference hardware regimes

Hardware requirements diverge sharply between foundation model pre-training and real-time inference:

| Architectural Dimension | Training (Pre-training / SFT) | Inference (Prefill & Decode) |
| :--- | :--- | :--- |
| **Primary Memory Footprint** | Weights ($2B$) + Gradients ($2B$) + AdamW ($12-16B$) + Activations | Model Weights ($4-8$ bit quantized) + Dynamically allocated KV Cache |
| **Compute / Memory Regime** | Heavily **Compute-Bound** (GEMM with thousands of tokens per batch) | Prefill is **Compute-Bound**; Decode is strictly **Memory-Bandwidth-Bound** ($B=1$, GEMV) |
| **Dominant Network Primitive** | High-throughput **AllReduce** & **ReduceScatter** over InfiniBand | Low-latency tensor-parallel **AllReduce** over NVLink, once or twice per layer per token |
| **Key Performance Metrics** | Model FLOPs Utilization (MFU), Step Time (ms), Gradient Stability | Time to First Token (TTFT), Inter-Token Latency (TPOT), Concurrent Requests |
| **Critical Failure Modes** | Loss spikes, silent data corruption (SDC), pipeline bubbles | KV Cache Out-of-Memory (OOM), memory fragmentation, tail latency spikes |

Part D of the poster makes the inference column tangible. Push 140 GB of BF16 model weights through every link and the layers separate by orders of magnitude:

```text
link                         time for 140 GB
HBM3 (one H100, 3.35 TB/s)   140 ÷ 3,350  ≈ 0.042 s  =  42 ms
NVLink (450 GB/s)            140 ÷ 450    ≈ 0.31 s
PCIe Gen5 (64 GB/s)          140 ÷ 64     ≈ 2.2 s
InfiniBand (50 GB/s per GPU) 140 ÷ 50     =  2.8 s
legacy storage (25 GB/s)     140 ÷ 25     =  5.6 s
```

In human terms, data that arrives from HBM in one second takes 7.4 s over NVLink, 52 s over PCIe, and 67 s over InfiniBand. The decode phase is bound by the top row: every generated token reads all weights once. This hardware hierarchy connects directly to how multi-GPU clusters scale, how MFU losses are managed under real workloads, and how operational failure mechanisms are mitigated in the second article of this series: [AI Infrastructure (2): Diagnosing an AI Supercomputer — Why 1,024 H100s Stall at 38% MFU](post.html?slug=from-gpus-to-ai-supercomputers).

---

## The whole story in six lines

- **Three Pillars Co-Design:** An AI supercomputer does not function as isolated chips; compute (SM/Tensor Core), network fabric (NVLink/InfiniBand), and storage (NVMe/GDS) must be balanced with matched bandwidths.
- **SM and Tensor Core Anatomy:** The H100 chip delivers 990 TFLOP/s BF16 across 132 SMs and 528 Tensor Cores; to overcome the memory wall, each SM maintains up to 64 resident warps to hide ~480-cycle HBM3 access latencies.
- **In-Chassis NVLink Crossbar:** Eight H100 GPUs connect across 4 NVSwitch ASICs in a non-blocking crossbar; 450 GB/s unidirectional bandwidth per GPU enables seamless intra-layer tensor parallel exchanges.
- **SuperPOD and Rail-Aligned Fat-Tree:** Clusters are built from 32-DGX Scalable Units (SUs) wired into a two-tier non-blocking fat-tree; rail optimization guarantees that GPUs sharing the same model slice communicate in a single switch hop.
- **In-Network Hardware Acceleration (SHARP & PXN):** Quantum-2 switch ASICs aggregate AllReduce reductions in-flight, cutting network traffic by 50%; PXN routes cross-rail packets across local NVLink to bypass spine congestion.
- **Direct Storage DMA (GDS):** Four independent network fabrics isolate traffic per node; GPUDirect Storage bypasses CPU page caches and host RAM bounce buffers to stream directly between NVMe and GPU HBM via PCIe DMA.

---

## Glossary

- **Streaming Multiprocessor (SM)** — The core multi-threaded execution cell of an NVIDIA GPU containing Tensor Cores, Warp Schedulers, CUDA cores, and on-chip SRAM.
- **Warp** — The fundamental scheduling unit in NVIDIA architectures, consisting of 32 parallel threads executed synchronously by an SM warp scheduler.
- **Warp Divergence** — Performance penalty occurring when threads within a single warp branch into different execution paths, forcing inactive cores to stall.
- **Occupancy** — The fraction of an SM's 64 warp slots actively occupied by a kernel; decreases as registers per thread increase and serves as the SM's mechanism for latency hiding.
- **Critical Arithmetic Intensity ($\alpha_{\text{crit}}$)** — The ratio of peak compute throughput to memory bandwidth ($C_{\text{peak}} / W_{\text{mem}} \approx 295$ FLOP/byte); operations below this ridge run memory-bound.
- **NVLink** — NVIDIA's proprietary high-bandwidth interconnect providing 450 GB/s unidirectional (900 GB/s bidirectional) memory transfer per H100 GPU.
- **NVSwitch** — On-node crossbar ASIC enabling non-blocking, all-to-all communication across all 8 GPUs in an HGX chassis.
- **Fabric Manager** — NVIDIA's software service that configures NVSwitch routing tables over PCIe; without it, the NVLink mesh does not initialize.
- **NUMA Domain** — A CPU socket with its dedicated memory and attached PCIe devices; a DGX H100 has two NUMA halves, and data pipelines must align to the local half.
- **Scalable Unit (SU)** — The modular building block of a DGX SuperPOD: 32 DGX nodes (256 GPUs) and the 8 leaf switches interconnecting them.
- **Leaf and Spine Switch** — The two tiers of a fat-tree network: compute nodes connect to leaf switches, leaves connect to every spine, and any two servers communicate within at most three switches.
- **Rail-Optimized Cabling** — Interconnect topology wiring NIC $i$ of every server to leaf switch $i$, allowing GPUs holding identical tensor partitions to meet in a single hop.
- **PXN (PCIe Acceleration / NVLink)** — Hardware/NCCL feature routing cross-rail packets across local NVLink to reach the target rail's NIC, bypassing spine switches.
- **Mellanox SHARP** — Scalable Hierarchical Aggregation and Reduction Protocol; Quantum switch hardware that computes floating-point AllReduce operations in-flight.
- **GPUDirect Storage (GDS)** — Direct DMA technology connecting NVMe storage directly to GPU High Bandwidth Memory, bypassing host CPU page caches and bounce buffers.

---

## Going deeper

- On this blog: [AI Infrastructure (2): Diagnosing an AI Supercomputer — Why 1,024 H100s Stall at 38% MFU](post.html?slug=from-gpus-to-ai-supercomputers).
- Jacob Austin, Swapnil Patil, Adam Paszke, Reiner Pope (2025). [How to Scale Your Model: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). JAX-ML Scaling Book, Chapter 12.
- SemiAnalysis (2025). [NVIDIA Tensor Core Evolution from Volta to Blackwell](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/). Microarchitectural retrospective.
- NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Official microarchitecture guide: GPCs, TPCs, SM subpartitions, HBM3 stacks and memory controllers.
- NVIDIA Corporation. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html). The SM limits behind Figure 1.1: 64 warps and 64K registers per SM, at most 255 registers per thread, up to 228 KB of shared memory, and the TMA copy engine.
- Weile Luo et al. (2024). [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499). arXiv:2402.13499. Measured Hopper load latencies used in Figure 1.1: SMEM ~29, L1 ~41, L2 ~263 and global memory ~479 cycles.
- NVIDIA Corporation. [DGX SuperPOD Reference Architecture (DGX H100): Key Components](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-components.html) and [Network Fabrics](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html). Scalable units, the four networks, 127 nodes with a UFM slot, and the separate storage fabric; diagrams in Figure 1.1 were inspired by these.
- NVIDIA Corporation. [Introduction to NVIDIA DGX H100/H200 Systems](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). Node hardware: CPUs, memory, NICs and NVMe drives; its DGX H100/H200 system topology diagram inspired the node level of Figure 1.1.
- NVIDIA Corporation. [Fabric Manager for NVIDIA NVSwitch Systems](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). How the 18 NVLinks of each H100 split as 5 + 4 + 4 + 5 over four NVSwitches, and why the NVLink fabric does not come up without Fabric Manager.
- NVIDIA Corporation. [DGX SuperPOD Architecture (DGX H100)](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html). Scaling table: two tiers up to 8 SUs, a core tier from 16 SUs; 512 leaf, 512 spine and 256 core switches for 2,048 DGX.
- NVIDIA Corporation. [QM97X0 NDR 400Gb/s InfiniBand Switch Systems User Manual: Introduction](https://docs.nvidia.com/networking/display/qm97x0um/introduction). Quantum-2 QM9700: 64 NDR 400 Gb/s ports, 51.2 Tb/s aggregate bidirectional throughput per switch, SHARP in-network reduction.
- Charles Clos (1953). [A Study of Non-Blocking Switching Networks](https://doi.org/10.1002/j.1538-7305.1953.tb01433.x). Bell System Technical Journal 32(2), 406–424. The multi-stage switching idea behind today's leaf–spine networks.
- Charles E. Leiserson (1985). [Fat-trees: Universal Networks for Hardware-Efficient Supercomputing](https://doi.org/10.1109/TC.1985.6312192). IEEE Transactions on Computers C-34(10), 892–901. Where the fat-tree name comes from: links that thicken toward the top.
- NVIDIA Developer Blog (2022). [Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/). PXN: moving cross-rail traffic onto NVLink first so it skips the spine.
