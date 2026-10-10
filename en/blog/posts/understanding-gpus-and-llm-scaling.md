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

<svg viewBox="0 0 560 1764" role="img" aria-label="The big picture of AI infrastructure as three stacked pillars. Pillar 1, compute, inside one GPU: a Streaming Multiprocessor has 4 identical subpartitions, each with a Tensor Core, a warp scheduler, 32 FP32 ALUs and 16,384 registers; the H100 has 132 SMs and 990 TFLOP/s BF16, the B200 148 SMs and 2,250. The memory hierarchy runs from registers (256 KiB per SM), shared memory or L1 (256 KiB), TMEM on B200, L2 (50 or 126 MB, about 5.5 TB/s) down to HBM (80 GB at 3.35 TB/s on H100, 192 GB at 8 TB/s on B200). The roofline knee is 990 divided by 3.35, about 295.5 FLOPs per byte: decode at batch 1 has an intensity near 1 and reaches only 1 to 2 percent MFU, while training batches above about 295 tokens are compute-bound. Memory wall: FLOPs grew over 1,000 times in a decade, HBM bandwidth under 30 times. GPUs talk to each other over NVLink at 450 GB/s and to the CPU over PCIe Gen5 at 64 GB/s. Pillar 2, network: inside a node, 8 GPUs connect through 4 NVSwitches with 6.4 TB/s switching and 3.6 TB/s bisection, 18 ports times 25 GB/s giving 450 GB/s per GPU; NVL72 extends this to 72 GPUs. Across nodes, an InfiniBand fat-tree joins 16 spine switches and 4 scalable units of 256 GPUs, 1,024 GPUs in all; each node has 8 400 Gb/s NICs, 400 GB/s per node or 50 GB/s per GPU, and GPU i connects to leaf i. Tensor parallelism stays inside NVLink: 28,672 divided by 2,200 is about 13, so 8 GPUs. Pipeline parallelism splits layers, with bubble (P minus 1) over (P minus 1 plus M); expert parallelism spreads MoE experts with AllToAll, as in DeepSeek-V3&#x27;s 64-way EP; data parallelism AllReduces gradients, as in LLaMA-3 405B&#x27;s 128-way ZeRO-1. A ring AllReduce takes about 2B over W; SHARP sums in the switch and needs about B over W, halving traffic. Network wall: leaving the node drops bandwidth from 450 to 50 GB/s per GPU, 9 times. Storage feeds tokens upward at 2 TB/s and receives 7.5 TB checkpoints. Pillar 3, storage: the legacy path goes from NVMe through the page cache and a user buffer to HBM with two memcpys and the CPU at 100 percent; GPUDirect Storage goes from NVMe-oF through a PCIe switch straight into HBM by DMA, cutting latency 80 percent. The data feed streams 15 trillion tokens as .bin and .idx files with zero-copy mmap; Meta Tectonic has 7,500 servers, 240 PB, 2 TB/s read and 7 TB/s burst write. A 405B checkpoint is 16 bytes per parameter: 0.81 TB weights, 0.81 TB gradients, 4.86 TB AdamW state, about 7.5 TB with activations and RNG. Young-Daly sets the interval to the square root of 2 times delta times M; with M of 3 hours, 25 GB/s gives delta 307 s, tau 42.9 minutes and about 24 percent loss, while 250 GB/s gives delta 31 s, tau 13.6 minutes and about 7.5 percent loss. I/O wall: a 7.5 TB checkpoint at 25 GB/s idles 16,384 GPUs for 307 s. A final log-scale bandwidth ladder per GPU: L2 about 5.5 TB/s, HBM 3.35 TB/s, NVLink 450 GB/s, PCIe 64 GB/s, InfiniBand 50 GB/s, and legacy storage 25 GB/s for the whole cluster. The slowest rung sets the pace." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="tri-a-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="tri-a-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="tri-a-wrn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="tri-a-bad" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
<marker id="tri-a-ok" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700">The Big Picture: The 3 Pillars of AI Infrastructure</text>
<text x="16" y="42" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Follow a byte: storage → HBM → SM, and from GPU to GPU over NVLink → InfiniBand</text>
<rect x="16" y="52" width="11" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="32" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">compute</text>
<rect x="86.11500000000001" y="52" width="11" height="11" rx="2" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="102.11500000000001" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">network</text>
<rect x="156.23000000000002" y="52" width="11" height="11" rx="2" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="172.23000000000002" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">storage</text>
<rect x="226.34500000000003" y="52" width="11" height="11" rx="2" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="242.34500000000003" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">wall / bottleneck</text>
<rect x="350.91" y="52" width="11" height="11" rx="2" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="366.91" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">fix</text>
<rect x="16" y="74" width="528" height="420" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.4"/>
<text x="28" y="98" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700">1 · COMPUTE</text>
<text x="532" y="98" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">inside one GPU · H100 / B200</text>
<text x="28" y="118" text-anchor="start" style="fill:var(--c-text);font-size:12px">Question: can the Tensor Cores be fed data fast enough?</text>
<rect x="28" y="130" width="236" height="200" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="148" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Streaming Multiprocessor</text>
<text x="38" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100: 132 SM · 990 TFLOP/s BF16</text>
<text x="38" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">B200: 148 SM · 2,250 TFLOP/s BF16</text>
<path d="M56 310 H222 V206" style="fill:none;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:3 3;stroke-opacity:.7"/>
<path d="M52 306 H218 V202" style="fill:none;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:3 3;stroke-opacity:.7"/>
<path d="M48 302 H214 V198" style="fill:none;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:3 3;stroke-opacity:.7"/>
<rect x="38" y="188" width="172" height="110" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="203" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:600">1 subpartition</text>
<rect x="46" y="210" width="156" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="124.0" y="223.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="46" y="231" width="156" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="124.0" y="244.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Warp Scheduler</text>
<rect x="46" y="252" width="156" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="124.0" y="265.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32 × FP32 ALU</text>
<rect x="46" y="273" width="156" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="124.0" y="286.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16,384 registers</text>
<text x="236" y="254" text-anchor="middle" style="fill:var(--c-accent);font-size:15px;font-weight:700">×4</text>
<text x="38" y="322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">each SM = 4 identical subpartitions</text>
<rect x="276" y="130" width="256" height="200" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="286" y="148" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Memory hierarchy</text>
<text x="286" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">↑ fast and small · ↓ large and slow</text>
<rect x="286" y="174" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.28;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="190.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Register</text>
<text x="358" y="190" text-anchor="start" style="fill:var(--c-text);font-size:11px">256 KiB/SM · ~33 MB per chip</text>
<rect x="286" y="202" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="218.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SMEM/L1</text>
<text x="358" y="218" text-anchor="start" style="fill:var(--c-text);font-size:11px">256 KiB/SM · &lt;10 ns</text>
<rect x="286" y="230" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="318.0" y="246.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">TMEM</text>
<text x="358" y="246" text-anchor="start" style="fill:var(--c-text);font-size:11px">256 KiB/SM · B200 only</text>
<rect x="286" y="258" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="274.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2</text>
<text x="358" y="274" text-anchor="start" style="fill:var(--c-text);font-size:11px">50 / 126 MB · ~5.5 TB/s</text>
<rect x="286" y="286" width="64" height="36" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="308.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<text x="358" y="301" text-anchor="start" style="fill:var(--c-text);font-size:11px">80 GB · 3.35 TB/s (H100)</text>
<text x="358" y="316" text-anchor="start" style="fill:var(--c-text);font-size:11px">192 GB · 8.0 TB/s (B200)</text>
<rect x="28" y="338" width="504" height="114" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="358" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Roofline: α = FLOPs per byte moved</text>
<text x="40" y="378" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">α_crit = C / W = 990 / 3.35</text>
<text x="40" y="394" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">≈ 295.5 FLOP/byte (H100)</text>
<rect x="40" y="402" width="300" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="190.0" y="416.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">decode (B = 1): α ≈ 1 → MFU 1–2%</text>
<rect x="40" y="426" width="300" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="190.0" y="440.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">training (B ≳ 295 tokens): compute-bound</text>
<line x1="372" y1="350" x2="372" y2="436" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="372" y1="436" x2="520" y2="436" style="stroke:var(--c-text-mute);stroke-width:1"/>
<path d="M372 436 L444 374 L516 374" style="fill:none;stroke:var(--c-text);stroke-width:1.8"/>
<line x1="444" y1="374" x2="444" y2="436" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="378" y="356" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FLOP/s</text>
<text x="450" y="392" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">α_crit ≈ 295</text>
<circle cx="384" cy="425.7" r="4" style="fill:var(--c-danger)"/>
<text x="384" y="448" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">decode</text>
<circle cx="496" cy="374" r="4" style="fill:var(--c-success)"/>
<text x="496" y="366" text-anchor="middle" style="fill:var(--c-success);font-size:11px">training</text>
<rect x="28" y="460" width="504" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="475.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Memory wall: in a decade FLOPs grew 1,000×+, HBM bandwidth less than 30×</text>
<line x1="100" y1="497" x2="100" y2="525" marker-start="url(#tri-a-gen)" marker-end="url(#tri-a-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="110" y="515" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">GPU ↔ GPU: NVLink 450 GB/s</text>
<line x1="338" y1="497" x2="338" y2="525" marker-start="url(#tri-a-mut)" marker-end="url(#tri-a-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="348" y="515" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU ↔ CPU: PCIe Gen5 64 GB/s</text>
<rect x="16" y="528" width="528" height="488" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="28" y="552" text-anchor="start" style="fill:var(--c-accent-2);font-size:15px;font-weight:700">2 · NETWORK</text>
<text x="532" y="552" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">GPU to GPU · node → cluster</text>
<text x="28" y="572" text-anchor="start" style="fill:var(--c-text);font-size:12px">Question: how fast can GPUs share partial results?</text>
<rect x="28" y="584" width="244" height="196" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="602" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Inside a node: NVLink</text>
<text x="38" y="618" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 GPUs + 4 NVSwitch (crossbar)</text>
<rect x="38" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="63.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU0</text>
<rect x="96" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="121.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU1</text>
<rect x="154" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="179.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU2</text>
<rect x="212" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU3</text>
<rect x="38" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="63.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU4</text>
<rect x="96" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="121.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU5</text>
<rect x="154" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="179.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU6</text>
<rect x="212" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU7</text>
<line x1="63" y1="650" x2="63" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="63" y1="692" x2="63" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="121" y1="650" x2="121" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="121" y1="692" x2="121" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="179" y1="650" x2="179" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="179" y1="692" x2="179" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="237" y1="650" x2="237" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="237" y1="692" x2="237" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<rect x="38" y="668" width="224" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="150.0" y="684.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4 × NVSwitch · 6.4 TB/s</text>
<text x="38" y="750" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">18 × 25 GB/s = 450 GB/s/GPU</text>
<text x="38" y="768" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">bisection 3.6 TB/s · NVL72: 72 GPUs</text>
<rect x="288" y="584" width="244" height="196" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="602" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Across nodes: InfiniBand</text>
<text x="298" y="618" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">fat-tree · 4 SU × 256 = 1,024 GPUs</text>
<rect x="298" y="628" width="224" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="410.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16 spine switches · 51.2 TB/s</text>
<line x1="324.0" y1="670" x2="309.2" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="324.0" y1="670" x2="326.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="324.0" y1="670" x2="342.8" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="298.0" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU0</text>
<line x1="324.0" y1="692" x2="324.0" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="298.0" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="324.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<line x1="381.3" y1="670" x2="365.2" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="381.3" y1="670" x2="382.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="381.3" y1="670" x2="398.8" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="355.3" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="381.3" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU1</text>
<line x1="381.3" y1="692" x2="381.3" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="355.3" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="381.3" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<line x1="438.6" y1="670" x2="421.20000000000005" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="438.6" y1="670" x2="438.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="438.6" y1="670" x2="454.79999999999995" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="412.6" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="438.6" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU2</text>
<line x1="438.6" y1="692" x2="438.6" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="412.6" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.6" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<line x1="495.9" y1="670" x2="477.20000000000005" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="495.9" y1="670" x2="494.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="495.9" y1="670" x2="510.79999999999995" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="469.9" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="495.9" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU3</text>
<line x1="495.9" y1="692" x2="495.9" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="469.9" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="495.9" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<text x="298" y="750" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 × 400 Gb/s NIC = 400 GB/s per node</text>
<text x="298" y="768" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">50 GB/s per GPU · GPU i → leaf i</text>
<rect x="28" y="788" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="806" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">TP · tensor parallel</text>
<rect x="212.4" y="794" width="53.6" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="239.20000000000002" y="807.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVLink</text>
<text x="38" y="822" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">splits each weight matrix</text>
<text x="38" y="838" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Y ≤ 28,672 / 2,200 ≈ 13 → 8</text>
<rect x="284" y="788" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="806" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">PP · pipeline parallel</text>
<rect x="494.8" y="794" width="27.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="508.40000000000003" y="807.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">IB</text>
<text x="294" y="822" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">splits the layers into stages</text>
<text x="294" y="838" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">bubble = (P−1) / (P−1+M)</text>
<rect x="28" y="852" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="870" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">EP · expert parallel</text>
<rect x="238.8" y="858" width="27.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="252.4" y="871.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">IB</text>
<text x="38" y="886" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spreads MoE experts · AllToAll</text>
<text x="38" y="902" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">DeepSeek-V3: 64-way, 8 nodes</text>
<rect x="284" y="852" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="870" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">DP / FSDP · data parallel</text>
<rect x="494.8" y="858" width="27.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="508.40000000000003" y="871.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">IB</text>
<text x="294" y="886" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AllReduces the gradients</text>
<text x="294" y="902" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">LLaMA-3 405B: 128-way ZeRO-1</text>
<rect x="28" y="920" width="248" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="939" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Ring AllReduce</text>
<text x="38" y="958" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">RS + AG → T ≈ 2·B / W</text>
<rect x="284" y="920" width="248" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="294" y="939" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">SHARP: sum inside the switch</text>
<text x="294" y="958" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">T ≈ B / W · traffic −50%</text>
<rect x="28" y="982" width="504" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Network wall: leaving the node, 450 → 50 GB/s per GPU (9×); TP stays at 8</text>
<line x1="100" y1="1047" x2="100" y2="1019" marker-end="url(#tri-a-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="110" y="1037" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:600">read: token stream 2 TB/s</text>
<line x1="338" y1="1019" x2="338" y2="1047" marker-end="url(#tri-a-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="348" y="1037" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:600">write: 7.5 TB checkpoints</text>
<rect x="16" y="1050" width="528" height="500" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-warn);stroke-width:1.4"/>
<text x="28" y="1074" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700">3 · STORAGE</text>
<text x="532" y="1074" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">datasets and checkpoints</text>
<text x="28" y="1094" text-anchor="start" style="fill:var(--c-text);font-size:12px">Question: can data and checkpoints flow without stalling GPUs?</text>
<rect x="28" y="1106" width="504" height="124" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1124" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">How does a byte get from disk to HBM?</text>
<text x="40" y="1142" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:600">Legacy: through the CPU page cache · 2 × memcpy, CPU at 100%</text>
<rect x="40" y="1150" width="96" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="88.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe SSD</text>
<rect x="162" y="1150" width="104" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="214.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">page cache</text>
<rect x="292" y="1150" width="104" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="344.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">user buffer</text>
<rect x="422" y="1150" width="98" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="471.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU HBM</text>
<line x1="137" y1="1161" x2="160" y2="1161" marker-end="url(#tri-a-bad)" style="stroke:var(--c-danger);stroke-width:1.4"/>
<line x1="267" y1="1161" x2="290" y2="1161" marker-end="url(#tri-a-bad)" style="stroke:var(--c-danger);stroke-width:1.4"/>
<line x1="397" y1="1161" x2="420" y2="1161" marker-end="url(#tri-a-bad)" style="stroke:var(--c-danger);stroke-width:1.4"/>
<text x="40" y="1194" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">GPUDirect Storage (cuFile): skips CPU and RAM · latency −80%, CPU ≈ 0%</text>
<rect x="40" y="1202" width="96" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="88.0" y="1217.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe / NVMe-oF</text>
<rect x="162" y="1202" width="234" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="279.0" y="1217.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe switch · direct DMA · 64+ GB/s</text>
<rect x="422" y="1202" width="98" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="471.0" y="1217.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU HBM</text>
<line x1="137" y1="1213" x2="160" y2="1213" marker-end="url(#tri-a-ok)" style="stroke:var(--c-success);stroke-width:1.4"/>
<line x1="397" y1="1213" x2="420" y2="1213" marker-end="url(#tri-a-ok)" style="stroke:var(--c-success);stroke-width:1.4"/>
<rect x="28" y="1238" width="248" height="148" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="1256" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Data feed</text>
<text x="38" y="1274" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">15+ trillion tokens (LLaMA-3)</text>
<text x="38" y="1290" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">.bin + .idx · zero-copy mmap</text>
<text x="38" y="1306" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NUMA: each socket feeds its 4 GPUs</text>
<text x="38" y="1322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16K GPUs: 2–4 TB/s sustained read</text>
<rect x="38" y="1336" width="228" height="42" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="46" y="1353" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:600">Meta Tectonic: 7,500 servers · 240 PB</text>
<text x="46" y="1370" text-anchor="start" style="fill:var(--c-text);font-size:11px">2 TB/s read · 7 TB/s burst write</text>
<rect x="284" y="1238" width="248" height="148" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="1256" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Checkpoint · 405B model</text>
<text x="294" y="1274" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16 B/param = 2 + 2 + 12</text>
<rect x="294.0" y="1284" width="24.6" height="18" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<rect x="318.6" y="1284" width="24.6" height="18" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<rect x="343.2" y="1284" width="147.7" height="18" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<rect x="491.0" y="1284" width="31.0" height="18" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="294" y="1313" width="10" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="310" y="1322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">weights BF16 · 0.81 TB</text>
<rect x="294" y="1329" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="310" y="1338" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradients BF16 · 0.81 TB</text>
<rect x="294" y="1345" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="310" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AdamW FP32 · 4.86 TB</text>
<rect x="294" y="1361" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="310" y="1370" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:600">+ activations, RNG → ~7.5 TB</text>
<rect x="28" y="1394" width="504" height="116" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1414" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Young-Daly: the optimal checkpoint interval</text>
<text x="520" y="1414" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">τ = √(2 · δ · M)</text>
<text x="40" y="1432" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">cluster MTBF: 26,000 h ÷ 2,048 ≈ 12.7 h; in practice 2–3 h (we use M = 3 h)</text>
<rect x="40" y="1442" width="100" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="90.0" y="1457.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">A · 25 GB/s</text>
<text x="150" y="1457" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">δ = 307 s → τ ≈ 42.9 min</text>
<rect x="340" y="1446" width="119.5" height="14" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1"/>
<text x="465.5" y="1457" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:600">loss ≈ 24%</text>
<rect x="40" y="1470" width="100" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="90.0" y="1485.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">B · 250 GB/s</text>
<text x="150" y="1485" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">δ = 31 s → τ ≈ 13.6 min</text>
<rect x="340" y="1474" width="37.5" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1"/>
<text x="383.5" y="1485" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">loss ≈ 7.5%</text>
<text x="40" y="1502" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">loss = δ/τ (writing) + τ/2M (rewinding) · 10× faster writes → √10 ≈ 3× less</text>
<rect x="28" y="1516" width="504" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="1531.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">I/O wall: a 7.5 TB checkpoint at 25 GB/s takes 307 s; 16,384 GPUs sit idle</text>
<rect x="16" y="1560" width="528" height="196" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="1582" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Bandwidth ladder · per GPU, H100 (log scale)</text>
<line x1="196" y1="1592" x2="196" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="196" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 GB/s</text>
<line x1="292" y1="1592" x2="292" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="292" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 GB/s</text>
<line x1="388" y1="1592" x2="388" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="388" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 TB/s</text>
<line x1="484" y1="1592" x2="484" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="484" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 TB/s</text>
<text x="28" y="1607" text-anchor="start" style="fill:var(--c-text);font-size:11px">L2 cache</text>
<rect x="196" y="1596" width="263.1" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="465.0748181914474" y="1607" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">~5.5 TB/s</text>
<text x="28" y="1626" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3</text>
<rect x="196" y="1615" width="242.4" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="444.4043014755371" y="1626" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3.35 TB/s</text>
<text x="28" y="1645" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (in node)</text>
<rect x="196" y="1634" width="158.7" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="360.708401322433" y="1645" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">450 GB/s</text>
<text x="28" y="1664" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU)</text>
<rect x="196" y="1653" width="77.4" height="14" rx="2" style="fill:var(--c-text-mute);fill-opacity:.45;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="279.3932775024532" y="1664" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">64 GB/s</text>
<text x="28" y="1683" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (off node)</text>
<rect x="196" y="1672" width="67.1" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="269.1011204162578" y="1683" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">50 GB/s</text>
<text x="28" y="1702" text-anchor="start" style="fill:var(--c-text);font-size:11px">legacy storage (whole cluster)</text>
<rect x="196" y="1691" width="38.2" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="240.2022408325156" y="1702" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">25 GB/s</text>
<text x="28" y="1746" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">The slowest rung sets the pace for the whole cluster: co-design all three.</text>
</svg>

*Figure 1.1: The three pillars at a glance. Compute (SM anatomy, memory hierarchy, roofline), network (NVLink node, InfiniBand fat-tree, 4D parallelism, SHARP) and storage (GDS path, data feed, checkpoint, Young-Daly), with the per-GPU bandwidth ladder at the bottom.*

Read the figure top to bottom. Each pillar opens with a question and closes with its wall in the red strip; the arrows between pillars are the links that join them. The ladder at the bottom puts every link on one axis: bandwidth drops about 7× from HBM to NVLink and 9× from NVLink to InfiniBand, and with legacy storage the whole cluster shares a pipe slower than a single GPU's InfiniBand link. The next three sections open the pillars one by one.

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
- **NUMA-Aware Prefetching:** Compute nodes feature two CPU sockets, each tied to a dedicated NUMA node and 4 GPUs. Workers are pinned to the specific CPU socket physically wired to the GPU's PCIe root complex, utilizing asynchronous CUDA streams (`pin_memory=True`) to stream tokens into HBM without crossing NUMA interconnects.

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

To consolidate our three pillars, consider pre-training LLaMA-3 70B and 405B on an NVIDIA H100 SuperPod:

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
- NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Official microarchitecture guide.
- On this blog: [A Prompt's Journey (8): The Output Layer (LM Head) — Megatron-LM and the Memory Wall](post.html?slug=inside-the-output-layer).
- On this blog: [A Prompt's Journey (9): Inside the Context Window — From the 2K Wall to FlashAttention](post.html?slug=inside-the-context-window).
