A training run makes progress at 38% MFU. `nvidia-smi` reports 100% GPU utilization. Checkpoints take several minutes to write, and some ranks constantly stall waiting for peers. Is the bottleneck GPU compute, network fabric, or storage?

**The central idea:** Diagnosing an AI cluster is not about staring at a single metric; it is about proving the physical location of waiting.

> **What we carry forward:** [The third article](post.html?slug=multi-gpu-networks-and-distributed-training) laid out parallel groups and tensor communication paths. We now connect physical bottlenecks and symptoms back to root causes when those paths stall.

> **By the end of this article**
>
> - You will separate storage stalls from GPU/kernel bottlenecks on a profiler timeline.
> - You will calculate optimal checkpoint intervals and wasted work budgets using cluster MTBF and the Young–Daly formula.
> - You will explain the conceptual and quantitative differences between GPU utilization, MFU, and HFU.
> - You will account for the missing 62% in a 38% MFU run across six physical components.
> - You will systematically resolve stragglers, silent data corruption, and fabric congestion scenarios in production systems.

**In this article**

- [1. How does storage make GPUs wait?](#1-how-does-storage-make-gpus-wait)
- [2. Why are failures normal in large clusters?](#2-why-are-failures-normal-in-large-clusters)
- [3. GPU utilization, MFU, and HFU](#3-gpu-utilization-mfu-and-hfu)
- [4. How do we diagnose low MFU?](#4-how-do-we-diagnose-low-mfu)
- [5. Five symptoms and five experiments](#5-five-symptoms-and-five-experiments)
- [6. Which metric connects performance to cost and energy?](#6-which-metric-connects-performance-to-cost-and-energy)
- [7. Seven core system design questions](#7-seven-core-system-design-questions)
- [What questions can the reader now answer?](#what-questions-can-the-reader-now-answer)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [References and verification links](#references-and-verification-links)

---

## 1. How does storage make GPUs wait?

In an AI supercomputer, while tensor math churns between HBM and Tensor Cores at peak line rate, the slowest subsystem in the data center—storage (NVMe arrays and distributed parallel filesystems)—can effortlessly starve GPUs. On its journey from storage to Tensor Cores, a byte can hit three physical walls: the **I/O Wall**, the **Network Wall**, and the **Memory Wall**.

<!-- restored-svg:en_full:1 -->
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

### 1.1. Reads and writes are different bottlenecks

In large language model training, the storage workload is highly asymmetric; dataset reads and checkpoint writes exhibit diametrically opposing traffic profiles:

1. **Dataloader Ingestion Stream (Continuous, Low-Bandwidth):**
   - Consider a massive LLM training step processing $16\text{ million}$ tokens globally. Token IDs are encoded as uint16 (2 bytes) or int32 (4 bytes).
   - Raw data ingested per global step:
     $$16 \times 10^6 \times 4\text{ bytes} = 64\text{ MB}$$
   - If a step takes $1.5\text{ seconds}$, aggregate cluster-wide read throughput is only $\approx 42.6\text{ MB/s}$. In a 1,024-GPU cluster, per-GPU read demand is just a few kilobytes per second!
   - *The Real Bottleneck:* Not raw throughput, but filesystem metadata lock contention (POSIX open/stat storms) when thousands of worker processes concurrently traverse directories, accompanied by host CPU dataloader jitter.
2. **Checkpoint Burst Wave (Synchronous, Massive Surge):**
   - At checkpoint intervals, every GPU must dump its complete model weights, optimizer states (FP32 master weights, momentum, variance), and scheduler state to persistent storage.
   - For a 405-billion parameter model, a single cluster-wide snapshot exceeds **$7.5\text{ TB}$**!
   - When 16,384 GPUs flush this massive tensor payload to a shared parallel storage pool (Lustre/Ceph) within the same second, an intense write tsunami ($> 250\text{ GB/s}$) hits the storage fabric.

<!-- restored-svg:en_full:37 -->
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

### 1.2. Checkpoint lower bound

Under traditional synchronous (blocking) checkpointing, all GPU computation halts while the snapshot writes out. The duration of this stall is strictly bounded by the sustained write throughput of the storage backend:

$$
T_{\mathrm{chk}} = \frac{M_{\mathrm{state}}}{B_{\mathrm{storage}}}
$$

**Numerical Walk:**
For a model state $M_{\mathrm{state}} = 7.5\text{ TB}$ ($7,500\text{ GB}$) on a storage backend with sustained write bandwidth of $25\text{ GB/s}$:
$$T_{\mathrm{chk}} = \frac{7,500\text{ GB}}{25\text{ GB/s}} = 300\text{ seconds} = 5.0\text{ minutes}$$
This means 16,384 GPUs sit completely idle for 5 full minutes during every checkpoint step. In a supercomputer costing tens of thousands of dollars per hour, this represents an intolerable efficiency loss.

### 1.3. Asynchronous checkpointing

To eliminate this 5-minute freeze, production AI platforms implement **Asynchronous Two-Tier Checkpointing (Non-blocking Snapshot)**:

1. **In-Memory Snapshot (Copy-on-Write):** At the checkpoint step, GPUs pause training for only 1–2 seconds. Optimizer states and model weights are DMA-copied from GPU HBM to local CPU host RAM or local node NVMe via high-speed PCIe Gen5 ($> 50\text{ GB/s}$).
2. **Background Drain:** The instant the local memory copy finishes, GPUs immediately resume compute on the next training step (Forward GEMM).
3. **Asynchronous Distributed Flush:** Background host daemon processes asynchronously stream the snapshot from host RAM over the network to the central parallel filesystem. GPU math execution is completely insulated from remote storage I/O latency.

### 1.4. What does GPUDirect Storage not solve?

NVIDIA GPUDirect Storage (GDS) establishes a direct DMA (Direct Memory Access) path between local NVMe drives or remote storage NICs and GPU HBM (`cuFile` API), bypassing host CPU RAM entirely.

<!-- restored-svg:en_full:42 -->
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

A 4-step engineering breakdown of what GDS does and does not solve:

1. **Eliminates Host Bounce Buffers:** In classical I/O, data copies from storage to an intermediate CPU RAM bounce buffer, then over PCIe to HBM. GDS collapses this into a single DMA transfer.
2. **Reduces CPU and PCIe Contention:** Host CPU cores are freed from memory-copy loops; redundant traffic crossing the host PCIe root complex is halved.
3. **What It Does Not Solve (Physical Storage Limits):** If the underlying NVMe tier or distributed Lustre storage servers cap out at $25\text{ GB/s}$ physically, enabling GDS cannot turn that pipeline into $100\text{ GB/s}$. Ingestion speed remains bounded by the source.
4. **The Compatibility Mode Trap:** If kernel drivers or filesystem mounts lack proper GDS integration, cuFile silently falls back to traditional CPU bounce-buffer staging; profilers might report GDS enabled while performance remains unchanged.

### 1.5. Which experiment separates dataset and checkpoint bottlenecks?

When an AI cluster exhibits storage-induced stalls (Input Stalls), use this synthetic isolation experiment to identify the root cause:

- **Synthetic In-Memory Dataloader Test:** Replace disk I/O with synthetic, pre-allocated dummy tensors in GPU HBM, running 100 training steps without touching disk.
- *Outcome A:* If step time drops significantly and MFU jumps, the bottleneck lies strictly in the dataset ingestion pipeline (insufficient dataloader workers, disk IOPS saturation, or CPU token preprocessing).
- *Outcome B:* If step time remains flat, storage ingestion is innocent; stalls originate from compute kernels or collective network synchronization.
- *Checkpoint Isolation:* Double the checkpoint interval (e.g., from 30 minutes to 60 minutes). If average hourly step time improves, checkpoint write bursts are degrading cluster MFU.

---

## 2. Why are failures normal in large clusters?

In a single server, a hardware failure is a rare anomaly. In a 16,384-GPU supercomputer, hardware failure is not an exception; it is the **continuous baseline operating condition** of the system.

<!-- restored-svg:en_full:43 -->
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

### 2.1. Cluster-MTBF intuition

As component counts grow, independent failure probabilities compound. The Mean Time Between Failures (MTBF) of a cluster is calculated by dividing the MTBF of an individual node by the total node count:

$$
\mathrm{MTBF}_{\mathrm{cluster}} = \frac{\mathrm{MTBF}_{\mathrm{node}}}{N_{\mathrm{nodes}}}
$$

Assume a single DGX H100 node (including 8 GPUs, dual CPUs, memory, power supplies, and 8 InfiniBand NICs) is exceptionally reliable, exhibiting an annual hardware failure rate of only 10%:
- Single-node MTBF $\approx 10\text{ years} \approx 87,600\text{ hours}$.
- In a cluster of 2,048 DGX servers (16,384 GPUs):
  $$\mathrm{MTBF}_{\mathrm{cluster}} = \frac{87,600\text{ hours}}{2,048} \approx 42.7\text{ hours}$$
And that accounts solely for server node faults! When factoring in optical transceivers, cable bends, switch ports, cooling pumps, and software kernel crashes, the cluster MTBF drops to mere hours.

**Meta Llama 3 405B Empirical Data:**
Meta's technical report documented **466 job interruptions** over 54 days of 405B pre-training. Of those, 419 were unexpected hardware failures. This empirical record proves that the cluster experienced an unexpected failure every **3.09 hours**!

### 2.2. How frequently should we checkpoint?

With failures occurring every few hours, how often should checkpoints be taken?
- **Too frequently:** Training time is squandered on constant checkpoint write stalls.
- **Too infrequently:** When a failure inevitably occurs, hours of progress between the last valid snapshot and the crash are lost, requiring costly recomputation.

This trade-off is resolved mathematically by the **Young–Daly Optimum Checkpoint Formula**:

$$
\tau_{\mathrm{opt}} \approx \sqrt{2 \times \delta \times M}
$$

Where:
- $\tau_{\mathrm{opt}}$: Optimal interval between checkpoints (minutes or hours),
- $\delta$: Time required to write a checkpoint to persistent storage,
- $M$: Cluster MTBF ($\mathrm{MTBF}_{\mathrm{cluster}}$).

<!-- restored-svg:en_full:44 -->
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

**Numerical Walk and the Young–Daly Curve:**
Consider a cluster with an MTBF of $M = 3.0\text{ hours} = 180\text{ minutes}$ across two storage performance regimes:

- **Scenario A (Slow Storage - $\delta = 5.0\text{ minutes} = 300\text{ seconds}$):**
  $$\tau_{\mathrm{opt}} = \sqrt{2 \times 5.0 \times 180} = \sqrt{1,800} \approx 42.4\text{ minutes}$$
  Checkpoints should occur every 42.4 minutes. Total wasted cluster time (writing overhead + expected recomputation upon crash) consumes **23.6%** of all operating hours!
- **Scenario B (Fast / Asynchronous Storage - $\delta = 0.5\text{ minutes} = 30\text{ seconds}$):**
  $$\tau_{\mathrm{opt}} = \sqrt{2 \times 0.5 \times 180} = \sqrt{180} \approx 13.4\text{ minutes}$$
  The optimal checkpoint interval drops to 13.4 minutes, shrinking total wasted time to **7.5%**. Faster storage not only cuts snapshot write overhead; it drastically curtails the volume of lost work per crash!

### 2.3. Checkpoints are not enough

Taking checkpoints is only the first component of resilience. When a GPU or optical transceiver faults, rapid cluster recovery requires an automated **Node Quarantine and Remediation Loop**:

<!-- restored-svg:en_full:45 -->
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

The 4 phases of an automated quarantine loop:

1. **Failure Detection:** The monitoring fabric catches DCGM Xid events, PCIe CRC errors, or training step latency spikes, flagging the faulty rank.
2. **Drain and Quarantine:** The faulty node is drained immediately and isolated into a diagnostic triage pool.
3. **Multi-Tier Diagnostics:** Automated test suites run on the quarantined node: Level 1 DCGM checks, single-node cuBLAS GEMM stress tests, and pairwise NCCL bandwidth tests.
4. **Hotswap and Resume:** The job scheduler swaps in a pre-warmed healthy spare node; training automatically resumes from the last valid checkpoint in 2–3 minutes. The degraded server opens a ticket for hardware RMA.

### 2.4. Silent Data Corruption

The most insidious hardware defect is not a machine crash; it is a machine computing corrupt results without alerting the OS. This is **Silent Data Corruption (SDC)** or **Silent Data Errors (SDE)**.

- **Mechanism:** A single bit flips ($0 \to 1$) in an SM ALU register or L2 cache data path, escaping DRAM ECC detection.
- **Impact:** Neither the operating system nor the CUDA runtime flags an error. However, a matrix multiplication evaluates to $10^9$ instead of $1.0$. In backward propagation, this defect triggers a NaN/Inf loss explosion or silently degrades model language capabilities.
- **Detection:** Ranks run periodic deterministic canary GEMM kernels against known reference checksums; loss monitor daemons alert on sudden gradient norm anomalies, isolating suspect SMs.

### 2.5. How do we calculate loss from checkpoint interval?

Total wasted cluster capacity ($W_{\mathrm{total}}$) driven by checkpointing and failures decomposes into three components:

$$
W_{\mathrm{total}} = \frac{\delta}{\tau} + \frac{\tau}{2M} + \frac{R}{M}
$$

Where:
- $\frac{\delta}{\tau}$: Fraction of time spent writing checkpoints,
- $\frac{\tau}{2M}$: Average fraction of work lost between the last checkpoint and a crash that must be recomputed,
- $\frac{R}{M}$: Fraction of time spent recovering, rebooting ranks, and reloading weights after each failure ($R$).

This formula highlights the direct mathematical leverage of fast storage ($\delta$) and rapid automated recovery ($R$) on cluster ROI.

---

## 3. GPU utilization, MFU, and HFU

The gravest mistake an AI infrastructure engineer can make is looking at a 100% reading in `nvidia-smi` and concluding that hardware is running efficiently.

| Metric | What Does It Measure? | Common Misleading Scenario | When to Trust It? |
|---|---|---|---|
| **GPU Utilization (`nvidia-smi`)** | Was any CUDA kernel active on any SM during the sample window? | A GPU stalled on memory, doing uncoalesced memory lookups or spin-waiting shows 100% | Never sufficient on its own as a performance benchmark |
| **Model FLOPs Utilization (MFU)** | Ratio of useful algorithmic theoretical model FLOPs to peak hardware FLOPS | Excludes activation recomputation FLOPs; reflects true algorithmic productivity | The gold standard for measuring distributed training efficiency |
| **Hardware FLOPs Utilization (HFU)** | Ratio of all physical floating-point operations executed on silicon to peak FLOPS | Inflated by recomputing forward passes repeatedly to save memory | Useful for assessing raw silicon arithmetic saturation |

### 3.1. GPU utilization

The utilization percentage reported by `nvidia-smi` is a coarse sampling metric. Over a 1-second window, if a single 1-microsecond CUDA kernel executes on just one of the GPU's 132 SMs, that sample interval registers as active.
Even if Tensor Cores sit completely idle for 500 clock cycles awaiting DRAM fetch, `nvidia-smi` happily reports **100% Utilization**. It is a binary presence signal, not a throughput metric.

### 3.2. MFU

**Model FLOPs Utilization (MFU)** measures the theoretical minimum floating-point operations required by the model architecture divided by the theoretical peak compute capacity of the hardware over the observed time window:

$$
\mathrm{MFU} = \frac{\text{Observed Tokens per Second} \times \text{Model FLOPs per Token}}{N_{\mathrm{GPU}} \times \text{Peak TFLOP/s per GPU}}
$$

For a standard dense Transformer (excluding activation recomputation), training requires approximately **$6P$ FLOPs per parameter ($P$) per token** ($2P$ in forward, $4P$ in backward):
$$\text{FLOPs per token} \approx 6P$$
MFU reflects true algorithmic throughput and serves as the industry's most rigorous efficiency benchmark.

### 3.3. HFU

**Hardware FLOPs Utilization (HFU)** accounts for every floating-point operation physically executed on Tensor Cores.
When memory pressure requires **Activation Checkpointing (Recomputation)**, forward passes of certain layers run a second time during backward ($+2P$ additional FLOPs).
Hardware executes $8P$ FLOPs per token instead of $6P$:
$$\mathrm{HFU} \approx \frac{8P}{6P} \times \mathrm{MFU} \approx 1.33 \times \mathrm{MFU}$$
HFU gauges how close silicon operates to theoretical peak math capacity, but a portion of that work is an architectural tax on memory constraints.

### 3.4. A 38% MFU sanity check

When an engineering team trains a 70B model across 1,024 NVIDIA H100 GPUs and logs **38% MFU**, is that a success?
An H100 SXM5 GPU delivers $989.5 \approx 990\text{ TFLOP/s}$ of dense BF16 math. A 1,024-GPU cluster provides a theoretical ceiling of:
$$1,024 \times 990\text{ TFLOP/s} \approx 1.014\text{ PFLOP/s} = 1.014\text{ EFLOP/s}$$
At 38% MFU, the cluster delivers useful model math at:
$$\text{Useful Compute} = 0.38 \times 1.014\text{ EFLOP/s} \approx 0.385\text{ EFLOP/s}$$

<!-- restored-svg:en_full:46 -->
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

Where did the **remaining 62% of compute potential go?** Are 62% of the GPUs switched off?
No! The missing 62% fraction divides across six physical factors:

1. **Memory-Bound Layers (10–12%):** LayerNorm, RMSNorm, RoPE, Softmax, and bias additions exhibit low arithmetic intensity; they stall on HBM memory bandwidth rather than Tensor Cores.
2. **Unhidden Collective Communication (12–15%):** Intra-layer TP AllReduces, PP stage transfers, and unhidden FSDP ReduceScatter queues.
3. **Pipeline Parallelism Bubble (8–12%):** Idle time spent draining and filling pipeline stages under 1F1B scheduling ($F_{\mathrm{bubble}}$).
4. **Recomputation Gap (HFU vs. MFU) (10–14%):** Activation GEMMs executed to conserve HBM (counted in HFU, excluded from MFU).
5. **Dataloader, Checkpoint, and Straggler Jitter (5–8%):** Disk ingestion stalls, snapshot pauses, and thermal clock throttling on outlier ranks.
6. **Tensor Core Sub-Optimal Tiling (4–6%):** Quantization and matrix dimension mismatches ($M, N, K$) failing to perfectly tile across 16-element Tensor Core execution units.

Accounting for these realities demonstrates that 38% MFU represents a highly tuned, world-class engineering accomplishment.

### 3.5. Which assumptions affect an MFU calculation?

Three subtle traps that distort MFU calculations:
1. **The Structural Sparsity Trap:** NVIDIA datasheets cite $1,979\text{ TFLOP/s}$ for H100 by assuming 2:4 structured sparsity. Dense foundation models cannot leverage 2:4 sparsity during pre-training; calculations must strictly use dense peak ratings ($989.5\text{ TFLOP/s}$).
2. **Padding Tokens:** If padding tokens in sequence batches are counted as valid model FLOPs, MFU appears artificially inflated; only non-pad tokens count.
3. **Attention FLOP Formula:** In long-context runs ($S > 8\text{K}$), omitting the quadratic attention FLOP term ($4 \times L \times H \times S$) artificially depresses reported MFU by 3–5%.

---

## 4. How do we diagnose low MFU?

When cluster MFU falls below targets, altering training hyperparameters at random guarantees confusion. Engineers follow a disciplined diagnostic framework.

### 4.1. Freeze the comparison first

Before initiating diagnostics, lock all foundational variables:
- Freeze global batch size and microbatch size.
- Lock sequence length and numerical precision (BF16).
- Keep optimizer (AdamW) and 4D parallel topology ($TP, PP, DP$) unchanged.
- Execute runs for at least 100 steps, discarding the first 10 warmup steps.

### 4.2. Classify waiting

Decompose GPU time into four fundamental stall categories:

1. **Compute Stall:** The GPU computes matrix operations, but kernels run inefficiently (small matrices, register spilling).
2. **Communication Stall:** The GPU blocks inside an NCCL collective (AllReduce, AllToAll) awaiting peers.
3. **Memory Bandwidth Stall:** Tensor Cores stall while SMs wait on HBM DRAM cache lines.
4. **I/O & Input Stall:** The GPU sits ready, but the CPU dataloader fails to supply the next batch.

<!-- restored-svg:en_full:41 -->
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

<!-- restored-svg:en_full:47 -->
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

### 4.3. Three observation layers

A robust diagnostic regime collects and correlates signals across three observational tiers simultaneously:

1. **Layer 1 — Hardware and Driver Telemetry (DCGM):**
   - GPU core temperatures and thermal throttling flags (`SW Thermal Slowdown`).
   - Core and memory clock frequencies (`Clocks Throttle Reasons`).
   - PCIe link width and speed (flagging links downgraded from $x16$ to $x8$ or $x4$).
   - DRAM single-bit and double-bit ECC errors.
2. **Layer 2 — Network and Fabric Telemetry:**
   - InfiniBand port counters: `PortXmitWait` (congestion), `SymbolError` (faulty optics).
   - RoCE Ethernet counters: PFC Pause frame counters, ECN congestion markers, buffer overruns.
   - NCCL telemetry: Collective latency distribution histograms and straggler detection.
3. **Layer 3 — Framework and Profiler Traces:**
   - PyTorch Profiler / NVIDIA Nsight Systems (nsys) timeline traces.
   - CUDA stream timelines, kernel execution durations, and exposed communication spans.

<!-- restored-svg:en_full:48 -->
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

### 4.4. One hypothesis, one experiment

The cardinal rule of systems debugging: **Never modify multiple variables simultaneously.**
- *Hypothesis:* "The step latency spike is caused by dataloader storage read congestion."
- *Experiment:* Switch dataloaders to synthetic in-memory dummy tensors; do not touch model or network settings.
- *Decision:* If step time recovers, the hypothesis is confirmed; if step time remains stalled, discard the hypothesis and move to the next layer.

### 4.5. Why does the order of the diagnostic tree matter?

The diagnostic tree must always be traversed **bottom-up (from physical hardware to software layers)**:

```
[ Hardware Health (DCGM / ECC) ] -> [ Network Fabric (RDMA / Links) ] -> [ Storage & Dataloader ] -> [ Kernel & MFU ]
```

Why?
If a GPU heatsink has unseated, causing thermal throttling to slash core clock frequencies from $1,980\text{ MHz}$ to $800\text{ MHz}$, spending days tuning FlashAttention CUDA kernels in PyTorch is futile! Physical integrity must be established first.

<!-- restored-svg:en_full:49 -->
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

---

## 5. Five symptoms and five experiments

Five recurring failure patterns in production clusters and their precise root-cause isolation tests:

### 5.1. `nvidia-smi` shows 100%, tokens/s is low
- **Root Cause:** Kernels with exceptionally low arithmetic intensity (such as microbatches too small to saturate Tensor Cores) or unfused element-wise operations.
- **Experiment:** Capture an Nsight Systems trace. If SMs spend most execution time in `at::native::elementwise_kernel`, apply `torch.compile()` for kernel fusion and increase per-GPU microbatch size.

### 5.2. AllReduce is slower than expected
- **Root Cause:** A straggler rank is lagging behind, or a misconfigured GPU-to-NIC binding is crossing the host CPU UPI bridge.
- **Experiment:** Inspect topology logs via `NCCL_DEBUG=INFO` and `NCCL_TOPO_DUMP=1`. Run pairwise `nccl-tests` between node pairs to isolate degraded switch ports or asymmetric rails.

### 5.3. Mean TTFT is good, p99 is bad
- **Root Cause:** Inference engines experience KV-cache memory allocation lock contention, or long prompts in multi-tenant queues starve concurrent decode iterations.
- **Experiment:** Enable Chunked Prefill (splitting prompt prefill across iterations) and optimize vLLM PagedAttention block size from 16 to 32.

### 5.4. Checkpoints take minutes
- **Root Cause:** All ranks simultaneously attempt synchronous writes to a shared directory, overwhelming storage metadata servers.
- **Experiment:** Transition to two-tier asynchronous checkpointing: write snapshots to local host NVMe (`/tmp`), streaming to distributed storage in the background.

### 5.5. A few GPUs are always behind
- **Root Cause:** Thermal throttling or PCIe bus link width degradation.
- **Experiment:** Execute `nvidia-smi -q -d PERFORMANCE,CLOCK`. Check `Clocks Throttle Reasons`. If `SW Thermal Slowdown` or `PCIe Link Width: 8x` appears, dispatch hardware technicians to replace thermal paste or reseat PCIe risers.

### 5.6. Case: Why did MFU fall from 42% to 22%?

**Real-World Production Incident:**
In a 1,024-GPU cluster running a 70B training job, MFU suddenly plummeted from 42% to 22% following scheduled weekend maintenance. Training throughput collapsed by half.

Step-by-step investigation:
1. *Hardware Scan:* DCGM diagnostics ran clean; no GPUs were experiencing thermal or clock throttling.
2. *Dataloader Test:* A synthetic in-memory dataloader test left MFU unchanged at 22% (storage was blameless).
3. *Network Trace:* Profiler traces revealed backward-pass AllReduce latency had jumped from $12\text{ ms}$ to $48\text{ ms}$.
4. *Root Cause:* During switch reboots, 16 ConnectX-7 leaf ports had reverted from MTU 9000 (Jumbo Frames) to the default MTU 1500! Massive packet fragmentation flooded the fabric, triggering severe RoCE PFC pause storms.
5. *Remediation:* Switch MTUs were restored to 9000; MFU instantly rebounded to 42%.

### 5.7. A short incident-response template

When cluster performance drops or alarms fire, execute this 4-step protocol:
1. **Contain:** Identify and isolate degraded nodes to quarantine; resume training with healthy spares from the last clean checkpoint.
2. **Preserve:** Do not reboot nodes blindly! Dump DCGM logs, `dmesg`, NCCL debug traces, and switch port statistics.
3. **Reproduce:** Execute isolated synthetic `nccl-tests` and `cuBLAS` stress runs on the quarantined host to trigger the failure mode.
4. **Remediate & Verify:** Apply hardware or configuration fixes, run Level 1–4 acceptance tests, and stage the node back into production.

---

## 6. Which metric connects performance to cost and energy?

Cluster engineering cannot judge efficiency purely by token velocity; token production must be anchored to FinOps realities: dollars spent and kilowatt-hours consumed.

### 6.1. GPU-hours are not enough

Standard cloud billing tracks "GPU-Hours". This metric is economically blind.
- Team A: Spends 1,000 GPU-hours at 20% MFU, training on 100 billion tokens.
- Team B: Spends 1,000 GPU-hours at 40% MFU, training on 200 billion tokens.
Both teams receive identical cloud invoices! Yet Team B's **Cost per Useful Token** is exactly 50% lower. Altyapı success is judged by tokens per dollar, not rented hours.

### 6.2. TCO components

Total Cost of Ownership (TCO) extends far beyond silicon purchase prices (CapEx):

$$
\mathrm{TCO} = \mathrm{CapEx}_{\mathrm{GPU+Servers}} + \mathrm{CapEx}_{\mathrm{Network+Storage}} + \mathrm{OpEx}_{\mathrm{Energy}} \times \mathrm{PUE} + \mathrm{OpEx}_{\mathrm{Facility+Cooling}}
$$

- **Power and PUE Multiplier:** A 1,024-GPU H100 cluster consumes **$1.5 - 2.0\text{ Megawatts}$** of continuous power. At a Power Usage Effectiveness (PUE) of 1.3, an additional 0.3 MW of cooling energy is expended for every 1 MW of compute.
- Over a 3-year lifecycle, electricity and facilities cooling costs can exceed the initial hardware acquisition price!

### 6.3. Tokens per joule

The fundamental metric of sustainable AI infrastructure is tokens generated per Joule:

$$
\text{Energy Efficiency} = \frac{\text{Total Tokens Processed}}{\text{Total Energy Consumed (Joules)}}
$$

Elevating MFU from 25% to 40% cuts the megawatt-hours and carbon emissions required to train a model by 37.5%.

### 6.4. Higher MFU is not always better

The final trap for systems engineers is MFU fetishization.
MFU can be artificially inflated by ballooning the global batch size (e.g., from 4M to 32M tokens). Larger matrices saturate Tensor Cores, pushing MFU past 50%.
However, exceeding the **Gradient Noise Scale** yields diminishing algorithmic returns per step. Total steps required for loss convergence multiply, extending wall-clock training time and increasing aggregate cloud expenditure! The ultimate objective is not peak MFU, but **minimizing total dollar and energy expenditure to target validation loss**.

---

## 7. Seven core system design questions

### 7.1. GPU utilization versus MFU?

GPU Utilization from `nvidia-smi` measures the percentage of time that at least one CUDA kernel was active on any SM. A GPU stalled on memory bandwidth, locked in a spin-loop, or executing uncoalesced loads reports 100% utilization while accomplishing little mathematical progress.

MFU (Model FLOPs Utilization) measures the ratio of useful algorithmic floating-point operations executed to the theoretical peak compute capacity of the silicon. It evaluates true mathematical productivity rather than mere activity.

### 7.2. Why can MFU not reach 100%?

Five fundamental physical constraints prevent MFU from reaching 100%:
1. Memory-bound layers (LayerNorm, Softmax, RoPE) have low arithmetic intensity, saturating HBM bandwidth rather than Tensor Cores.
2. Distributed communication barriers (TP AllReduces, PP bubbles, gradient sync) impose unhidden latency.
3. Finite SRAM and L2 cache sizes require tiled transfers from HBM, incurring memory latencies.
4. Matrix dimensions failing to align with Tensor Core warp tile multiples (16x16x16) incur padding waste.
5. Storage ingestion jitter and checkpoint writes inject exposed stalls into the critical path.

### 7.3. Why can HFU exceed MFU?

Hardware FLOPs Utilization (HFU) tallies all floating-point operations physically executed by Tensor Cores. When activation memory pressure requires **Activation Checkpointing (Recomputation)**, forward passes for selected layers execute a second time during backpropagation.

This redundant forward math generates physical FLOPs on silicon, raising HFU. However, because these recomputed operations yield no new algorithmic output, they are excluded from MFU. Consequently, HFU typically exceeds MFU by 30–35% in systems using activation checkpointing.

### 7.4. Why does faster checkpointing help twice?

Accelerating checkpoint writing (e.g., from 5 minutes to 30 seconds) delivers compounding benefits:
1. **Direct Write Savings:** GPU stall duration during each checkpoint interval drops immediately by 4.5 minutes.
2. **Reduced Lost Work (Young-Daly Effect):** Under the Young-Daly equation ($\tau_{\mathrm{opt}} \approx \sqrt{2\delta M}$), faster writes ($\delta$) justify taking checkpoints far more frequently. More frequent snapshots drastically cut the average steps lost and recomputed when an unexpected failure occurs.

### 7.5. What is the first metric to inspect?

When investigating cluster throughput regressions, the first telemetry to inspect is **DCGM Hardware Health and Throttling Flags**.

If any single GPU throttles clocks due to thermal stress (`SW Thermal Slowdown`), drops PCIe bandwidth ($x16 \to x8$), or logs double-bit DRAM ECC errors, it turns into a straggler, stalling the entire cluster at collective synchronization barriers. Higher-level software metrics are meaningless until physical hardware integrity is confirmed.

### 7.6. How do you find a straggler?

Isolating a straggler among thousands of GPUs relies on a 3-step triage process:
1. **Step-Time Distribution:** Plot forward and backward latencies across all ranks; identify ranks skewing the P-99 tail.
2. **NCCL Collective Timing:** In profiler traces, the ranks entering the AllReduce barrier first are waiting peers; the rank entering the barrier last is the true culprit causing the delay.
3. **Hardware Telemetry Correlation:** Cross-reference the laggard rank with DCGM clock frequencies, PCIe link counters, and switch port `PortXmitWait` counters to pinpoint hardware degradation.

### 7.7. Why is a microbenchmark not production performance?

Synthetic microbenchmarks (like isolated `nccl-tests` or raw GEMM loops) fail to reflect production cluster dynamics:
1. They generate tensors synthetically in memory, bypassing filesystem metadata contention and dataloader jitter.
2. They run for minutes, failing to expose long-term memory fragmentation, thermal saturation, or gradual memory leaks.
3. They omit the complex interplay of asynchronous optimizer updates, dynamic sequence lengths, activation caching, and bursty checkpoint waves.

---

### What questions can the reader now answer?

- How do you differentiate a storage ingestion stall from an unhidden collective network delay on a profiler trace?
- How do you apply the Young-Daly formula to determine optimal checkpoint frequency given cluster MTBF?
- How do you explain the missing 62% compute potential in a 38% MFU training run across six physical components?
- Why can `nvidia-smi` report 100% utilization while model training makes negligible forward progress?
- How does an automated node quarantine loop prevent degraded hardware from cascading into cluster-wide job failures?

---

## The whole story in six lines

- Cluster performance diagnosis is not about staring at a single metric; it is about proving the physical location of waiting.
- Dataloader reads bottleneck on filesystem metadata locks, while checkpoints create intense write tsunamis.
- In multi-thousand-GPU clusters, failure is baseline; optimal checkpoint frequency is governed by the Young-Daly formula.
- GPU utilization merely confirms silicon activity; true algorithmic throughput is reflected solely by MFU.
- The 62% gap behind 38% MFU decomposes into memory-bound layers, collectives, pipeline bubbles, and recomputation.
- Robust diagnostics traverse bottom-up from physical hardware to software using single-variable controlled experiments.

---

## Glossary

- **MFU (Model FLOPs Utilization):** Ratio of useful theoretical algorithmic floating-point operations to peak hardware capacity.
- **HFU (Hardware FLOPs Utilization):** Ratio of all physically executed floating-point operations (including recomputation) to peak capacity.
- **MTBF (Mean Time Between Failures):** Expected average operating time elapsed between consecutive hardware or system failures.
- **Young–Daly Formula:** Mathematical model balancing checkpoint write time and MTBF to determine optimal snapshot intervals.
- **Straggler:** A degraded GPU that executes slower due to thermal throttling or hardware faults, blocking all ranks at collective barriers.
- **GPUDirect Storage (GDS):** Architecture enabling direct DMA transfers between NVMe storage and GPU HBM, bypassing CPU host RAM.
- **Silent Data Corruption (SDC):** Unflagged bit flips in silicon computing corrupt math without raising operating system or driver exceptions.
- **Quarantine Loop:** Automated infrastructure workflow isolating, diagnosing, and remediating faulty nodes.
- **Tokens per Joule:** Green AI efficiency metric quantifying useful tokens generated per unit of electrical energy consumed.

---

## References and verification links

- Previous article: [AI Infrastructure (3): Multi-GPU Networks and Distributed Training](post.html?slug=multi-gpu-networks-and-distributed-training)
- Next article: [AI Infrastructure (5): Enterprise AI Platform Architecture — Hardware, Cost, and Operations](post.html?slug=enterprise-ai-platform-architecture)
- **[K1] Llama 3 405B Cluster Experience:** Meta, [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) (Section 3: Infrastructure, MTBF, and Failure Analysis).
- **[K2] Checkpointing Theory:** J. T. Daly, [*A higher order estimate of the optimum checkpoint interval for restart dumps*](https://doi.org/10.1016/j.jpdc.2004.02.004).
- **[K3] MFU and PaLM Analysis:** Google Research, [*PaLM: Scaling Language Modeling with Pathways*](https://arxiv.org/abs/2204.02311).
- **[K4] GPUDirect Storage:** NVIDIA, [*GPUDirect Storage Architecture Guide*](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).
- **[K5] DCGM and Observability:** NVIDIA, [*DCGM User Guide*](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html).
