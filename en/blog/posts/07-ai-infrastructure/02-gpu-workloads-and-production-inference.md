Whether pre-training a foundation model (training), ingesting an extensive prompt context for the first time (prefill), or streaming an answer token by token (decode), the exact same GPU silicon executes the workload. The underlying Transformer blocks, tensor projections, and weight matrices remain identical. Yet the hardware collides with fundamentally different physical ceilings across these three regimes: training demands billions of arithmetic FLOPs and immense activation memory; prompt ingestion requires wide, compute-heavy parallel matrix multiplications; and token-by-token generation grinds to a halt against memory bus bandwidth.

Imagine deploying a 70-billion-parameter model on a production inference cluster. A user submits a 2,000-token legal contract, requesting an executive summary. The instant the request lands, the GPU's Tensor Cores achieve 70% operational efficiency, devouring thousands of prompt tokens in seconds. But the moment the model begins generating the response, that same supercomputing silicon slows dramatically. To emit each subsequent word, all 140 gigabytes of model weights must stream completely from HBM into the registers—burning memory bandwidth while arithmetic execution units sit idle.

**The core idea of this article:** In large language model serving, "inference cost" is not a monolithic number. Systems operate across two diametrically opposed physical regimes: the compute-bound **prefill** phase (which digests prompt context) and the memory-bandwidth-bound **decode** phase (which generates tokens autoregressively). The secret to scaling production serving lies in managing KV cache growth, continuous batching schedulers, and physical memory fragmentation to ensure the hardware never starves.

> **Our place in the series:** This article is the second installment of our five-part *AI Infrastructure* series. In Part 1, we established the physical hardware map: SM microarchitecture, the memory pyramid, and the Roofline model. Here, we analyze the workloads that run on that hardware: the distinct memory, KV cache, and latency dynamics of training, prefill, and decode. In Part 3, we scale these workloads across thousands of GPUs using high-speed network fabrics and multi-dimensional parallelism.

> **By the end of this article, you will be able to:**
>
> - Mathematically prove why the exact same Transformer layer encounters opposing bottlenecks in training, prefill, and decode.
> - Calculate the itemized memory footprint of model weights, KV cache, and activations from first principles.
> - Demonstrate how Grouped-Query Attention (GQA) and Multi-Head Latent Attention (MLA) compress the KV cache budget by orders of magnitude.
> - Master the hardware-level mechanics of continuous batching, PagedAttention, and prefix caching.
> - Evaluate disaggregated serving (splitting prefill and decode pools) to distinguish genuine throughput gains from network cost traps.

**In this article**

- [1. Why is the same Transformer three different workloads?](#1-why-is-the-same-transformer-three-different-workloads)
- [2. How to construct a complete memory budget](#2-how-to-construct-a-complete-memory-budget)
- [3. How serving engines keep the GPU saturated](#3-how-serving-engines-keep-the-gpu-saturated)
- [4. Planning capacity and latency together](#4-planning-capacity-and-latency-together)
- [5. What changes with vision and video workloads?](#5-what-changes-with-vision-and-video-workloads)
- [6. A concrete 70B serving case study](#6-a-concrete-70b-serving-case-study)
- [7. Six essential system design questions](#7-six-essential-system-design-questions)
- [What questions can the reader now answer independently?](#what-questions-can-the-reader-now-answer-independently)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. Why is the same Transformer three different workloads?

In a standard decoder-only Transformer block, every layer consists of Query, Key, and Value projections, Multi-Head or Grouped-Query Attention, an output projection, and a two-layer Feed-Forward / MLP network (Gate, Up, and Down projections).

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

*Figure 1.1: End-to-end data flow through a Transformer block. Token IDs pass through embeddings, enter $N$ successive layers of attention and MLP GEMMs, and output final vocabulary logits.*

Despite sharing this identical math, hardware performance changes radically depending on whether these layers are computed for thousands of tokens simultaneously or evaluated one token at a time:

| Phase | Input Shape ($B, S$) | Dominant Operation | Primary Bottleneck | Critical Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | Large batch, long context ($B \ge 64, S \ge 4,096$) | Forward + Backward GEMMs + Optimizer | Compute FLOP/s, HBM Capacity (Activations), Interconnect | Sustained MFU, Step Time |
| **Prefill** | Prompt sequence ($B=1..16, S \ge 2,048$) | Parallel Attention + Dense GEMM | Compute FLOP/s, HBM Bandwidth, TMA Transfer | Time-To-First-Token (TTFT) |
| **Decode** | Single token step ($B=1..64, S=1$) | Matrix-Vector Multiplications (GEMV) + KV Cache | HBM Memory Bandwidth (Memory-Bandwidth Bound) | Time-Per-Output-Token (TPOT / ITL) |

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

*Figure 1.2: Single-request execution breakdown (2,000 prompt tokens, 200 output tokens). Prefill processes all prompt tokens in parallel via dense GEMMs, while decode executes a serial loop 200 times.*

### 1.1. Training: Why compute scales to $6N$

Pre-training diverges fundamentally from inference due to two phases: **backward propagation** (computing gradients) and the **optimizer step** (updating weights).

In a standard Transformer block, each token and each model parameter requires approximately $2$ FLOPs during the forward pass (one multiply, one accumulate):

$$C_{\text{forward}} \approx 2 N \text{ FLOP/token}$$

Where $N$ is parameter count. During backpropagation, the chain rule dictates two distinct gradient computations:
1. Activation gradients passed to the preceding layer ($\nabla_X$): $\approx 2 N \text{ FLOP/token}$
2. Weight gradients used to update parameters ($\nabla_W$): $\approx 2 N \text{ FLOP/token}$

Thus, backpropagation demands exactly twice the compute of the forward pass ($4 N \text{ FLOP/token}$). The total theoretical compute required to train on a single token is:

$$C_{\text{train}} \approx 2N + 4N = \mathbf{6N \text{ FLOP/token}}$$

For a dataset of $D$ tokens, total compute scales to:

$$C_{\text{total}} \approx 6 N D \text{ FLOP}$$

Training LLaMA 3 70B ($N = 70 \times 10^9$) across 15 trillion tokens ($D = 15 \times 10^{12}$) requires:

```text
6 × (70 × 10^9) × (15 × 10^12) = 6.3 × 10^24 FLOP = 6.3 YottaFLOPs!
```

Because training operates on thousands of tokens per batch, execution is compute-bound, saturating Tensor Cores. However, training's primary penalty is memory: forward activations must be retained across all layers until backpropagation completes. To prevent out-of-memory crashes, systems employ **Activation Checkpointing (Gradient Checkpointing)**: intermediate activations are discarded during the forward pass and recomputed on demand during the backward pass. This saves gigabytes of VRAM at the cost of increasing total compute from $6N$ to $8N$ (a 33% compute overhead).

### 1.2. Prefill: Why prompts process in parallel

When a user submits a 2,000-token document, all 2,000 tokens are already available. The model does not need to wait for autoregressive dependencies; it ingests the entire prompt in a single parallel step.

Because the full sequence is processed simultaneously:
- The activation matrix is wide ($S = 2,000$).
- Multiplying $X \in \mathbb{R}^{S \times D}$ by weight matrix $W \in \mathbb{R}^{D \times F}$ is a dense **General Matrix Multiplication (GEMM)**.
- Arithmetic intensity sits far above the Roofline ridge point ($\alpha_{\text{crit}} \approx 295$ FLOP/byte), operating at $I \approx 400 \dots 600$ FLOP/byte.

On Hopper H100 hardware, this phase leverages **TMA (Tensor Memory Accelerator)** and **WGMMA (Warpgroup Matrix Multiply-Accumulate)** double-buffering:

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

*Figure 1.3: Asynchronous double-buffering on Hopper. While WGMMA computes block $k$ on the Tensor Cores, the hardware TMA engine concurrently fetches block $k+1$ from HBM into shared memory, masking memory latency.*

Prefill performance dictates **Time-To-First-Token (TTFT)**: the latency from when the user clicks submit until the first generated character appears on screen.

### 1.3. Decode: Why memory bandwidth bounds throughput

Once the first token is generated, execution flips into a serialized regime. Autoregressive language models generate tokens sequentially: predicting token $N+1$ strictly requires the embedding of token $N$. Generating a 200-token response requires invoking the model **200 consecutive times**.

This causes a severe collapse in hardware efficiency:
- Each iteration feeds only a single token ($B=1, S=1$) into the model.
- The activation matrix collapses into a single row vector ($X \in \mathbb{R}^{1 \times D}$).
- The operation drops from dense GEMM to a **General Matrix-Vector Multiplication (GEMV)**.
- As proven in Part 1, arithmetic intensity collapses to **$I \approx 1$ FLOP/byte**!

An H100 GPU capable of 990 TFLOP/s is limited by its 3.35 TB/s HBM bandwidth to just **3.35 TFLOP/s**—utilizing a microscopic **0.34%** of its peak arithmetic throughput.

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

*Figure 1.4: Short interactive prompt ("Hello!") versus an extensive contract analysis. Long-running decode phases hold growing KV cache allocations across hundreds of iterations, tying up precious HBM.*

Decode speed dictates **Time-Per-Output-Token (TPOT)** or **Inter-Token Latency (ITL)**: the interval between consecutive streamed words. To match natural human reading comprehension, production systems target 30–50 ms per token (20–30 tokens per second).

### 1.4. Tracing a token's physical lifecycle

Follow a request's end-to-end journey from arrival to completion:

1. **Ingress and Tokenization:** Raw user text enters the serving gateway (e.g., vLLM). The tokenizer maps characters to integer token IDs.
2. **Scheduling:** The request enters the serving queue. The scheduler inserts the prompt into the active continuous batch.
3. **Prefill (GEMM):** Prompt tokens copy to GPU memory. All layers execute in parallel; generated Key and Value vectors write to the **KV Cache** in HBM.
4. **First Token Sampling:** The final layer's logits vector is filtered (temperature, top-p, top-k) to sample the first token. TTFT concludes here.
5. **Autoregressive Decode Loop (GEMV):**
   - The sampled token ID feeds back into the model.
   - Model weights stream from HBM across all layers.
   - The new token performs attention against all historical KV vectors stored in HBM.
   - The new token's own Key and Value vectors append to the KV cache.
   - The next token is sampled. This loop repeats until an `<EOS>` token is emitted or max length is reached.
6. **Detokenization and Streaming:** Each sampled token immediately converts back to text and streams to the client via Server-Sent Events (SSE).

---

## 2. How to construct a complete memory budget

A common sizing mistake is assuming GPU memory needs to hold only model weights. In a production inference cluster, VRAM is partitioned across four major stakeholders:

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{activations}} + M_{\text{workspace}}$$

### 2.1. Weight memory

Weight footprint scales with parameter count and numeric precision:

$$M_{\text{weights}} = N \times \frac{b}{8} \text{ bytes}$$

Where $N$ is parameter count and $b$ is bit-width:

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

*Figure 2.1: Weight memory footprint across precisions (BF16, INT8, INT4) for 8B, 70B, and 405B models, relative to 80 GB (single H100) and 640 GB (8-GPU DGX) hardware boundaries.*

| Model Scale | BF16 / FP16 (16-bit, 2 B) | FP8 / INT8 (8-bit, 1 B) | INT4 / FP4 (4-bit, 0.5 B) | Minimum Hardware (BF16) |
| :--- | :--- | :--- | :--- | :--- |
| **8B (LLaMA 3)** | 16 GB | 8 GB | 4 GB | 1x GPU (24 GB / 80 GB) |
| **70B (LLaMA 3)** | 140 GB | 70 GB | 35 GB | 2x H100 (FP8) or 8x H100 (TP=8) |
| **405B (LLaMA 3)** | 810 GB | 405 GB | 202.5 GB | 2x DGX Nodes (16x H100, 1,280 GB) |

### 2.2. The KV cache formula

Key and Value tensors computed across all Transformer layers must persist in memory to avoid recalculation. The memory consumed per token across all layers is:

$$M_{\text{KV\_token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times b_{\text{bytes}}$$

Where:
- $2$: Accounts for both Key and Value matrices.
- $n_{\text{layers}}$: Total Transformer layers in the model.
- $n_{\text{kv\_heads}}$: Number of Key/Value attention heads per layer.
- $d_{\text{head}}$: Dimension per head ($d_{\text{model}} / n_{\text{heads}}$).
- $b_{\text{bytes}}$: Precision (2 bytes for FP16/BF16, 1 byte for FP8).

For a total context length $s$ (prompt + output tokens), request KV footprint is:

$$M_{\text{KV\_request}} = M_{\text{KV\_token}} \times s$$

Across $C$ concurrent requests:

$$M_{\text{KV\_total}} = C \times M_{\text{KV\_token}} \times s$$

### 2.3. The structural impact of GQA and MLA

Under traditional **Multi-Head Attention (MHA)** (used in LLaMA 1), each Query head paired with a dedicated Key and Value head ($n_{\text{kv\_heads}} = n_{\text{q\_heads}} = 64$).

Evaluating this on a 70B architecture ($n_{\text{layers}} = 80$, $n_{\text{kv\_heads}} = 64$, $d_{\text{head}} = 128$, $b = 2$ bytes):

$$M_{\text{KV\_token}} = 2 \times 80 \times 64 \times 128 \times 2 = 2,621,440 \text{ bytes} \approx \mathbf{2.62 \text{ MB/token}}$$

For an 8,192-token request:
$$2.62 \text{ MB} \times 8,192 \approx \mathbf{21.47 \text{ GB}}!$$

A single user session consumed over 21 GB of VRAM. Ten concurrent users saturated 214 GB, triggering CUDA OOM crashes.

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

*Figure 2.2: Memory comparison of MHA, GQA, and MLA architectures for an 8k-context 70B model.*

**Grouped-Query Attention (GQA)** resolves this: 8 Query heads share a single Key/Value head group ($n_{\text{kv\_heads}} = 8$, an 8:1 ratio):

$$M_{\text{KV\_token}} = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ bytes} \approx \mathbf{0.328 \text{ MB/token}}$$

For 8,192 tokens:
$$0.328 \text{ MB} \times 8,192 \approx \mathbf{2.68 \text{ GB}}$$

The KV cache shrinks **eight-fold**, allowing 8x more concurrent users on identical hardware.

**Multi-Head Latent Attention (MLA)**, introduced in **DeepSeek-V3**, compresses Key and Value heads into a low-rank latent subspace ($d_c = 512$), decompressing representations on the fly during computation to shrink KV memory even further.

### 2.4. Why training memory dwarfs inference

Inference requires memory only for weights and the KV cache. In contrast, training under the AdamW optimizer requires tracking massive state per parameter:

| State Component | Precision | Bytes per Parameter | Total for 70B Model |
| :--- | :--- | :--- | :--- |
| **Model Weights** | BF16 (16-bit) | 2 bytes | 140 GB |
| **Gradients** | BF16 (16-bit) | 2 bytes | 140 GB |
| **FP32 Master Weights** | FP32 (32-bit) | 4 bytes | 280 GB |
| **AdamW Momentum (1st moment)**| FP32 (32-bit) | 4 bytes | 280 GB |
| **AdamW Variance (2nd moment)** | FP32 (32-bit) | 4 bytes | 280 GB |
| **Total Static Training State**| — | **16 bytes/parameter** | **1,120 GB (1.12 TB)** |

Training a 70B model requires **1,120 GB** just for weights and optimizer states! Factoring in forward activations ($O(B \cdot S \cdot d_{\text{model}} \cdot n_{\text{layers}})$) pushes total memory past 2–3 Terabytes.

### 2.5. Complete memory ledger for 70B serving

Deploying LLaMA 3.1 70B (BF16, TP=8) across an 8-GPU H100 SXM5 server (640 GB total VRAM):

```text
Total Usable HBM3:                             640.0 GB
- Model Weights (BF16, 70B):                  -140.0 GB
- CUDA Driver & PyTorch Runtime:               -16.0 GB (~2 GB per GPU)
- Activation Scratchpad / GEMM Buffers:        -24.0 GB (~3 GB per GPU)
========================================================================
Remaining Dedicated KV Cache Pool:             460.0 GB
```

Evaluating capacity: under GQA, an average 4,096-token session consumes $1.34 \text{ GB}$:
$$\text{Max Concurrency} = \frac{460 \text{ GB}}{1.34 \text{ GB}} \approx \mathbf{343 \text{ concurrent sessions}}$$

The node serves 343 concurrent active users without a single memory eviction!

---

## 3. How serving engines keep the GPU saturated

Naive inference loops leave over 80% of GPU compute unutilized. Modern LLM serving engines (vLLM, TensorRT-LLM, TGI) deploy advanced scheduling and paging algorithms to keep hardware saturated.

### 3.1. Continuous batching

Static batching aggregates requests into fixed groups. But request lengths vary widely: one user needs 10 tokens, another needs 500:

**Two Failures of Static Batching:**
1. **Padding Waste:** Short requests must be padded with `<PAD>` tokens to match the longest sequence. The GPU burns billions of FLOPs computing padding.
2. **Head-of-Line Blocking:** A request that finishes in 10 tokens cannot leave the GPU; it sits idle in VRAM until the 500-token sequence finishes.

Introduced by **Orca (OSDI 2022)**, **Continuous Batching (Iteration-Level Scheduling)** eliminates this:
- Scheduling operates at the granularity of **individual token iterations**.
- As soon as a request completes, it is evicted immediately and its response returned to the client.
- A new request's prefill or another request's decode step dynamically injects into the vacant slot.
- Pipeline bubbles vanish, improving serving throughput by 2–4x.

### 3.2. PagedAttention

Traditional frameworks allocate memory in **contiguous** blocks. If a server caps maximum context at 2,048 tokens, it allocates a 2,048-token buffer up front.

This causes massive memory waste:
- **Internal Fragmentation:** If a user only generates 50 tokens, the reserved 1,998-token buffer sits empty and locked.
- **External Fragmentation:** Free memory fragments into scattered slices too small to accommodate new contiguous requests.

Historically, **60% to 80% of KV cache VRAM** was wasted due to fragmentation!

Developed for **vLLM (SOSP 2023)**, **PagedAttention** brings **Virtual Memory Paging** to GPU serving:
- KV caches are divided into small, fixed-size **physical blocks** (e.g., 16 or 32 tokens).
- As requests grow, the engine allocates physical blocks dynamically and records their locations in a page table.
- Fragmentation drops to **under 4%**, nearly doubling concurrent serving capacity on identical hardware.

### 3.3. Prefix caching

Production workloads feature heavily repeated prefixes: system instructions ("You are a helpful assistant..."), multi-turn dialogues, and few-shot examples.

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

*Figure 3.1: Prefix caching (Radix Attention). When two requests share a common prefix, the pre-computed KV blocks are referenced directly without recomputation.*

**Prefix Caching (Radix Attention)**:
- Stores computed KV blocks in a **Radix Tree** indexed by token hashes.
- When a new request arrives, the matching prefix is detected instantly.
- The pre-computed KV cache loads directly from HBM, **completely skipping the prefill phase** for those tokens!
- TTFT plummets, saving substantial GPU compute.

### 3.4. Disaggregated serving: Splitting prefill and decode

Prefill and decode phases are fundamentally incompatible on the same GPU:
- Prefill: Dense GEMMs, bursty compute spikes, saturates SMs.
- Decode: Memory-bandwidth bound, long-running, holds steady KV cache.

When a massive prefill request lands on a shared GPU, active decode steps stall, causing severe inter-token latency spikes (jitter).

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

*Figure 3.2: Disaggregated Serving (Splitwise). Prefill nodes focus on compute throughput to minimize TTFT, while Decode nodes optimize memory bandwidth for smooth TPOT; KV caches transfer over high-speed RDMA.*

**Disaggregated Serving (Splitwise / Tetris):**
- Partitions the cluster into dedicated **Prefill Nodes** and **Decode Nodes**.
- Prefill nodes ingest prompts, execute dense GEMMs, and write KV caches.
- KV caches transfer across high-speed fabrics (**GPUDirect RDMA over 400G InfiniBand**) to decode nodes.
- Decode nodes stream tokens smoothly without experiencing prefill interruptions.

> [!WARNING]
> **Trade-Off Warning:** Disaggregated serving requires immense network bandwidth. If prompt context is short, transferring gigabytes of KV cache across the fabric may take longer than computing prefill locally. Disaggregation is only advantageous when high-speed fabrics (400 Gbps+) pair with long prompt contexts.

### 3.5. Speculative decoding

Speculative decoding bypasses the memory-bandwidth wall of autoregressive decoding:

1. **Draft Model:** A compact model (e.g., an 8B model paired with a 70B target model) quickly predicts $K = 4$ candidate tokens ($t_1, t_2, t_3, t_4$).
2. **Target Model Verification:** The 70B model evaluates all 4 candidates in a **single parallel GEMM step**.
3. If the target model accepts the draft predictions, **4 tokens are produced in a single step**!
4. Speedup scales with acceptance rate ($\alpha$):

$$\text{Speedup} \approx \frac{1}{1 - \alpha + \frac{\alpha}{K}}$$

At an 80% acceptance rate, the 70B model generates text 2–3x faster, with zero degradation in mathematical output quality.

---

## 4. Planning capacity and latency together

Balancing throughput against latency requires rigorous queuing analysis.

### 4.1. Distinguishing the four core metrics

Always separate these metrics in systems design:
- **TTFT (Time-To-First-Token):** Latency to first emitted token (perceived responsiveness).
- **TPOT (Time-Per-Output-Token):** Inter-token streaming interval (human reading pace).
- **Throughput:** Aggregate tokens generated per second ($\text{Tokens/s}$).
- **Goodput:** Tokens successfully delivered within strict SLA thresholds (e.g., $\text{TTFT} < 500 \text{ ms}$ and $\text{TPOT} < 40 \text{ ms}$).

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

*Figure 4.1: Representative TTFT waterfall chart illustrating the cumulative impact of network transport, authentication, queue wait, tokenization, and prefill execution.*

### 4.2. Why queues explode: Little's Law

Queue dynamics are governed by **Little's Law**:

$$L = \lambda W$$

Where $L$ is average queue depth, $\lambda$ is arrival rate, and $W$ is residence time.

In queuing theory (M/M/1 and M/G/1 models), as utilization ($\rho = \lambda / \mu$) approaches 1, queue wait time explodes asymptotically:

$$W_q = \frac{\rho}{\mu (1 - \rho)}$$

```text
Utilization (ρ) = 50%  --> Wait Multiplier: 1x
Utilization (ρ) = 80%  --> Wait Multiplier: 4x
Utilization (ρ) = 90%  --> Wait Multiplier: 9x
Utilization (ρ) = 95%  --> Wait Multiplier: 19x!
Utilization (ρ) = 99%  --> Wait Multiplier: 99x (Complete Queue Saturation!)
```

Pushing cluster utilization from 80% to 95% gains 15% throughput but **quadruples queue latency**. Production interactive services must target 65–75% utilization to preserve p99 SLAs.

---

## 5. What changes with vision and video workloads?

Multimodal **Vision-Language Models (VLMs)** invalidate text-only capacity models.

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

*Figure 5.1: Multimodal (VLM) request pipeline. Media inputs are decoded, partitioned into image patches, and encoded by a Vision Transformer (ViT) into thousands of token representations.*

### 5.1. Why an image is not "a single token"

When a user submits a single $1,024 \times 1,024$ image:
1. The image partitions into $14 \times 14$ pixel patches: $(1,024 / 14) \times (1,024 / 14) \approx 5,329 \text{ patches}$.
2. The **Vision Transformer (ViT)** encodes these into **thousands of visual tokens**.
3. The language model processes the image as a 3,000-token prompt, inflating prefill duration and KV cache volume.

### 5.2. Video workloads

A 10-second video at 2 fps introduces 20 frames, generating **10,000 to 25,000 tokens** per request!
Mitigations include dynamic resolution scaling, token pruning of redundant backgrounds, and running vision encoders on separate pre-processing workers.

---

## 6. A concrete 70B serving case study

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

*Figure 6.1: Engineering decision tree for production serving optimization across precision, batching, and architectural topology.*

Deploying LLaMA 3.1 70B on 8x NVIDIA H100 SXM5 GPUs (640 GB total HBM3):

1. **Topology:** 8-way Tensor Parallelism ($Y=8$) across NVLink (140 GB BF16 weights).
2. **Dedicated KV Pool:** 480 GB net available KV cache.
3. **Concurrency:** At 4k context ($1.34 \text{ GB}$ per session), the node sustains **358 concurrent sessions**.
4. **Theoretical TPS Ceiling:** 8 GPUs provide $8 \times 3,350 = 26,800 \text{ GB/s}$ aggregate bandwidth. Under $B=1$:
   $$\text{Max TPS} = \frac{26,800 \text{ GB/s}}{140 \text{ GB}} \approx \mathbf{191 \text{ tokens/s}}$$
   (With continuous batching at $B \ge 16$, amortized weight streaming elevates throughput to 1,500–2,500 tokens/s).

---

## 7. Six essential system design questions

### 7.1. Why is prefill compute-bound while decode is memory-bound?

Prefill processes thousands of prompt tokens concurrently. The activation matrix is wide, executing dense General Matrix Multiplications (GEMM) with arithmetic intensities of 400–600 FLOP/byte, saturating Tensor Cores.

Decode operates autoregressively on a single token ($B=1, S=1$). The operation collapses to a General Matrix-Vector multiplication (GEMV). The entire 140 GB weight matrix must stream from HBM to compute just 2 FLOPs per parameter ($I \approx 1 \text{ FLOP/byte}$), bottlenecking strictly on HBM memory bandwidth.

### 7.2. Why does KV cache scale linearly with concurrency?

Each session maintains an independent conversation history. The model must retain Key and Value vectors for all prior tokens to compute attention. Consequently, KV memory scales as $O(C \cdot S)$, growing strictly linearly with concurrent users ($C$) and context length ($S$).

### 7.3. Does PagedAttention reduce theoretical KV memory footprint?

No. It does not alter the mathematical size of a token's Key and Value vectors. Instead, it **eliminates memory fragmentation and reservation waste**. By allocating memory in small, fixed-size physical blocks (virtual memory paging), it cuts fragmentation from 60–80% to under 4%, nearly doubling the concurrent capacity of physical VRAM.

### 7.4. Why does 2x quantization not always yield a 2x throughput speedup?

Halving precision (e.g., FP16 to FP8) cuts memory traffic by half, speeding up memory-bound decode phases. However:
1. Dequantization overhead can consume ALU cycles if weights must convert back to higher precision for accumulation.
2. In compute-bound regimes (large batches or prefill), execution is limited by raw FLOP capacity rather than memory streaming bandwidth.

### 7.5. When does disaggregated serving degrade performance?

When network interconnect bandwidth is insufficient or prompts are short. For brief prompts, local prefill takes only a few milliseconds. Transferring the resulting KV cache across the fabric may introduce more network transport latency than the local compute saved.

### 7.6. Should you optimize for throughput or latency?

It depends on the product use case. For real-time conversational agents, priority focuses on **Latency (low TTFT and smooth TPOT)** via smaller batch sizes. For asynchronous batch workloads (bulk document analysis or synthetic data generation), priority focuses on **Throughput (maximum tokens/s per dollar)** via large batch sizes.

---

### What questions can the reader now answer independently?

1. How do you calculate the exact KV cache memory requirement for an 8k-context 70B model under MHA versus GQA?
2. Why is an H100 with 990 TFLOP/s peak capacity restricted to 3.35 TFLOP/s during single-token generation?
3. How does continuous batching eliminate the padding waste inherent in static batching?
4. How does PagedAttention apply OS virtual memory concepts to eliminate GPU memory fragmentation?
5. Why does pushing cluster utilization from 80% to 95% trigger an exponential surge in queue latency?

---

## The whole story in six lines

- A Transformer layer runs compute-bound during prefill via dense GEMMs, but collapses to memory-bandwidth bound during decode due to single-token GEMVs.
- Preserving training state ($16N$ bytes for AdamW + activations) requires 4x to 8x more memory than pure inference serving.
- GQA groups Key and Value heads, slashing KV cache memory consumption eight-fold compared to traditional MHA.
- PagedAttention eliminates internal and external VRAM fragmentation, reducing wasted allocation from 60–80% to under 4%.
- Disaggregated serving decouples compute-bound prefill from memory-bound decode pools, preventing prompt ingestion spikes from freezing streamed output.
- Queuing theory dictates that latency explodes exponentially beyond 80% utilization; reliable goodput demands provisioning clusters at 65–75% load.

---

## Glossary

- **Prefill:** The initial compute-bound phase where all prompt tokens process in parallel to populate the KV cache.
- **Decode:** The autoregressive memory-bandwidth-bound phase generating tokens one by one.
- **KV Cache:** Storage in HBM holding past Key and Value vectors to avoid recomputing attention history.
- **Continuous Batching:** Cellular iteration-level scheduling that evicts finished sequences and admits new ones per token step.
- **PagedAttention:** A memory management algorithm using virtual memory paging to store KV caches in non-contiguous physical blocks.
- **Grouped-Query Attention (GQA):** An attention pattern where multiple Query heads share single Key/Value groups to compress KV memory.
- **Disaggregated Serving:** An architecture physically separating prefill worker pools from decode worker pools.
- **Speculative Decoding:** Accelerating generation by having a small draft model propose tokens verified in parallel by the target model.
- **Goodput:** The rate of useful tokens delivered strictly within predefined latency SLAs.

---

## Going deeper

- [K1] Woosuk Kwon et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023. vLLM architecture and PagedAttention design.
- [K2] Gyeong-In Yu et al. (2022). [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022. Continuous iteration-level batching.
- [K3] Joshua Ainslie et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023. Grouped-Query Attention and KV cache savings.
- [K4] Pratyush Patel et al. (2024). [Splitwise: Efficient Generative LLM Serving Using Phase Splitting](https://arxiv.org/abs/2311.18677). ISCA 2024. Disaggregated serving architecture.
- [K5] Charlie Chen et al. (2023). [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318). Speculative decoding analysis.
- [K6] DeepSeek-AI (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). Multi-Head Latent Attention (MLA) and low-rank KV compression.
- [K7] Meta AI (2024). [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). Hyperparameters for 8B, 70B, and 405B architectures.
- [K8] On this blog: [AI Infrastructure (Part 1): How Does a GPU Work? — From One SM to 16,384 GPUs](post.html?slug=understanding-gpus-and-llm-scaling).
- [K9] On this blog: [AI Infrastructure (Part 3): Multi-GPU Fabrics and Distributed Training — NVLink, InfiniBand, and 4D Parallelism](post.html?slug=multi-gpu-networks-and-distributed-training).
