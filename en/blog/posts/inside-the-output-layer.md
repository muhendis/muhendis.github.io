A prompt's journey through the Transformer backbone has unfolded across our previous seven chapters:
1. [Tokenization](post.html?slug=how-tokenization-works): Text is segmented into discrete integer IDs (`[1054, 492, 281]`).
2. [Embedding Layer](post.html?slug=inside-the-embedding-layer): IDs map to dense vectors and receive RoPE rotational frequencies.
3. [Semantic Vector Space](post.html?slug=how-embeddings-work): Meaning is encoded as geometric angles and distances.
4. [Self-Attention](post.html?slug=inside-self-attention): Tokens project $Q, K, V$ vectors to exchange contextual information.
5. [Causal Attention](post.html?slug=inside-causal-attention): Lower-triangular masking enforces strict autoregressive time.
6. [Multi-Head Attention](post.html?slug=inside-multi-head-attention): Independent subspaces capture distinct grammatical axes.
7. [Transformer Block](post.html?slug=inside-the-transformer-block): Residual highways, RMSNorm, and SwiGLU stabilize representation learning.

In our running 3-token prompt (**`["the", "dog", "chased"]`**), the terminal block (layer 32 in Llama 3.1 8B) outputs position 3 (`"chased"`) as a dense hidden state vector $h_3 \in \mathbb{R}^{d_{\text{model}}}$ ($d_{\text{model}} = 4096$).

Here lies the fundamental engineering reality: **A continuous 4,096-dimensional floating-point vector cannot be printed onto a screen.** Human language is discrete and selected from a vocabulary of hundreds of thousands of candidate tokens. To output the next token (e.g. `"rabbit"`), the model must project this vector across the vocabulary, compute log-odds (logits), normalize them via Softmax, and sample the winner.

This final frontier is governed by the **Language Model Head (`lm_head`)**. Far from a trivial projection, it represents modern AI's two most severe hardware bottlenecks:
- **The Memory Wall in Training:** In Llama 3.1 8B (128k vocabulary, 8k context, batch=4), raw logits `[4, 8192, 128256]` consume **$8.4\text{ GB}$ of VRAM** in forward pass and another **$8.4\text{ GB}$** in backward pass, triggering instant OOM without fused kernels.
- **The Memory Bandwidth Wall in Inference:** Generating each token streams all 16 GB of weights from HBM to cores. With arithmetic intensity $< 1\text{ FLOP/byte}$, GPU Tensor Cores sit idle $>95\%$ of the time waiting for memory.

In this final article, we dissect the tensor anatomy of the LLM Head, tied vs. untied embeddings, logit soft-capping, the $\frac{\partial \mathcal{L}}{\partial z} = P - y$ gradient highway, a step-by-step 4D numerical walkthrough, Megatron-LM and Liger Kernel memory solutions, Medusa/EAGLE speculative decoding, and the complete `model.generate()` loop.

**In this article**

- [1. The big picture and data path: From the final transformer block to the screen](#1-the-big-picture-and-data-path-from-the-final-transformer-block-to-the-screen)
- [2. LLM Head architecture: Tensor dimensions, token_num, and broadcasting](#2-llm-head-architecture-tensor-dimensions-tokennum-and-broadcasting)
- [3. The weight tying dilemma: Parameter efficiency vs. orthogonal representations](#3-the-weight-tying-dilemma-parameter-efficiency-vs-orthogonal-representations)
- [4. Taming the logits: Gemma 2 and the logit soft-capping innovation](#4-taming-the-logits-gemma-2-and-the-logit-soft-capping-innovation)
- [5. Transition to probability space: Logits, Softmax, and shift invariance](#5-transition-to-probability-space-logits-softmax-and-shift-invariance)
- [6. Cross-Entropy Loss: Information theory and asymptotic penalty dynamics](#6-cross-entropy-loss-information-theory-and-asymptotic-penalty-dynamics)
- [7. The elegance of backpropagation: Derivative of Softmax + Cross-Entropy (dL/dz = P - y)](#7-the-elegance-of-backpropagation-derivative-of-softmax-cross-entropy-dldz-p---y)
- [8. Step-by-step numerical walkthrough: Forward pass, soft-capping, loss, and backward pass](#8-step-by-step-numerical-walkthrough-forward-pass-soft-capping-loss-and-backward-pass)
- [9. The hardware memory wall: Megatron-LM vocab parallelism and Liger Kernel](#9-the-hardware-memory-wall-megatron-lm-vocab-parallelism-and-liger-kernel)
- [10. Fine-tuning and vocabulary expansion: LoRA and the modules_to_save trap](#10-fine-tuning-and-vocabulary-expansion-lora-and-the-modulestosave-trap)
- [11. Inference acceleration: Speculative decoding with Medusa and EAGLE](#11-inference-acceleration-speculative-decoding-with-medusa-and-eagle)
- [12. The autoregressive generation loop: Inside model.generate() and sampling strategies](#12-the-autoregressive-generation-loop-inside-modelgenerate-and-sampling-strategies)
- [13. Two engineering lenses: Training vs. inference dynamics](#13-two-engineering-lenses-training-vs-inference-dynamics)
- [14. Step-by-step verification with Python and PyTorch](#14-step-by-step-verification-with-python-and-pytorch)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [To dive deeper](#to-dive-deeper)

---

## 1. The big picture and data path: From the final transformer block to the screen

### Intuitive mental model: The corkboard, the detective, and the massive dictionary

Before diving into the tensor linear algebra, let us establish an intuitive mental model to anchor the relationship between sequence length (`context_len`) and the output layer (`lm_head`). Picture the neural network as a **Detective**, and the context window as the detective's **Corkboard**:

```text
[STEP 1: THE CORKBOARD] (Context Length / context_len)
All prompt tokens are pinned to the investigative board:
"The", "murderer", "in", "the", "garden", "holding", "a" ...
(This is dynamic GPU VRAM allocation = The KV Cache)
        │
        ▼
[STEP 2: THE DETECTIVE'S SUMMARY] (Final Hidden Vector: h)
The detective cross-examines all pinned notes (Attention) and synthesizes a single thought:
A dense vector of 4,096 continuous numbers: [0.5, -1.2, 0.8, 1.5 ...]
        │
        ▼
[STEP 3: THE VOCABULARY DESK] (Output Layer / LLM Head)
A colossal 128,000-word dictionary is opened (the W_head weight matrix).
The detective takes the 4,096-dimensional thought and dot-products it across every word (Logits):
• "knife"  : +45 score   |   • "pistol" : +42 score
• "apple"  : +2 score    |   • "truck"  : -18 score
        │
        ▼
[STEP 4: PROBABILITY NORMALIZATION] (Softmax)
Scores are normalized into a valid 100% probability pie chart:
• "knife"  : 65%         |   • "pistol" : 30%
• "apple"  : 4.9%        |   • "truck"  : 0.0001%
        │
        ▼
[STEP 5: SAMPLING & GENERATION] (Sampling & Next Token)
A dice roll samples "knife" with 65% probability, printing it to the user's screen.
Crucially: "knife" is immediately pinned back to THE CORKBOARD (KV Cache) for the next step.
```

This 5-step flow forms the intuitive backbone of how a dense $d$-dimensional hidden vector translates into discrete human language. This backbone diverges into two fundamentally distinct engineering pathways across training and inference:

### The dual destiny of the hidden state: Training optimization vs. inference decoding

When the final hidden representation $H$ emerges from the last Transformer block, it does not immediately enter the vocabulary projection matrix. Modern decoder-only architectures first apply a final normalization layer—typically **RMSNorm** (Root Mean Square Normalization)—to standardize the feature variance across the hidden dimension:

$$H_{\text{norm}} = \text{RMSNorm}(H) = \frac{H}{\sqrt{\frac{1}{d} \sum_{i=1}^d H_i^2 + \epsilon}} \odot \gamma$$

Where $\gamma \in \mathbb{R}^d$ is a learned affine scaling parameter.

Once normalized, the data path bifurcates into two fundamentally distinct engineering workflows:

1. **The Training Path (Parallel Loss Optimization):**
   - The entire batch of sequences $H_{\text{norm}} \in \mathbb{R}^{B \times S \times d}$ is projected simultaneously through $W_{\text{head}}^T$ using dense General Matrix Multiplication (**GEMM**).
   - If enabled (e.g. Gemma 2), logits are constrained via hyperbolic tangent soft-capping.
   - Numerically stabilized Cross-Entropy Loss evaluates every position's prediction against the ground-truth shifted label sequence ($t+1$).
   - The error signal $\frac{\partial \mathcal{L}}{\partial z} = P - y$ is backpropagated to compute parameter gradients and update the optimizer states (AdamW).
2. **The Inference Path (Autoregressive Token Generation):**
   - In the **prefill phase**, the prompt tokens are processed in parallel to populate the Key-Value (KV) cache, and only the logit vector of the very last prompt position is computed.
   - In the **decode phase**, a single token vector $h_{-1} \in \mathbb{R}^{1 \times d}$ is projected through $W_{\text{head}}^T$ using memory-bandwidth-bound Matrix-Vector Multiplication (**GEMV**).
   - Logits pass through a pipeline of `LogitsProcessor` transforms (repetition penalty, temperature scaling, bad-word suppression).
   - A sampling strategy (Greedy, Top-$K$, Nucleus Top-$p$) samples a single integer token ID $t_{\text{next}}$.
   - The token ID is appended to the KV cache, decoded back into a text fragment via the tokenizer, and streamed to the user interface.

<svg viewBox="0 0 560 330" role="img" aria-label="End-to-end data path of the output layer. The hidden state from the final transformer block passes through the final RMSNorm, projects through the lm_head matrix, and bifurcates into training loss computation and inference token sampling." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="out-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="out-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="out-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<rect x="16" y="16" width="528" height="58" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Transformer Stack Exit</text>
<text x="32" y="58" text-anchor="start" style="fill:var(--c-text);font-size:13.5px;font-weight:600">Final Block Hidden State H [B, S, d] → Final RMSNorm [B, S, d]</text>
<line x1="280" y1="74" x2="280" y2="94" marker-end="url(#out-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="110" y="96" width="340" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.4"/>
<text x="280" y="116" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Language Model Head (lm_head)</text>
<text x="280" y="132" text-anchor="middle" style="fill:var(--c-accent);font-size:11.5px;font-family:var(--font-mono)">Linear(d_model, vocab_size, bias=False) → Z = H · W^T</text>
<line x1="280" y1="144" x2="280" y2="164" marker-end="url(#out-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="166" width="260" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="280" y="187" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">Optional: Logit Soft-Capping (c · tanh(z / c))</text>
<path d="M220 200 C 220 224, 140 224, 140 240" marker-end="url(#out-blue)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M340 200 C 340 224, 420 224, 420 240" marker-end="url(#out-green)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="20" y="244" width="240" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="140" y="264" text-anchor="middle" style="fill:var(--c-accent);font-size:12px;font-weight:700">TRAINING PATH</text>
<text x="140" y="282" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Cross-Entropy Loss: L = -log(P_target)</text>
<text x="140" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">dL/dz = P - y (Gradient Highway)</text>
<rect x="300" y="244" width="240" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-success);stroke-width:1.2"/>
<text x="420" y="264" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:700">INFERENCE PATH</text>
<text x="420" y="282" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">LogitsProcessor (Temp, Penalty, Top-p)</text>
<text x="420" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Multinomial Sampling → Next Token ID</text>
</svg>

---

## 2. LLM Head architecture: Tensor dimensions, token_num, and broadcasting

### Resolving the linear layer misconception: Where does context live?

In popular deep learning frameworks such as PyTorch and Hugging Face `transformers`, the LM Head is invariably instantiated with a remarkably compact declaration:

```python
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

Engineers examining this initialization often confront a conceptual stumbling block: *If the layer is declared as `nn.Linear(hidden_size, vocab_size)`, where does the sequence length (`token_num` or `context_length`) reside? Is `token_num` a fixed parameter of the head?*

The mathematical answer lies in the tensor contraction rules of multi-dimensional matrix multiplication. The linear layer's internal weight parameter matrix $W_{\text{head}}$ has a fixed two-dimensional shape:

$$W_{\text{head}} \in \mathbb{R}^{V \times d}$$

Where $V = \text{vocab\_size}$ and $d = \text{hidden\_size}$. The layer possesses zero knowledge of batch size or sequence length in its static memory footprint.

During the forward pass, the activation tensor fed into this layer has three dimensions:

$$H \in \mathbb{R}^{B \times S \times d}$$

Where $B$ is the batch size, $S$ is the sequence length (the number of processed tokens in the context), and $d$ is the hidden dimension. When PyTorch evaluates `self.lm_head(H)`, it invokes an optimized batched GEMM kernel (such as `cublasGemmStridedBatched` or cutlass kernels) that contracts exclusively over the final dimension ($d$):

$$Z_{b, s, v} = \sum_{k=1}^d H_{b, s, k} \cdot (W_{\text{head}})_{v, k} = (H \cdot W_{\text{head}}^T)_{b, s, v}$$

Read this as:
- $Z \in \mathbb{R}^{B \times S \times V}$ is the resulting 3D Logits tensor.
- For every individual batch $b \in \{1, \dots, B\}$ and for every distinct sequence position $s \in \{1, \dots, S\}$, the model computes a complete, independent $V$-dimensional score vector.
- The context length `token_num` is strictly dynamic; PyTorch broadcasts the static weight matrix across all batches and time steps.

### Training vs. inference tensor slicing

This broadcasting mechanic exposes a profound operational contrast between training and inference:

```text
Training Forward Pass:
  H:       [Batch, Seq_Len, Hidden]       e.g., [4, 8192, 4096]
  W^T:     [Hidden, Vocab]                e.g., [4096, 128256]
  Logits:  [Batch, Seq_Len, Vocab]        e.g., [4, 8192, 128256]  <-- 8.4 GB tensor!

Inference Autoregressive Step (Decode):
  h_last:  [Batch, 1, Hidden]             e.g., [1, 1, 4096]
  W^T:     [Hidden, Vocab]                e.g., [4096, 128256]
  Logits:  [Batch, 1, Vocab]              e.g., [1, 1, 128256]    <-- Only 256 KB!
```

During training with **Teacher Forcing**, the model evaluates all tokens in parallel. Position $s$ predicts token $s+1$. To align the predictions with the targets, Hugging Face implements an explicit slice:

```python
# Shift so that tokens < n predict n
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

# Flatten the tokens for cross-entropy
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(
    shift_logits.view(-1, config.vocab_size),
    shift_labels.view(-1)
)
```

In contrast, during autoregressive decoding, computing logits for past tokens would be completely redundant because their identities are already fixed. The inference runtime extracts solely the final position: `logits = self.lm_head(hidden_states[:, -1:, :])`.


---

## 3. The weight tying dilemma: Parameter efficiency vs. orthogonal representations

### Historical roots: Press & Wolf (2017) and parameter budgets

In early neural language modeling, input word embeddings ($W_{\text{in}} \in \mathbb{R}^{V \times d}$) and output projection weights ($W_{\text{out}} \in \mathbb{R}^{V \times d}$) were treated as two distinct parameter matrices. In 2017, Ofir Press and Lior Wolf published *Using the Output Embedding to Improve Language Models*, proposing that the input embedding matrix and the output `lm_head` should share the exact same physical memory buffer:

$$W_{\text{head}} = W_{\text{embed}}$$

In the era of small models (such as GPT-2 Small with 117M parameters, $d=768$, and $V=50257$), the input embedding matrix accounted for $50,257 \times 768 \approx 38.6\text{M}$ parameters—a staggering **$33\%$ of the entire model's parameter budget**. Duplicating this matrix for the output head would have pushed the vocabulary tables to two-thirds of the network. Weight tying was an indispensable engineering necessity.

```
Embedding parameters scale as O(V · d)   --> Linear in vocabulary size
Transformer blocks scale as   O(L · d^2) --> Quadratic in depth and hidden size
```

As foundation models scaled to 8B, 70B, and 405B parameters with 32 to 128 transformer layers, the parameter fraction occupied by vocabulary tables collapsed. In Llama 3.1 8B ($L=32, d=4096, V=128256$), the embedding layer accounts for $128,256 \times 4,096 \approx 525\text{M}$ parameters, representing just **$6.5\%$ of the total network**. This shift ignited a major architectural debate: *Should modern LLMs tie their weights or decouple them?*

| Model Architecture | Parameter Scale | Vocabulary ($V$) | Hidden Dimension ($d$) | Weight Tying (`tie_word_embeddings`) | Output Parameter Overhead |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta Llama 3.1** | 8 Billion | 128,256 | 4,096 | **False (Untied)** | +525 Million parameters |
| **Alibaba Qwen 2.5** | 7 Billion | 152,064 | 3,584 | **False (Untied)** | +545 Million parameters |
| **Google Gemma 2** | 9 Billion | 256,000 | 3,584 | **True (Tied)** | 0 (Reuses input embedding) |
| **DeepSeek-V3** | 671B (MoE) | 129,280 | 7,168 | **False (Untied)** | +926 Million parameters |

### The geometric discovery: Orthogonal semantic spaces

Why did Meta (Llama) and Alibaba (Qwen) sacrifice half a billion parameters to uncouple their output layers? The answer lies in representation geometry:
1. **Input Embedding ($W_{\text{in}}$):** Clusters synonyms and builds a continuous semantic manifold for attention querying.
2. **Output Head ($W_{\text{out}}$):** Acts as a high-dimensional linear classifier, hyper-separating closely related candidates (`"sat"` vs. `"lay"`).

In untied models, input and output spaces learn nearly orthogonal coordinate systems ($\cos \theta \approx 0.0007$). SVD analysis reveals that weight-tied models compress $90\%$ of spectral energy into a single dominant component, whereas untied models utilize all 4,096 dimensions at full rank.

Google **Gemma 2** selected `"tie_word_embeddings": true` due to its colossal **256,000-token vocabulary**: an untied head at $d=3584$ would add $917.5\text{M}$ parameters (nearly 1 GB of weights), making consumer GPU deployment impossible.

<svg viewBox="0 0 560 210" role="img" aria-label="Comparison between tied and untied embeddings showing geometric alignment and spectral properties." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="38" text-anchor="middle" style="fill:var(--c-accent);font-size:13px;font-weight:700">TIED WEIGHTS (Gemma 2)</text>
<text x="144" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">W_out = W_in (Shared Buffer)</text>
<rect x="36" y="70" width="216" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="144" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Cosine Similarity: cos(W_in, W_out) = 1.0</text>
<text x="36" y="126" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• 0 parameter overhead for output head</text>
<text x="36" y="146" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• SVD: 90% spectral energy in 1st component</text>
<text x="36" y="166" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px">• Risk of logit drift and overconfidence</text>
<rect x="288" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416" y="38" text-anchor="middle" style="fill:var(--c-success);font-size:13px;font-weight:700">UNTIED WEIGHTS (Llama 3.1)</text>
<text x="416" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">W_out ≠ W_in (Independent Matrices)</text>
<rect x="308" y="70" width="216" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="416" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Cosine Similarity: cos(W_in, W_out) ≈ 0.0007</text>
<text x="308" y="126" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• Full-rank orthogonal semantic spaces</text>
<text x="308" y="146" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• Distinct encoding vs classification roles</text>
<text x="308" y="166" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">• Cost: +525M parameters in VRAM</text>
</svg>

---

## 4. Taming the logits: Gemma 2 and the logit soft-capping innovation

### The pathology of unbounded logits

In large-vocabulary, weight-tied models, the dot product $z_i = h \cdot W_i^T$ has no theoretical ceiling. If an outlier logit surges to $+100$, $e^{100}$ crushes all competing candidates, assigning $P_i \approx 0.999999$ to a single token. Distribution entropy collapses, triggering severe overconfidence and intractable repetition loops during autoregressive generation.

### Hyperbolic tangent soft-capping

To neutralize logit explosion without clipping gradients with discontinuous step functions (which destroy backpropagation), Google DeepMind introduced **Logit Soft-Capping** in Gemma 2:

$$z_{\text{capped}} = c \cdot \tanh\left(\frac{z_{\text{raw}}}{c}\right)$$

Read this as:
- $z_{\text{raw}} = h \cdot W^T$ is the unconstrained logit produced by the linear layer.
- $c$ is a predefined capacity threshold (scaling hyperparameter).
- $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ is the hyperbolic tangent function, bounded strictly between $-1$ and $+1$.
- $z_{\text{capped}}$ is guaranteed to reside strictly within the open interval $(-c, +c)$.

In the official Gemma 2 configuration:
- **Attention Logit Soft-Capping:** $c = 50.0$ (`attn_logit_softcapping = 50.0`)
- **Final Output Head Soft-Capping:** $c = 30.0$ (`final_logit_softcapping = 30.0`)

```text
Behavioral Regime Analysis:
1. Linear Regime (|z| << c):
   For small logits (e.g. z = 1.0, c = 30.0):
   tanh(1/30) ≈ 1/30 ==> z_capped ≈ 30 * (1/30) = 1.0.
   The function acts as an identity mapping, perfectly preserving relative log-odds.

2. Saturation Regime (|z| >> c):
   As logits blow up toward ±inf:
   tanh(z/c) smoothly approaches ±1.0 ==> z_capped approaches ±30.0.
   No token can ever monopolize the distribution to force numerical overflow.
```

The derivative of soft-capping with respect to the raw logit is elegant and well-behaved:

$$\frac{\partial z_{\text{capped}}}{\partial z_{\text{raw}}} = c \cdot \frac{d}{dz_{\text{raw}}} \left[ \tanh\left(\frac{z_{\text{raw}}}{c}\right) \right] = c \cdot \frac{1}{c} \left(1 - \tanh^2\left(\frac{z_{\text{raw}}}{c}\right)\right) = 1 - \left(\frac{z_{\text{capped}}}{c}\right)^2$$

When $z_{\text{raw}}$ is small, the derivative is $1.0$ (gradients pass through unattenuated). When $z_{\text{raw}}$ blows up, the gradient smoothly decays to $0$, acting as an intrinsic, self-regulating gradient clipper that prevents training loss spikes.

---

## 5. Transition to probability space: Logits, Softmax, and shift invariance

### Logits as unnormalized log-odds

The raw outputs $z = [z_1, z_2, \dots, z_V]$ of the output head are termed **Logits**—a nomenclature originating from *log-odds* ($\ln \frac{p}{1-p}$). Logits are real numbers in the interval $(-\infty, +\infty)$. They represent relative, unnormalized evidence scores for each token in the vocabulary.

To transform these continuous scores into a valid discrete probability distribution where:

$$0 \le P(i) \le 1 \quad \text{and} \quad \sum_{i=1}^V P(i) = 1$$

The architecture routes the vector through the **Softmax** function:

$$P(i) = \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}$$

Softmax satisfies two essential properties:
1. **Strict Monotonicity:** If $z_a > z_b$, then $P(a) > P(b)$. The relative ranking of tokens is strictly preserved.
2. **Exponential Amplification:** Differences between logits are magnified exponentially, allowing the network to distinguish slight semantic preferences.

### The IEEE 754 floating-point overflow catastrophe

In modern AI hardware accelerators (NVIDIA H100, B200), training is executed in low-precision formats such as **bfloat16** or **float16**.

In standard IEEE 754 16-bit half-precision floating point (`float16`):
- Maximum representable value: $65,504$
- Minimum positive subnormal value: $5.96 \times 10^{-8}$

If an unconstrained logit reaches $z_i = 100$:

$$e^{100} \approx 2.688 \times 10^{43} \gg 65,504 \implies \textbf{OVERFLOW to } +\infty$$

The denominator becomes $+\infty$, the division produces $\frac{\infty}{\infty}$, and the tensor instantly fills with **NaN (Not a Number)**. The entire model's weights are corrupted in a single backward step.

### Shift invariance: The mathematical cure

Deep learning runtimes prevent this failure by leveraging the property of **Shift Invariance**. Subtracting an arbitrary constant $C$ from every element in the logit vector leaves the resulting Softmax distribution completely unchanged:

$$\frac{e^{z_i - C}}{\sum_{j=1}^V e^{z_j - C}} = \frac{e^{z_i} \cdot e^{-C}}{\sum_{j=1}^V (e^{z_j} \cdot e^{-C})} = \frac{e^{-C} \cdot e^{z_i}}{e^{-C} \cdot \sum_{j=1}^V e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}$$

By setting the shift constant to the exact maximum value in the vector:

$$C = \max(z) = \max_{k} z_k$$

$$z'_i = z_i - \max(z)$$

The largest shifted logit becomes:

$$z'_{\text{max}} = \max(z) - \max(z) = 0 \implies e^0 = 1.0$$

Every other shifted logit is non-positive ($z'_i \le 0$), ensuring that all exponential terms fall strictly within the safe numerical interval:

$$e^{z'_i} \in (0, 1.0]$$

Overflow is rendered mathematically impossible. The denominator $\sum e^{z'_j}$ is guaranteed to be at least $1.0$, preventing division-by-zero underflows as well.

---

## 6. Cross-Entropy Loss: Information theory and asymptotic penalty dynamics

### Information theory and surprisal

During training, how do we quantify the exact error of the model's vocabulary predictions? We turn to Claude Shannon's Information Theory.

The information content—or **surprisal**—of observing an event with probability $P$ is defined as:

$$I(x) = -\log_2 P(x)$$

If a model predicts an inevitable event with $P=1.0$, its surprisal is $-\log_2(1.0) = 0$. If it assigns an infinitesimal probability $P=0.001$, the surprisal is $-\log_2(0.001) \approx 9.96$ bits of information.

In foundation model training, the ground-truth target is represented by a **one-hot vector** $y \in \{0, 1\}^V$, where the index of the true next token $c$ has value $y_c = 1.0$, and all other vocabulary indices are zero ($y_j = 0$ for $j \ne c$).

The **Cross-Entropy Loss** ($\mathcal{L}$) measures the divergence between the true distribution $y$ and the model's predicted Softmax distribution $P$:

$$\mathcal{L} = -\sum_{j=1}^V y_j \ln(P_j)$$

Since $y_j = 0$ for all tokens except the ground-truth target $c$, this summation collapses into a single, breathtakingly concise expression:

$$\mathcal{L} = -\ln(P_c)$$

Read this as:
- $\mathcal{L}$ is the negative log-likelihood of the ground-truth token.
- The loss evaluates **exclusively** the probability the model assigned to the correct word. The competitor logits enter the equation only through the normalization denominator of $P_c$.

```text
The Cross-Entropy Penalty Curve:
  Target Probability P_c = 1.00  ==>  Loss = -ln(1.00) = 0.0000 nats (Zero penalty)
  Target Probability P_c = 0.50  ==>  Loss = -ln(0.50) = 0.6931 nats
  Target Probability P_c = 0.10  ==>  Loss = -ln(0.10) = 2.3026 nats
  Target Probability P_c = 0.01  ==>  Loss = -ln(0.01) = 4.6052 nats
  Target Probability P_c -> 0.00 ==>  Loss -> +infinity (The Asymptotic Wall)
```

This asymptotic wall penalizes confident errors with exponential severity. If a language model assigns $P_c \approx 0$ to a rare, long-tail word that appears in the training corpus, the loss skyrockets, generating massive gradient updates that force the network to retain low-frequency grammatical rules, factual names, and edge-case syntax.

---

## 7. The elegance of backpropagation: Derivative of Softmax + Cross-Entropy (dL/dz = P - y)

### The Jacobian of Softmax

A perennial question in machine learning is why Cross-Entropy and Softmax are universally coupled. The answer is found in the magical cancellation that occurs when their derivatives are joined via the chain rule.

First, consider the derivative of the Softmax probability $P_i$ with respect to an arbitrary logit $z_k$:

$$P_i = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}} = \frac{e^{z_i}}{S}$$

We apply the quotient rule $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$:

**Case 1: $i = k$ (Derivative with respect to its own logit)**

$$\frac{\partial P_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} \cdot \frac{S - e^{z_i}}{S} = P_i (1 - P_i)$$

**Case 2: $i \ne k$ (Derivative with respect to a competing logit)**

$$\frac{\partial P_i}{\partial z_k} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_k}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_k}}{S} = -P_i P_k$$

Combining both cases using the Kronecker delta $\delta_{ik}$ (where $\delta_{ik} = 1$ if $i=k$, else $0$):

$$\frac{\partial P_i}{\partial z_k} = P_i (\delta_{ik} - P_k)$$

### The chain rule synthesis

Now, we evaluate the gradient of the scalar loss $\mathcal{L} = -\sum_j y_j \ln(P_j)$ with respect to an input logit $z_k$:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{j=1}^V \frac{\partial \mathcal{L}}{\partial P_j} \frac{\partial P_j}{\partial z_k}$$

The derivative of the loss with respect to probability $P_j$ is simply $-\frac{y_j}{P_j}$:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{j=1}^V \left( -\frac{y_j}{P_j} \right) \left[ P_j (\delta_{jk} - P_k) \right]$$

The $P_j$ in the numerator and denominator cancel out completely:

$$\frac{\partial \mathcal{L}}{\partial z_k} = -\sum_{j=1}^V y_j (\delta_{jk} - P_k) = -\sum_{j=1}^V y_j \delta_{jk} + \sum_{j=1}^V y_j P_k$$

Because $y$ is a probability distribution whose sum is strictly one ($\sum_{j=1}^V y_j = 1$), and $\sum_{j=1}^V y_j \delta_{jk} = y_k$:

$$\frac{\partial \mathcal{L}}{\partial z_k} = -y_k + P_k \cdot 1 = P_k - y_k$$

In vector notation:

$$\frac{\partial \mathcal{L}}{\partial z} = P - y = \text{Predicted Probability} - \text{Actual Target}$$

```text
The Linear Gradient Miracle:
- Target Token c:      dL/dz_c = P_c - 1.0  (Always NEGATIVE: pushes logit UP toward +inf)
- Competing Tokens j:  dL/dz_j = P_j - 0.0  (Always POSITIVE: pushes logits DOWN toward -inf)
```

All complex non-linear exponentials, divisions, and logarithms dissolve into a subtraction. The gradient is linear in error, bounded strictly between $-1.0$ and $+1.0$. Unlike Sigmoid activations, it can never saturate or vanish when the model is wrong, creating an unobstructed gradient highway from the final loss back into the transformer layers.

### Flowing back into weights and hidden states

From the logit gradient $\frac{\partial \mathcal{L}}{\partial z} \in \mathbb{R}^V$, the chain rule routes the gradient backward across the linear layer $z = h \cdot W^T$:

1. **Gradient with respect to Head Weights ($W_{\text{head}}$):**
   $$\frac{\partial \mathcal{L}}{\partial W_{\text{head}}} = \left(\frac{\partial \mathcal{L}}{\partial z}\right) \otimes h^T \in \mathbb{R}^{V \times d}$$
   An outer product between the vocabulary prediction error and the incoming hidden state.
2. **Gradient with respect to Hidden State ($h$):**
   $$\frac{\partial \mathcal{L}}{\partial h} = W_{\text{head}}^T \cdot \left(\frac{\partial \mathcal{L}}{\partial z}\right) \in \mathbb{R}^d$$
   This vector is transmitted backward into the final RMSNorm layer and cascades through all 32+ Transformer blocks down to the input embeddings.

---

## 8. Step-by-step numerical walkthrough: Forward pass, soft-capping, loss, and backward pass

To make this entire mathematical architecture tangible at the hardware level, we walk through a complete, verifiable toy forward and backward pass across $d = 4$ hidden dimensions and a $V = 5$ vocabulary:

```text
   [Final Hidden State h: 1x4] ──> [0.5, -1.2, 0.8, 1.5]
                                         │
                                         ▼  (x W_head^T: 4x5)
   Raw Logits (z_raw):                   [0.69,  0.25,  1.71,  0.17, -1.24]
                                         │
                                         ▼  (Soft-capping: c=5 · tanh(z/5))
   Capped Logits (z_cap):                [0.69,  0.25,  1.65,  0.17, -1.22]
                                         │
                                         ▼  (Shift-Invariant: z - 1.6463)
   Softmax Probabilities (P):            [19.97%, 12.92%, 52.20%, 11.93%, 2.98%]
                                         │                          ▲
                                         │                     [GROUND TRUTH]
                                         ▼
   Cross-Entropy Loss:                   -ln(0.1193) = 2.1261 nats
                                         │
                                         ▼  (dL/dz_cap = P - y)
   Error Signal (dL/dz):                 [+0.196, +0.129, +0.465, -0.880, +0.028]
                                                                    ▲
                                         (Target pushed UP, competitors suppressed!)
```

### Setup and toy parameters

Let:
- Hidden dimension: $d = 4$
- Vocabulary size: $V = 5$ tokens:
  - Index 0: `[PAD]`
  - Index 1: `"the"`
  - Index 2: `"cat"`
  - Index 3: `"rabbit"` (**Ground-Truth Target**)
  - Index 4: `"dog"`
- Soft-capping threshold: $c = 5.0$
- Incoming hidden state from layer 32:
  $$h = [0.5, -1.2, 0.8, 1.5]$$
- Weight matrix $W_{\text{head}} \in \mathbb{R}^{5 \times 4}$:
  $$W_{\text{head}} = \begin{bmatrix}
   0.4 & -0.2 &  0.5 & -0.1 \\
  -0.3 &  0.6 & -0.1 &  0.8 \\
   0.7 & -0.5 &  0.2 &  0.4 \\
  -0.2 &  0.8 &  0.6 &  0.5 \\
   0.5 &  0.1 & -0.4 & -0.7
  \end{bmatrix}$$

### Step 1: Forward pass and soft-capping ($z_{\text{raw}} \to z_{\text{cap}}$)

Each logit score is the inner product $z_i = h \cdot W_{i, :}^T$. We then apply Gemma 2-style soft-capping $z_{\text{cap}} = 5.0 \cdot \tanh(z_{\text{raw}} / 5.0)$.

For the deceptive leader (index 2, `"cat"`):
- Raw score: $z_2 = (0.5)(0.7) + (-1.2)(-0.5) + (0.8)(0.2) + (1.5)(0.4) = 1.7100$
- Capped score: $5.0 \cdot \tanh(1.7100 / 5.0) \approx 1.6463$

### Step 2: Stable Softmax and Cross-Entropy Loss

Subtracting the maximum logit $m = 1.6463$ ($z' = z_{\text{cap}} - m$) guarantees numerical stability before computing exponentials. For our ground-truth target `"rabbit"` (index 3), the model predicts only $11.93\%$ probability, while the distractor `"cat"` dominates with $52.20\%$.

The Cross-Entropy Loss evaluates to:

$$\mathcal{L} = -\ln(P_3) = -\ln(0.1193) = 2.1261\text{ nats}$$

### Step 3: Backpropagation and gradient highway ($dL / dz$)

Because target $y = [0, 0, 0, 1, 0]$, the gradient highway takes $P - y$ and scales by the soft-capping derivative $(1 - \tanh^2)$ to yield the raw logit gradient:

| Index | Token | Role | Raw Logit ($z_{\text{raw}}$) | Capped ($z_{\text{cap}}$) | Probability ($P$) | Raw Gradient ($dL/dz$) | Physical Gradient Effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | `[PAD]` | Distractor | $+0.6900$ | $+0.6857$ | $19.97\%$ | $+0.1960$ | Suppress score downwards |
| `1` | `"the"` | Distractor | $+0.2500$ | $+0.2498$ | $12.92\%$ | $+0.1288$ | Suppress score downwards |
| `2` | `"cat"` | False Leader | $+1.7100$ | $+1.6463$ | $52.20\%$ | $+0.4654$ | Strongly suppress score! |
| `3` | `"rabbit"` | **Target** | $+0.1700$ | $+0.1699$ | $11.93\%$ | **$-0.8797$** | **Propel score upwards!** |
| `4` | `"dog"` | Distractor | $-1.2400$ | $-1.2152$ | $2.98\%$ | $+0.0281$ | Minimal / neutral effect |

### Step 4: Upstream gradients to weights and backbone

The resulting error signal flows backward through the chain rule:
1. **$W_{\text{head}}$ Parameter Updates:** Each row is updated via the outer product $\frac{\partial \mathcal{L}}{\partial W} = g_{\text{raw}} \otimes h^T$.
2. **Gradient Pumped to the Transformer Backbone:**
   $$\frac{\partial \mathcal{L}}{\partial h} = \sum_{i=0}^4 \left(\frac{\partial \mathcal{L}}{\partial z_{\text{raw}, i}}\right) W_{i, :} = [0.5555, -0.8956, -0.3609, -0.1899]$$

This 4-dimensional gradient vector passes through the final RMSNorm and propagates across all 32 layers of the Transformer stack.


---

## 9. The hardware memory wall: Megatron-LM vocab parallelism and Liger Kernel

### The HBM allocation explosion

Consider training Llama 3.1 8B with an 8,192 context window and batch size 4 on modern GPUs. In standard un-fused PyTorch:

```text
Logits Tensor Dimensions: [4, 8192, 128256]
Total Elements:           4 * 8,192 * 128,256 = 4,202,692,608 elements
Memory in bfloat16:       4.20B * 2 bytes     = 8.405 GB (Forward pass)
Gradient in bfloat16:     4.20B * 2 bytes     = 8.405 GB (Backward pass)
Cross-Entropy Intermediate Softmax Buffers:   ~8.4 GB
Total VRAM for Final Layer Alone:             > 25 GB!
```

### The Long-Context Crisis: When the LM Head overtakes Attention (The O(S · V) explosion)

A profound transformation occurs when models scale from 8k to 32k, 64k, and 128k context windows.

In the Transformer backbone, the quadratic $O(S^2)$ attention memory footprint has been effectively tamed by **FlashAttention** (Dao et al.), which tiles Query, Key, and Value blocks directly in GPU SRAM without ever writing the full $S \times S$ attention matrix into global HBM.

However, in the Output Layer, the un-fused logits tensor scales as $O(S \cdot V)$. Because the vocabulary dimension $V$ is massive ($128,256$ in Llama 3.1 or $256,000$ in Gemma 2), scaling the sequence length $S$ creates a staggering memory explosion:

```text
Logits Memory Scaling for Llama 3.1 (V = 128,256, bfloat16, Batch Size = 1):
- Context S = 4,096:    4,096 * 128,256 * 2 bytes   ≈  1.05 GB
- Context S = 8,192:    8,192 * 128,256 * 2 bytes   ≈  2.10 GB  (At Batch=4: 8.4 GB)
- Context S = 32,768:  32,768 * 128,256 * 2 bytes   ≈  8.40 GB  (Single sequence!)
- Context S = 131,072: 131,072 * 128,256 * 2 bytes  ≈ 33.60 GB  (Forward activation)
  Backward Gradient:   131,072 * 128,256 * 2 bytes  ≈ 33.60 GB
  TOTAL LOGITS VRAM:   33.60 GB + 33.60 GB          = 67.20 GB! (For ONE sequence!)
```

> [!CRITICAL]
> **In long-context regimes, the LM Head eclipses Attention as the #1 memory bottleneck in the entire model.**
> While FlashAttention allows an 80 GB GPU to easily process the attention layers of a 128k prompt, attempting to materialize the output logits tensor for that same 128k sequence demands 67.2 GB of VRAM for a single batch item, triggering immediate out-of-memory collapse.

On an 80 GB A100 or H100 GPU, dedicating $25\text{ GB}$ to $67\text{ GB}$ of VRAM strictly to the vocabulary projection of a single layer is a catastrophic bottleneck. The industry has converged on two state-of-the-art architectures to destroy this memory wall:

### 1. NVIDIA Megatron-LM: Vocab Parallelism (`VocabParallelCrossEntropy`)

In multi-GPU distributed clusters, Megatron-LM implements **Tensor Parallelism (TP)** across the vocabulary dimension. The linear head is instantiated as a `ColumnParallelLinear` layer.

If training across $N = 8$ GPUs (TP=8), the 128,256-token vocabulary matrix is partitioned column-wise: each GPU holds an exclusive shard of $16,032$ tokens.

```text
GPU 0 holds tokens [0      ... 16,031]
GPU 1 holds tokens [16,032 ... 32,063]
...
GPU 7 holds tokens [112,224... 128,255]
```

The engineering challenge: *How can GPUs compute Softmax if the denominator $\sum e^{z_j}$ requires summing across all 128,256 tokens?*

Gathering the full logit tensors across NVLink via `All-Gather` would saturate inter-GPU interconnects and reintroduce the memory explosion. Megatron-LM solves this via `VocabParallelCrossEntropy`:

1. **Local Maximum:** Each GPU calculates the maximum logit within its local shard: $m_{\text{local}} = \max(z_{\text{local}})$.
2. **Global Maximum Reduction:** A lightweight `all-reduce(op=MAX)` across GPUs determines the global maximum $m_{\text{global}} = \max(m_0, \dots, m_7)$—exchanging only a single scalar per token!
3. **Local Exponentials:** Each GPU computes $e^{z_{\text{local}} - m_{\text{global}}}$ and sums them to obtain $S_{\text{local}}$.
4. **Global Sum Reduction:** An `all-reduce(op=SUM)` yields the exact global denominator $S_{\text{global}} = \sum S_{\text{local}}$.
5. **Loss Computation:** The single GPU that hosts the ground-truth target token computes $\mathcal{L} = -(z_{\text{target}} - m_{\text{global}}) + \ln(S_{\text{global}})$.

Result: **Zero multi-gigabyte logit tensors are ever gathered across GPUs.** Inter-GPU communication is restricted to tiny scalar reductions.

### Context Parallelism (CP) and the distributed label slicing hazard

When pre-training on 128k context lengths across distributed clusters, Tensor Parallelism is joined by **Context Parallelism (CP)** (via `--context-parallel-size`, such as Ring Attention in Megatron-LM).

Under Context Parallelism, the sequence dimension $S$ is sharded across GPUs: an 8-GPU CP group splits 128k tokens into local sequence chunks of $S_{\text{local}} = 16,384$ tokens per GPU. At the final layer, each GPU's `lm_head` computes loss on only its local sequence partition.

This introduces a notorious engineering bug: **Label Misalignment**. If the pipeline's loss function receives the global sequence label tensor of length 128k, but the local logits tensor only contains 16k tokens, computing cross-entropy evaluates the wrong tokens and immediately diverges. Production runtimes must slice labels with exact CP-local sequence offsets ($\text{cp\_rank} \cdot S_{\text{local}}$) and execute fused loss reduction across both TP and CP communication groups.

### 2. Liger Kernel: Triton-based Fused Linear Cross-Entropy

For single-node or smaller clusters using standard Hugging Face pipelines without Megatron complexity, LinkedIn engineering developed **Liger Kernel**.

The core breakthrough is `FusedLinearCrossEntropy`. In standard PyTorch, matrix multiplication (`H @ W.T`) writes the entire 8.4 GB tensor into off-chip High Bandwidth Memory (HBM). Then, `CrossEntropyLoss` reads all 8.4 GB back into the GPU streaming multiprocessors (SMs) to compute the loss.

Liger Kernel uses OpenAI's **Triton** to fuse the linear projection and the cross-entropy computation into a single GPU kernel:
- **Chunking in Fast SRAM:** The kernel splits the sequence into chunks of 512 or 1,024 tokens.
- **On-Chip Computation:** It loads a single chunk of hidden states into ultra-fast on-chip SRAM (20 TB/s bandwidth), computes the matrix product for that chunk, immediately calculates the shifted exponentials and local loss, accumulates the gradients, and discards the logits.
- **Zero Logits Materialization:** The full `[B, S, V]` tensor is **never materialized in HBM**.

```text
Hugging Face + Liger Kernel Benchmark (Llama 3 8B, SeqLen 8K):
- Peak VRAM Allocation: Reduced by 60% (from 25 GB down to < 10 GB)
- Training Throughput:   Increased by 20% to 30%
- Supported Context:     Expands from 4K to 16K on a single GPU without OOM
```

<svg viewBox="0 0 560 210" role="img" aria-label="Comparison between Megatron-LM Vocab Parallelism and Liger Kernel Fused Linear Cross-Entropy." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="144" y="38" text-anchor="middle" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">MEGATRON-LM (Multi-GPU)</text>
<text x="144" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">VocabParallelCrossEntropy</text>
<rect x="32" y="70" width="224" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="144" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_head sharded: V / N per GPU</text>
<text x="32" y="124" text-anchor="start" style="fill:var(--c-text);font-size:11px">• Local max -> all_reduce(MAX)</text>
<text x="32" y="142" text-anchor="start" style="fill:var(--c-text);font-size:11px">• Local sum(exp) -> all_reduce(SUM)</text>
<text x="32" y="160" text-anchor="start" style="fill:var(--c-success);font-size:11px">• Zero full logits gathered over NVLink</text>
<text x="32" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">• Scales to 100k+ vocab on clusters</text>
<rect x="288" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="38" text-anchor="middle" style="fill:var(--c-success);font-size:12.5px;font-weight:700">LIGER KERNEL (Triton Fused)</text>
<text x="416" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">FusedLinearCrossEntropy</text>
<rect x="304" y="70" width="224" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="416" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Triton Kernel: MatMul + Loss Fused</text>
<text x="304" y="124" text-anchor="start" style="fill:var(--c-text);font-size:11px">• Chunked processing in GPU SRAM</text>
<text x="304" y="142" text-anchor="start" style="fill:var(--c-text);font-size:11px">• Full [B, S, V] NEVER written to HBM</text>
<text x="304" y="160" text-anchor="start" style="fill:var(--c-success);font-size:11px">• 60% VRAM savings on single GPU</text>
<text x="304" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">• Drop-in replacement for Hugging Face</text>
</svg>

---

## 10. Fine-tuning and vocabulary expansion: LoRA and the modules_to_save trap

### The vocabulary expansion dilemma

When adding domain-specific jargon or chat tokens (`<|im_start|>`, `<|im_end|>`), engineers invoke `model.resize_token_embeddings()`:
- $W_{\text{in}}$ and $W_{\text{head}}$ grow to $(V + \Delta V) \times d$.
- **Newly appended rows are initialized completely at random.**

### The silent PEFT failure

In standard LoRA training, base weights are frozen while low-rank adapters ($B \cdot A$) attach strictly to attention and MLP projections.

The trap: **Newly added rows in `lm_head` and `embed_tokens` remain frozen in their random initialization!** During inference, generating a newly added token projects random noise, collapsing the model into gibberish.

### The solution: `modules_to_save`

To properly train expanded vocabularies under PEFT/LoRA, the configuration must explicitly instruct the optimizer to unfreeze the vocabulary matrices and track their full-parameter gradients:

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]  # <-- CRITICAL PARAMETER!
)
model = get_peft_model(model, peft_config)
```

`modules_to_save` instructs PEFT to wrap the embedding and output head layers inside `ModulesToSaveWrapper`. In untied models (Llama), both matrices are fully trained and serialized into the adapter checkpoint. In tied models (Gemma 2), PEFT preserves the weight-tying pointer, preventing gradient divergence between input and output representations.

---

## 11. Inference acceleration: Speculative decoding with Medusa and EAGLE

### The memory bandwidth bottleneck of autoregressive decoding

During inference, generating text is not compute-bound; it is strictly **memory-bandwidth-bound**.

To generate a single token from an 8B parameter model in 16-bit precision, the GPU must stream $16\text{ GB}$ of model weights from HBM into the compute registers. On an NVIDIA A100 GPU with $2.0\text{ TB/s}$ memory bandwidth:

$$\text{Minimum Latency per Token} = \frac{16\text{ GB}}{2000\text{ GB/s}} = 8.0\text{ milliseconds} \implies \text{Max Throughput} \approx 125\text{ tokens/sec}$$

Even if you feed only a single token into the GPU, the tensor cores sit idle over $95\%$ of the time. To break through this physical wall, the AI community engineered **Speculative Decoding**.

Instead of generating one token at a time, a fast drafting mechanism proposes $K$ candidate future tokens, and the large model verifies all $K$ tokens in parallel in a single forward pass.

### Medusa: Multi-head speculative decoding without draft models

**Medusa** (Cai et al., 2024, arXiv:2401.10774) eliminates separate draft models by appending lightweight MLP decoding heads directly on top of the frozen backbone model's final hidden state:

```text
Hidden State h_t from Backbone:
  ├── Original lm_head  ──> Predicts token t+1 (Primary verification)
  ├── Medusa Head 1     ──> Predicts token t+2 (Speculative draft)
  ├── Medusa Head 2     ──> Predicts token t+3 (Speculative draft)
  └── Medusa Head 3     ──> Predicts token t+4 (Speculative draft)
```

In a single forward pass:
1. Medusa heads generate a combinatorial tree of future candidate paths.
2. Using **Tree Attention**, the original backbone model evaluates the entire candidate tree in a single parallel verification pass.
3. Accepted candidates are appended simultaneously, yielding **$2.2\times$ to $3.6\times$ speedups** without quality degradation.

### EAGLE: Feature-level speculative sampling

**EAGLE** (Li et al., 2024, arXiv:2401.15077) shifts speculation from volatile token IDs to continuous **Feature Space**. By extrapolating future continuous hidden states ($h_{t+1}, h_{t+2}$) and feeding them through the original model's frozen `lm_head`, contextual momentum pushes draft acceptance rates beyond $80\%$, delivering **$2.7\times$ to $3.5\times$ speedups**.

<svg viewBox="0 0 560 250" role="img" aria-label="Comparison between standard autoregressive generation, Medusa multi-head speculation, and EAGLE feature extrapolation." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="528" height="60" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="36" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">1. STANDARD AUTOREGRESSIVE</text>
<text x="28" y="56" text-anchor="start" style="fill:var(--c-text);font-size:12px">Pass 1: Token 1  ──>  Pass 2: Token 2  ──>  Pass 3: Token 3  (Memory Bandwidth Bound: 1 token / pass)</text>
<rect x="16" y="86" width="528" height="70" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="106" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:600">2. MEDUSA (Multi-Head Tree Speculation - Cai et al. 2024)</text>
<text x="28" y="124" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Backbone Hidden State h_t ──> Head 1 (t+1), Head 2 (t+2), Head 3 (t+3) in parallel</text>
<text x="28" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Tree Attention verifies candidates in 1 pass  ──>  2.2x to 3.6x speedup</text>
<rect x="16" y="166" width="528" height="70" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="186" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">3. EAGLE (Feature-Level Speculation - Li et al. ICML 2024)</text>
<text x="28" y="204" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Extrapolates top-layer hidden features h_(t+k) ──> Projects through frozen original lm_head</text>
<text x="28" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Bypasses token uncertainty, >80% draft acceptance rate  ──>  2.7x to 3.5x speedup</text>
</svg>

---

## 12. The autoregressive generation loop: Inside model.generate() and sampling strategies

When executing text generation in Hugging Face or vLLM, the entire pipeline is coordinated by the `generate()` loop. The raw logit vector $z \in \mathbb{R}^V$ produced by `lm_head` is transformed through a strict sequence of stages before producing a token:

### 1. Prefill vs. decode phase

The loop operates across two distinct computational regimes:
- **Prefill Phase:** The prompt tokens are ingested simultaneously. The attention mechanism computes $Q, K, V$ across all prompt tokens, stores all $K, V$ vectors in the **KV Cache**, and invokes `lm_head` only on the final prompt token to predict the first generated word.
- **Decode Phase:** The model generates one token per step. Only the newly sampled token is processed. It attends to past tokens via the KV cache, avoiding recomputation of historical states.

### Long-context prefill slicing: Avoiding the 100k-token GEMM blunder

A frequent architectural performance bug in custom serving pipelines occurs during the prefill phase of massive prompts (e.g. processing a 100,000-token PDF or code repository).

In a naive execution:
```python
# THE ROOKIE GEMM BLUNDER:
# Forces the GPU to compute 100,000 x 128,256 matrix multiplications!
# Allocates gigabytes of unneeded logits, causing massive TTFT latency spikes.
logits = model(input_ids).logits
```

Why is this a disaster? In the prefill phase, the model's only objective is to predict the **single first token after the prompt**. The historical logits for the preceding 99,999 prompt positions are discarded without ever being sampled!

Production inference engines (vLLM, TensorRT-LLM, Hugging Face `generate`) execute an essential optimization: **Slice before the LM Head**:
```python
# THE PRODUCTION SLICE (vLLM / Hugging Face generate):
# Extract strictly the final hidden state vector:
last_hidden_state = hidden_states[:, -1:, :]  # Shape: [1, 1, 4096]
next_token_logits = model.lm_head(last_hidden_state)  # Shape: [1, 1, 128256]
```
This single slice slashes the output layer's computational and memory footprint during prefill from $O(S \cdot V)$ down to $O(1 \cdot V)$—an instantaneous $100,000\times$ acceleration of the final layer!

### 2. LogitsProcessor pipeline

Before converting logits to probabilities, they pass through a chain of mathematical filters:

- **Repetition Penalty:** To prevent loops, previously generated tokens $g \in G$ are penalized:
  $$z_g = \begin{cases} z_g / \theta & \text{if } z_g > 0 \\ z_g \cdot \theta & \text{if } z_g \le 0 \end{cases} \quad (\theta > 1.0)$$
  Dividing positive logits by $\theta$ (e.g. $1.15$) reduces their exponentiated probability.
- **Temperature Scaling ($T$):** Controls distribution sharpness:
  $$z'_i = \frac{z_i}{T}$$
  - $T \to 0$: Distribution approaches a Dirac delta (one-hot argmax, fully deterministic).
  - $T = 1.0$: Unaltered model distribution.
  - $T > 1.0$: Flattens probabilities toward uniform distribution, increasing diversity.
- **Bad Words / Suppress Tokens:** Prohibited token indices (e.g. `<unk>` or sensitive words) have their logits set to $-\infty$, ensuring $e^{-\infty} = 0$.

### 3. Sampling strategies

- **Greedy Search:** Directly selects the highest-scoring token: $t = \text{argmax}(z)$. Fast and deterministic, but vulnerable to local optima and repetitive loops.
- **Beam Search:** Maintains $B$ parallel hypothesis trees. It expands all $B$ paths, computes cumulative log-probabilities $\sum \ln P$, and retains the top $B$ trajectories. Standard for translation and summarization, but computationally expensive for open-ended dialog.
- **Top-$K$ Sampling:** Discards all tokens outside the top $K$ highest logits (e.g. $K=50$). The remaining $K$ logits are re-normalized via Softmax, eliminating the bizarre long-tail tokens.
- **Top-$p$ (Nucleus) Sampling:** Dynamically adjusts the candidate pool. Tokens are sorted in descending order of probability:
  $$\sum_{i=1}^M P_{(i)} \ge p \quad (\text{e.g. } p = 0.90)$$
  The smallest set of tokens whose cumulative mass exceeds $p$ is retained; all others are set to $-\infty$. During predictable context, the pool shrinks to 1 or 2 tokens; during creative crossroads, the pool expands to dozens of options.

### 4. Stopping criteria and detokenization

After sampling $t_{\text{next}}$, the loop checks termination conditions:
- Did the model emit the End-of-Sequence token (`<|end_of_text|>`, `</s>`)?
- Has generation hit `max_new_tokens`?
- Did it match a custom `stop_string`?

If none apply, the token ID is appended to the context, and the tokenizer merges byte pairs back into UTF-8 characters via **Detokenization**.

<svg viewBox="0 0 560 300" role="img" aria-label="Anatomy of the autoregressive model.generate() loop showing prefill, KV cache, logits processing, sampling, and detokenization." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="528" height="42" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="42" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">1. PREFILL: Prompt Tokens → KV Cache Allocated → First Logits Vector</text>
<line x1="280" y1="58" x2="280" y2="76" marker-end="url(#out-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="80" y="78" width="400" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280" y="98" text-anchor="middle" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">2. AUTOREGRESSIVE DECODE LOOP</text>
<text x="280" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Fetch h_last [1, d] → lm_head(h_last) → Raw Logits [1, V]</text>
<line x1="280" y1="122" x2="280" y2="140" marker-end="url(#out-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="142" width="480" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">3. LOGITS PROCESSORS</text>
<text x="280" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Repetition Penalty (z / θ) ──> Temperature (z / T) ──> Suppress Bad Tokens (-inf)</text>
<line x1="280" y1="186" x2="280" y2="204" marker-end="url(#out-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="60" y="206" width="440" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-success);stroke-width:1.2"/>
<text x="280" y="226" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:700">4. SAMPLING STRATEGY</text>
<text x="280" y="242" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Greedy Argmax  |  Top-K (K=50)  |  Nucleus Top-p (p=0.90) → Sample Token ID</text>
<line x1="280" y1="250" x2="280" y2="268" marker-end="url(#out-green)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="120" y="270" width="320" height="24" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="280" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:11px">5. Check StoppingCriteria (EOS) → Detokenize to Text</text>
</svg>

---

## 13. Two engineering lenses: Training vs. inference dynamics

System engineers designing foundation model infrastructure must continuously navigate the radical asymmetries between the training and inference regimes of the output layer:

| Engineering Metric | Training Regime (Pre-training / SFT) | Inference Regime (Autoregressive Serving) |
| :--- | :--- | :--- |
| **Compute Primitive** | Dense GEMM: $H \in \mathbb{R}^{B \times S \times d}$ multiplied by $W_{\text{head}}^T$ in parallel across all tokens. | Memory-bound GEMV: Single vector $h_{-1} \in \mathbb{R}^{1 \times d}$ multiplied by $W_{\text{head}}^T$. |
| **Hardware Bottleneck** | **Compute & Capacity Bound:** High FLOP utilization on Tensor Cores; massive VRAM consumption for logits. | **Memory Bandwidth Bound:** $< 1$ FLOP/byte arithmetic intensity; stalled on HBM weight streaming. |
| **Logits Footprint** | Gigabytes ($B \cdot S \cdot V \cdot 2$ bytes). Triggers OOM without Megatron-LM or Liger Kernel. | Kilobytes ($1 \cdot 1 \cdot V \cdot 2$ bytes). Completely trivial in memory footprint. |
| **Loss & Derivatives** | Active: Computes $-\ln(P_c)$ and backpropagates $dL/dz = P - y$ across all layers. | Inactive: No loss computed; gradients and computation graphs are disabled (`torch.inference_mode()`). |
| **Optimizer Memory** | **Massive:** AdamW tracks 1st and 2nd moments in fp32 ($16$ bytes/param). For Llama 3 `lm_head`: **$8.4\text{ GB}$ pure optimizer VRAM**. | **Zero:** Optimizer states are completely discarded in production checkpoints. |
| **Parallelism Strategy** | Tensor Parallelism (TP) via Megatron `VocabParallelCrossEntropy` across NVLink. | Speculative Decoding (Medusa, EAGLE) or Pipeline/Tensor Parallelism for serving throughput. |
| **Output Processing** | Direct Softmax + Cross-Entropy Loss reduction. | Dynamic pipeline: Temperature scaling, repetition penalty, Top-$K$, Nucleus Top-$p$, detokenization. |

---

## 14. Step-by-step verification with Python and PyTorch

Below is a complete, standalone Python implementation that computes the forward projection, Gemma 2 soft-capping, shift-invariant Softmax, Cross-Entropy loss, and backpropagation gradients, matching our manual walkthrough to four decimal places.

```python
import math

def run_output_layer_walkthrough():
    # 1. Setup toy dimensions and inputs
    h = [0.5, -1.2, 0.8, 1.5]  # Hidden state from layer 32 (d=4)
    target = 3                 # Target token index: "rabbit"
    c = 5.0                    # Soft-capping threshold

    # Weight matrix W_head [V=5, d=4]
    W = [
        [ 0.4, -0.2,  0.5, -0.1],  # 0: [PAD]
        [-0.3,  0.6, -0.1,  0.8],  # 1: "the"
        [ 0.7, -0.5,  0.2,  0.4],  # 2: "cat"
        [-0.2,  0.8,  0.6,  0.5],  # 3: "rabbit" (TARGET)
        [ 0.5,  0.1, -0.4, -0.7]   # 4: "dog"
    ]

    # Step 1: Raw logits z_raw = h @ W^T
    z_raw = [sum(h[k] * W[i][k] for k in range(4)) for i in range(5)]
    print(f"1. Raw Logits (z_raw): {[round(x, 4) for x in z_raw]}")

    # Step 2: Logit Soft-Capping: c * tanh(z_raw / c)
    z_capped = [c * math.tanh(z / c) for z in z_raw]
    print(f"2. Soft-Capped Logits (z_cap): {[round(x, 4) for x in z_capped]}")

    # Step 3: Shift Invariance: z' = z - max(z)
    z_max = max(z_capped)
    z_shifted = [z - z_max for z in z_capped]
    exp_z = [math.exp(z) for z in z_shifted]
    sum_exp = sum(exp_z)
    print(f"3. Max Logit: {round(z_max, 4)} | Sum of Exponentials: {round(sum_exp, 4)}")

    # Step 4: Softmax Probabilities
    probs = [e / sum_exp for e in exp_z]
    print(f"4. Softmax Probabilities (P): {[round(x, 4) for x in probs]}")

    # Step 5: Cross-Entropy Loss: L = -ln(P_target)
    loss = -math.log(probs[target])
    print(f"5. Cross-Entropy Loss (L): {round(loss, 4)} nats")

    # Step 6: Gradient dL/dz_cap = P - y
    dL_dz_cap = [probs[i] - (1.0 if i == target else 0.0) for i in range(5)]
    print(f"6. Gradient dL/dz_cap (P - y): {[round(x, 4) for x in dL_dz_cap]}")

    # Derivative through soft-capping: d_cap/d_raw = 1 - tanh^2(z_raw / c)
    d_scale = [1.0 - math.tanh(z / c)**2 for z in z_raw]
    dL_dz_raw = [dL_dz_cap[i] * d_scale[i] for i in range(5)]
    print(f"   Gradient dL/dz_raw: {[round(x, 4) for x in dL_dz_raw]}")

    # Step 7: Weight Gradients dL/dW = dL/dz_raw (x) h
    dL_dW = [[round(dL_dz_raw[i] * h[k], 4) for k in range(4)] for i in range(5)]
    print("7. Head Weight Gradients (dL/dW):")
    for row in dL_dW:
        print(f"   {row}")

    # Step 8: Upstream Gradient into Backbone dL/dh = dL/dz_raw @ W
    dL_dh = [sum(dL_dz_raw[i] * W[i][k] for i in range(5)) for k in range(4)]
    print(f"8. Backbone Hidden Gradient (dL/dh): {[round(x, 4) for x in dL_dh]}")

if __name__ == "__main__":
    run_output_layer_walkthrough()
```

### PyTorch equivalent script

```python
import torch
import torch.nn as nn

h = torch.tensor([0.5, -1.2, 0.8, 1.5], dtype=torch.float32, requires_grad=True)
W = torch.tensor([
    [ 0.4, -0.2,  0.5, -0.1],
    [-0.3,  0.6, -0.1,  0.8],
    [ 0.7, -0.5,  0.2,  0.4],
    [-0.2,  0.8,  0.6,  0.5],
    [ 0.5,  0.1, -0.4, -0.7]
], dtype=torch.float32, requires_grad=True)

target = torch.tensor(3, dtype=torch.long)
c = 5.0

# Forward pass
z_raw = torch.matmul(h, W.T)
z_cap = c * torch.tanh(z_raw / c)
loss = nn.functional.cross_entropy(z_cap.unsqueeze(0), target.unsqueeze(0))

# Backward pass
loss.backward()

print("PyTorch Loss:", round(loss.item(), 4))
print("PyTorch dL/dh:", [round(x, 4) for x in h.grad.tolist()])
```

Executing this produces identical outputs: Loss is $2.1265$, and the gradient vector cascading into the 32nd Transformer block matches kuruşu kuruşuna: `[0.5555, -0.8956, -0.3609, -0.1899]`.

---

## The whole story in six lines

- **The LM Head bridges geometry to discrete language:** It projects continuous hidden state vectors ($h \in \mathbb{R}^d$) into vocabulary-wide log-odds ($z \in \mathbb{R}^V$) via linear projection ($W_{\text{head}}^T$).
- **Untied embeddings maximize semantic rank:** While tied weights save memory in huge vocabularies (Gemma 2), modern models (Llama 3.1, Qwen 2.5) uncouple input and output spaces to learn nearly orthogonal, full-rank representations ($\cos \theta \approx 0.0007$).
- **Soft-capping tames logit explosion:** Gemma 2's hyperbolic tangent scaling ($c \cdot \tanh(z/c)$) prevents overconfidence and repetition loops by bounding logits within $(-c, +c)$ without destroying gradients.
- **Shift invariance prevents IEEE 754 overflow:** Subtracting $\max(z)$ from logits before exponentiation guarantees all exponential terms reside within $(0, 1.0]$, shielding fp16/bf16 accelerators from catastrophic NaN failures.
- **Softmax and Cross-Entropy yield a linear gradient:** Joining their derivatives via the chain rule cancels non-linear denominators, producing the clean error signal $\frac{\partial \mathcal{L}}{\partial z} = P - y$ (Predicted minus Actual).
- **Modern systems conquer hardware walls:** Megatron-LM vocab parallelism and Liger Kernel fused Triton operators break the training memory wall, while Medusa and EAGLE speculative decoding accelerate memory-bandwidth-bound inference by $2\times$ to $3.5\times$.

---

## Glossary

- **Language Model Head (`lm_head`):** The final linear transformation layer in a language model that projects hidden states from the Transformer backbone into the vocabulary space.
- **Logits:** The unnormalized real-valued scalar scores produced by the output head, representing relative log-odds for each candidate token.
- **Softmax:** A normalized exponential function mapping arbitrary real numbers into a valid discrete probability distribution summing strictly to 1.0.
- **Shift Invariance:** The mathematical property of Softmax where adding or subtracting a constant $C$ to all inputs leaves the output probabilities completely unchanged; exploited by subtracting $\max(z)$ to prevent floating-point overflow.
- **Cross-Entropy Loss:** An information-theoretic objective measuring the divergence between the true one-hot distribution and predicted probabilities, collapsing to $\mathcal{L} = -\ln(P_c)$ for single-token prediction.
- **Predicted minus Actual ($P - y$):** The analytic derivative of Cross-Entropy Loss with respect to input logits, representing a linear, non-vanishing gradient signal.
- **Weight Tying:** An architectural practice where the input token embedding table and the output projection head share the exact same physical weight buffer ($W_{\text{out}} = W_{\text{in}}$).
- **Logit Soft-Capping:** A stabilizing technique introduced in Gemma 2 that scales logits through $c \cdot \tanh(z/c)$ to smoothly constrain them within $(-c, +c)$.
- **Vocabulary Parallelism (`VocabParallelCrossEntropy`):** A Megatron-LM distributed training technique that partitions vocabulary weights across multiple GPUs, computing distributed Cross-Entropy without gathering full logit tensors across interconnects.
- **Fused Linear Cross-Entropy:** A high-performance Triton kernel (developed by LinkedIn in Liger Kernel) that fuses matrix multiplication and loss calculation directly in GPU SRAM, eliminating multi-gigabyte HBM allocations.
- **Speculative Decoding:** An inference acceleration paradigm where a fast drafting mechanism proposes candidate tokens that the large model verifies in parallel in a single forward pass.
- **Medusa:** A draft-model-free speculative decoding architecture that trains multiple auxiliary MLP heads directly on top of the frozen backbone, evaluating candidate token trees via Tree Attention.
- **EAGLE:** A speculative sampling framework that extrapolates hidden features rather than token IDs, achieving $>80\%$ draft acceptance rates through continuous representation momentum.
- **Nucleus (Top-$p$) Sampling:** A dynamic decoding strategy that restricts the sampling pool to the smallest set of tokens whose cumulative probability mass exceeds threshold $p$.
- **Context Length (`context_len`):** The maximum sequence of tokens a model can attend to and process in a single forward pass or session, bounded by RoPE frequencies, KV cache VRAM, and output projection buffers.
- **Chunked Cross-Entropy (CCE):** An optimization technique that splits long sequences into small chunks (e.g. 512 tokens) directly in fast GPU SRAM, decoupling output layer memory from the total context length $S$.
- **Infini-Attention:** A dual-attention architecture (Google, 2024) combining local masked attention with compressive memory to handle theoretically infinite context lengths in bounded $O(1)$ memory.

---

## To dive deeper

- **Vaswani, A., et al. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — The seminal Transformer paper formalizing the linear output projection and shared embeddings.
- **Press, O., & Wolf, L. (2017).** *Using the Output Embedding to Improve Language Models.* [arXiv:1608.05859](https://arxiv.org/abs/1608.05859) — Theoretical foundation and empirical analysis of tied vs. untied embedding layers.
- **Shoeybi, M., et al. (2019).** *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) — Vocab parallelism, ColumnParallelLinear, and distributed Cross-Entropy.
- **Google DeepMind (2024).** *Gemma 2: Improving Open Language Models at a Practical Size.* [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) — Logit soft-capping formulation and 256k tied vocabulary architecture.
- **Meta AI (2024).** *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) — 128k untied vocabulary design and large-scale pretraining dynamics.
- **Munkhdalai, T., Faruqui, M., & Gopal, S. (2024).** *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention.* [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) — Compressive memory for unbounded streaming context.
- **Xiao, G., et al. (2023).** *Efficient Streaming Language Models with Attention Sinks.* [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — StreamingLLM and attention sink preservation.
- **Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024).** *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) — Multi-head tree attention speculative decoding.
- **Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024).** *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) (ICML 2024) — Feature-level extrapolation for speculative generation.
- **LinkedIn Engineering (2024).** *Liger Kernel: Efficient Triton Kernels for LLM Training.* [GitHub Repository](https://github.com/linkedin/Liger-Kernel) — Memory-efficient fused linear cross-entropy.
- **Previous Parts of A Prompt's Journey:**
  - [Tokenization: How It Works](post.html?slug=how-tokenization-works) — Part 1
  - [Inside the Embedding Layer](post.html?slug=inside-the-embedding-layer) — Part 2
  - [How Embeddings Work](post.html?slug=how-embeddings-work) — Part 3
  - [Inside Self-Attention](post.html?slug=inside-self-attention) — Part 4
  - [Inside Causal Attention](post.html?slug=inside-causal-attention) — Part 5
  - [Inside Multi-Head Attention](post.html?slug=inside-multi-head-attention) — Part 6
  - [Inside the Transformer Block](post.html?slug=inside-the-transformer-block) — Part 7
