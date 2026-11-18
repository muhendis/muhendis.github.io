When you submit a prompt to a large language model, the sequence of operations we examined across the previous six parts has already transformed raw text into continuous geometry and rich dynamic context:
1. At the [Tokenization](post.html?slug=tokenization-how-it-works) desk, text is chopped into discrete integer IDs (`[1054, 492, 281]`).
2. The [Embedding Layer](post.html?slug=embedding-layer-deep-dive) maps these integers into continuous vector space and embeds rotational positional encoding (RoPE).
3. The [Semantic Vector Space](post.html?slug=semantic-embeddings-deep-dive) establishes the geometric coordinate system where angles and distances encode meaning.
4. The [Self-Attention Mechanism](post.html?slug=self-attention-deep-dive) lets tokens project Query, Key, and Value vectors to attend to one another, synthesizing context-enriched representations.
5. [Causal Attention](post.html?slug=inside-causal-attention) enforces the arrow of time, masking future tokens behind lower-triangular matrices to guarantee strictly autoregressive generation.
6. [Multi-Head Attention](post.html?slug=inside-multi-head-attention) shatters the single-projector compromise, dedicating independent subspaces to distinct grammatical axes (subject–verb, local adjacency).

Recall our running 3-token prompt from Parts 4, 5, and 6: **`["dog", "cat", "chased"]`** ("the dog chased the cat"). In Part 6, we computed Multi-Head Attention for each token, concatenated the subspace heads, multiplied by $W^O$, and obtained a multi-perspective attention output vector ($O$) for each word.

At this exact juncture, a fundamental architectural dilemma emerges: **Can we simply pass this attention output directly into the next attention layer or output projection?**

The answer is an unequivocal **no**. In isolation, attention crashes into two fatal walls:

1. **The Gradient Wall (Vanishing Gradient):** In deep architectures (32 to 100+ layers), chaining functions linearly as $h_l = f_l(h_{l-1})$ multiplies local Jacobian derivatives via the backpropagation chain rule. Within 20 layers, gradients shrink to one-millionth ($r^{20} \approx 10^{-6}$); lower layers freeze and fail to learn.
2. **The Convex Hull Barrier:** Softmax weights are non-negative and sum strictly to one ($\sum_j A_{ij} = 1, A_{ij} \ge 0$). By definition, attention outputs are **convex combinations** (weighted averages) of the input Value vectors. Much like mixing yellow and blue paint can only produce shades of green, attention can never synthesize a brand-new color (a novel logical concept, XOR logic, or factual knowledge) absent from the input palette.

Modern architectures resolve these limits by enclosing attention within a self-contained, modular engine: **The Transformer Block**.

The block surrounds the attention core with three vital systems:
- **Residual Connections (Skip Connection):** Establishes an identity gradient highway ($\mathbf{I} + f'(x)$) through $y = x + f(x)$, allowing backpropagation signals to traverse 100+ layers without decay.
- **Layer Normalization (LayerNorm / RMSNorm):** Acts as an audio mixing console, stabilizing activation variance and preventing the repeated skip additions from blowing up.
- **Feed-Forward Network (FFN / MLP):** Expands the dimension to $4d_{\text{model}}$ and applies non-linear activations (GELU, SwiGLU) to break out of the convex hull; storing factual world knowledge across two-thirds of the model's total parameter budget as an associative key-value memory.

In this article, the 7th step in our series, we dissect the anatomy of a single Transformer block. We trace our running three-token prompt through manual tensor arithmetic, explore the geometric mechanics, and verify every step with runnable PyTorch code.

**In this article**

- [1. The big picture and data path: Anatomy of a single transformer block](#1-the-big-picture-and-data-path-anatomy-of-a-single-transformer-block)
- [2. Residual connections: The identity highway and the philosophy of deltas](#2-residual-connections-the-identity-highway-and-the-philosophy-of-deltas)
- [3. The vanishing gradient catastrophe: Signals decaying to one-millionth across 20 layers](#3-the-vanishing-gradient-catastrophe-signals-decaying-to-one-millionth-across-20-layers)
- [4. How residual connections solve the crisis: The I + f'(x) guarantee](#4-how-residual-connections-solve-the-crisis-the-i-fx-guarantee)
- [5. Layer Normalization (LayerNorm): The mixing console of the orchestra](#5-layer-normalization-layernorm-the-mixing-console-of-the-orchestra)
- [6. Step 1 after attention: Post-attention residual and LayerNorm numerical walkthrough](#6-step-1-after-attention-post-attention-residual-and-layernorm-numerical-walkthrough)
- [7. The invisible boundary of attention: The Convex Hull barrier](#7-the-invisible-boundary-of-attention-the-convex-hull-barrier)
- [8. Two-thirds of the parameters: The FFN as an associative key-value memory](#8-two-thirds-of-the-parameters-the-ffn-as-an-associative-key-value-memory)
- [9. Evolution of activation functions: ReLU, GELU, and SwiGLU](#9-evolution-of-activation-functions-relu-gelu-and-swiglu)
- [10. Step 2 through the FFN: Feed-forward and second residual numerical walkthrough](#10-step-2-through-the-ffn-feed-forward-and-second-residual-numerical-walkthrough)
- [11. Block evolution in modern production models: Llama 3, Gemma 2, and DeepSeek-V3](#11-block-evolution-in-modern-production-models-llama-3-gemma-2-and-deepseek-v3)
- [12. Two engineering lenses: Training vs inference dynamics](#12-two-engineering-lenses-training-vs-inference-dynamics)
- [13. Step-by-step verification with PyTorch](#13-step-by-step-verification-with-pytorch)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [To dive deeper](#to-dive-deeper)

---

## 1. The big picture and data path: Anatomy of a single transformer block

### Division of labor between two engines: Communication (Attention) and computation (FFN)

A Transformer block is not an arbitrary assembly of layers; it divides labor cleanly between two distinct engines:

1. **The Attention sub-layer (Communication):** Operates along the sequence axis ($N \times N$), enabling tokens to exchange context. Each token asks: *"Which other tokens in this sequence clarify my contextual meaning?"* This operation blends information across tokens.
2. **The FFN sub-layer (Computation & Memory):** Once communication completes, each token retreats into its own isolated channel. The FFN is applied **identically and independently** to each token's $d_{\text{model}}$ vector (token-wise). There is zero sequence-level cross-talk. Here, the token asks: *"How should I process the context I gathered, which factual knowledge must I recall, and how do I transform my internal representation?"*

```
Attention = The town hall meeting (tokens converse and share context).
FFN       = Retreating to private desks (individual thinking, calculation, and recall).
```

### Inside-the-block data flow and the Pre-LN vs Post-LN evolution

The data path within a single Transformer block consists of eight stages:

<svg viewBox="0 0 560 284" role="img" aria-label="Data path through one transformer block, Post-LN layout. Sub-layer 1, communication: token input X becomes the position-enriched Z, multi-head self-attention produces AttnOut, a skip path carries Z around it, the two are summed and LayerNorm or RMSNorm gives the intermediate H1. Sub-layer 2, computation: H1 goes through the feed-forward network W2 sigma of W1 H1 plus b1, plus b2, a skip path carries H1 around it, the sum is normalized again and gives the final block output H2." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bf-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bf-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="26" width="528" height="92" rx="8" style="fill:none;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="28" y="20" text-anchor="start" style="fill:var(--c-accent);font-size:12.5px;font-weight:600">sub-layer 1: communication (attention)</text>
<rect x="16" y="148" width="528" height="92" rx="8" style="fill:none;stroke:var(--c-success);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="90" y="142" text-anchor="start" style="fill:var(--c-success);font-size:12.5px;font-weight:600">sub-layer 2: computation (FFN)</text>
<rect x="26" y="66" width="38" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="45.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">X</text>
<line x1="64" y1="84" x2="78" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="80" y="66" width="44" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="102.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">Z</text>
<line x1="124" y1="84" x2="140" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="142" y="62" width="150" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="217.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">multi-head attention</text>
<line x1="292" y1="84" x2="297" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="312" cy="84" r="13" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.4"/><text x="312" y="89.5" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700">+</text>
<path d="M102 66 C 102 44, 312 44, 312 69" marker-end="url(#bf-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="206" y="37" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">skip path (Z)</text>
<line x1="325" y1="84" x2="338" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="340" y="66" width="100" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="390.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LayerNorm</text>
<line x1="440" y1="84" x2="456" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="458" y="66" width="72" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="494.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">H₁</text>
<path d="M494 102 V130 H52 V186" marker-end="url(#bf-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="26" y="188" width="52" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="52.0" y="210.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">H₁</text>
<line x1="78" y1="206" x2="92" y2="206" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="94" y="184" width="198" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="193.0" y="202.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">FFN / MLP</text>
<text x="193.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">W₂·σ(W₁·H₁ + b₁) + b₂</text>
<line x1="292" y1="206" x2="297" y2="206" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="312" cy="206" r="13" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.4"/><text x="312" y="211.5" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700">+</text>
<path d="M60 188 C 60 166, 312 166, 312 191" marker-end="url(#bf-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="196" y="159" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">skip path (H₁)</text>
<line x1="325" y1="206" x2="338" y2="206" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="340" y="188" width="100" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="390.0" y="210.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LayerNorm</text>
<line x1="440" y1="206" x2="456" y2="206" marker-end="url(#bf-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="458" y="188" width="72" height="36" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="494.0" y="210.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">H₂</text>
<text x="16" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Post-LN (2017), as drawn: the norm sits after each sum.</text>
<text x="16" y="278" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Pre-LN models (Llama, Mistral) move it in front of each sub-layer, off the skip path.</text>
</svg>

The original 2017 Transformer (Vaswani et al.) applied normalization after the addition, in a **Post-LN** layout:

$$H_1 = \text{LayerNorm}(Z + \text{MultiHead}(Z))$$

$$H_2 = \text{LayerNorm}(H_1 + \text{FFN}(H_1))$$

In Post-LN, normalization sits directly on the residual skip path. As networks deepen ($L > 12$), backpropagated error signals are repeatedly attenuated by the normalization derivative. Without a delicate learning-rate warmup schedule, gradients explode or vanish, destabilizing early training steps.

Modern frontier models (Llama 3, Mistral, Gemma 2) place normalization at the input of each sub-layer, adopting **Pre-LN / Pre-RMSNorm**:

$$H_1 = Z + \text{MultiHead}(\text{Norm}(Z))$$

$$H_2 = H_1 + \text{FFN}(\text{Norm}(H_1))$$

In Pre-LN, the residual skip path passes through untouched as an uninhibited identity highway (the residual stream), enabling stable training across hundreds of layers.

---

## 2. Residual connections: The identity highway and the philosophy of deltas

### Learning deltas versus reconstructing from scratch

The principle of the residual connection is straightforward: rather than forcing a layer to recreate an entire target representation $\mathcal{H}(x)$, the layer learns only an incremental delta $\mathcal{F}(x)$, adding it directly to the original input:

$$y = x + \mathcal{F}(x)$$

This simple formulation confers three profound engineering benefits:

1. **Identity as the default:** If layer weights are initialized near zero or fail to discover a useful feature, the layer gracefully defaults to an identity pass: $\mathcal{F}(x) \approx 0 \implies y \approx x$. Information traverses the layer undamaged.
2. **Refining rather than redrawing:** Without skip connections, if layer 45 receives a bad gradient update, the representations crafted by the preceding 44 layers are destroyed. With residuals, layer 45 merely adds a small touch-up brushstroke to an existing canvas.
3. **Preserving low-level features:** Morphological and syntactic features learned in early layers can propagate directly to the final layer without being obscured by higher-level abstractions.

### Skip connection circuit diagram and identity guarantee

Think of a residual block as a parallel circuit or dual-track railway. The input $x$ splits into two parallel tracks:

<svg viewBox="0 0 560 150" role="img" aria-label="A residual block as two parallel tracks. The input x goes through the processing track F(x) and, at the same time, along an identity express line that skips it; the two meet at a sum and the output is y = x plus F(x)." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sk-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sk-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<rect x="16" y="70" width="56" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="44.0" y="94.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">x</text>
<path d="M72 82 C 100 82, 100 36, 130 36 H 352 C 380 36, 380 74, 380 76" marker-end="url(#sk-g)" style="fill:none;stroke:var(--c-success);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="240" y="28" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:600">skip / identity track: the express line</text>
<line x1="72" y1="98" x2="168" y2="98" marker-end="url(#sk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="76" width="150" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="245.0" y="94.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">F(x)</text>
<text x="245.0" y="110.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">processing track</text>
<line x1="320" y1="98" x2="365" y2="98" marker-end="url(#sk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="380" cy="98" r="13" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.4"/><text x="380" y="103.5" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700">+</text>
<line x1="393" y1="98" x2="420" y2="98" marker-end="url(#sk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="422" y="76" width="122" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="483.0" y="102.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">y = x + F(x)</text>
<text x="16" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">However noisy F(x) gets, x always has an untouched path to the output.</text>
</svg>

No matter how non-linear, saturating, or noisy the function $\mathcal{F}(x)$ becomes, a clean path exists for input $x$ to reach the output unchanged.

---

## 3. The vanishing gradient catastrophe: Signals decaying to one-millionth across 20 layers

### The chain rule product and Jacobian decay

To appreciate the true role of residual connections, we must confront the crisis they solved: **The Vanishing Gradient Problem**.

In a plain $L$-layer network lacking skip connections, forward propagation is function composition:

$$h_l = f_l(h_{l-1}) \quad (l = 1, \dots, L)$$

During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to the initial input $h_0$ is evaluated via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdot \frac{\partial h_{L-1}}{\partial h_{L-2}} \cdots \frac{\partial h_1}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^L \frac{\partial h_l}{\partial h_{l-1}}$$

Each $\frac{\partial h_l}{\partial h_{l-1}}$ term represents a local Jacobian matrix. If the average spectral norm or scalar derivative of these transitions is below unity ($r < 1$ — due to saturating activations or small weight norms), gradient magnitude decays exponentially with depth:

$$\left\| \frac{\partial \mathcal{L}}{\partial h_0} \right\| \propto r^L$$

### Numerical decay table: How much does gradient diminish over 20 layers?

Tracking how gradient signals decay across layers for different attenuation rates $r$:

| Layer Depth ($L$) | $r = 0.9$ (Mild Decay) | $r = 0.7$ (Moderate Decay) | $r = 0.5$ (Severe Decay) |
|:---|:---|:---|:---|
| **0 (Loss layer)** | 1.000 | 1.000 | 1.000 |
| **5 layers** | 0.590 | 0.168 | 0.031 |
| **10 layers** | 0.349 | 0.028 | 0.00098 |
| **15 layers** | 0.206 | 0.0047 | 0.0000305 |
| **20 layers** | 0.122 | 0.0008 | **0.00000095 ($9.5 \times 10^{-7}$)** |

Look closely at the $r = 0.5$ column: after just 20 layers, the backpropagated gradient has shriveled to **under one-millionth** of its initial magnitude. In an 80-layer architecture like Llama 3 70B, $(0.5)^{80} \approx 8.27 \times 10^{-25}$. In IEEE 754 float16 or bfloat16, this underflows directly to zero. Weights in early layers receive no training signal, remaining frozen in their random initialization.

---

## 4. How residual connections solve the crisis: The I + f'(x) guarantee

### The identity shield in backprop: Highway remains open even if layer dies

With residual connections, the layer equation becomes $h_l = h_{l-1} + f_l(h_{l-1})$. Differentiating with respect to the input $h_{l-1}$:

$$\frac{\partial h_l}{\partial h_{l-1}} = \frac{\partial}{\partial h_{l-1}} \left[ h_{l-1} + f_l(h_{l-1}) \right] = \mathbf{I} + f'_l(h_{l-1})$$

Here, $\mathbf{I}$ is the **Identity Matrix**, derived from the skip path. Expanding the chain rule product:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^L \left( \mathbf{I} + f'_l(h_{l-1}) \right)$$

Multiplying out the product terms yields:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \left( \mathbf{I} + \sum_{l=1}^L f'_l + \sum \text{cross terms} \right)$$

> [!IMPORTANT]
> **The Identity Shield:** Even if every single sub-layer derivative $f'_l$ collapses to zero (due to dying units or saturation), the additive $\mathbf{I}$ guarantees that the overall derivative retains at least unit magnitude. Error signals travel through the identity matrix directly back to layer zero, completely unimpeded.

### Numerical comparison: 20 layers with and without residuals

Assuming $f'(x) \approx 0.5$ (each transformation halves the signal):

| Layer Depth ($L$) | Without Residual: $(0.5)^L$ | With Residual: $(1 + 0.5)^L$ |
|:---|:---|:---|
| **0** | 1.000 | 1.000 |
| **5** | 0.031 | 7.59 |
| **10** | 0.00098 | 57.66 |
| **20** | **0.00000095** | **3325.26** |

The same numbers on a log axis, over every layer from 0 to 20:

<svg viewBox="0 0 560 316" role="img" aria-label="Line chart on a logarithmic axis of how much gradient survives after 0 to 20 layers. Without residual connections, a per-layer factor of 0.9 leaves 0.122 after 20 layers, 0.7 leaves 0.0008, and 0.5 leaves 9.5 times 10 to the minus 7, under one millionth. With a residual connection the factor becomes 1 + 0.5 and the signal grows to about 3,325 instead: no vanishing, but a growth that normalization has to tame." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<line x1="66" y1="240.0" x2="420" y2="240.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="244.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁷</text>
<line x1="66" y1="220.9" x2="420" y2="220.9" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="224.9090909090909" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁶</text>
<line x1="66" y1="201.8" x2="420" y2="201.8" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="205.8181818181818" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁵</text>
<line x1="66" y1="182.7" x2="420" y2="182.7" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="186.72727272727272" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁴</text>
<line x1="66" y1="163.6" x2="420" y2="163.6" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="167.63636363636363" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻³</text>
<line x1="66" y1="144.5" x2="420" y2="144.5" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="148.54545454545456" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻²</text>
<line x1="66" y1="125.5" x2="420" y2="125.5" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="129.45454545454544" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻¹</text>
<line x1="66" y1="106.4" x2="420" y2="106.4" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="60" y="110.36363636363637" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">1</text>
<line x1="66" y1="87.3" x2="420" y2="87.3" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="91.27272727272728" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10¹</text>
<line x1="66" y1="68.2" x2="420" y2="68.2" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="72.18181818181819" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10²</text>
<line x1="66" y1="49.1" x2="420" y2="49.1" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="53.09090909090909" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10³</text>
<line x1="66" y1="30.0" x2="420" y2="30.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="34.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁴</text>
<text x="66.0" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">0</text>
<text x="154.5" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">5</text>
<text x="243.0" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">10</text>
<text x="331.5" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">15</text>
<text x="420.0" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">20</text>
<text x="243.0" y="272" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">layers the gradient travels back through</text>
<text x="16" y="18" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">surviving gradient (log scale)</text>
<path d="M66.0 106.4 L83.7 103.0 L101.4 99.6 L119.1 96.3 L136.8 92.9 L154.5 89.6 L172.2 86.2 L189.9 82.8 L207.6 79.5 L225.3 76.1 L243.0 72.7 L260.7 69.4 L278.4 66.0 L296.1 62.7 L313.8 59.3 L331.5 55.9 L349.2 52.6 L366.9 49.2 L384.6 45.9 L402.3 42.5 L420.0 39.1" style="fill:none;stroke:var(--c-success);stroke-width:2"/>
<circle cx="154.5" cy="89.6" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="72.7" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="55.9" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="39.1" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="43.128791996921706" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">1 + 0.5 (residual)</text>
<text x="428.0" y="57.128791996921706" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">3,325</text>
<path d="M66.0 106.4 L83.7 107.2 L101.4 108.1 L119.1 109.0 L136.8 109.9 L154.5 110.7 L172.2 111.6 L189.9 112.5 L207.6 113.4 L225.3 114.2 L243.0 115.1 L260.7 116.0 L278.4 116.8 L296.1 117.7 L313.8 118.6 L331.5 119.5 L349.2 120.3 L366.9 121.2 L384.6 122.1 L402.3 123.0 L420.0 123.8" style="fill:none;stroke:var(--c-accent);stroke-width:2"/>
<circle cx="154.5" cy="110.7" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="115.1" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="119.5" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="123.8" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="127.83467821407595" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">r = 0.9</text>
<text x="428.0" y="141.83467821407595" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0.122</text>
<path d="M66.0 106.4 L83.7 109.3 L101.4 112.3 L119.1 115.2 L136.8 118.2 L154.5 121.1 L172.2 124.1 L189.9 127.1 L207.6 130.0 L225.3 133.0 L243.0 135.9 L260.7 138.9 L278.4 141.9 L296.1 144.8 L313.8 147.8 L331.5 150.7 L349.2 153.7 L366.9 156.6 L384.6 159.6 L402.3 162.6 L420.0 165.5" style="fill:none;stroke:var(--c-warn);stroke-width:2"/>
<circle cx="154.5" cy="121.1" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="135.9" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="150.7" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="165.5" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="169.5080210854656" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">r = 0.7</text>
<text x="428.0" y="183.5080210854656" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0.0008</text>
<path d="M66.0 106.4 L83.7 112.1 L101.4 117.9 L119.1 123.6 L136.8 129.4 L154.5 135.1 L172.2 140.8 L189.9 146.6 L207.6 152.3 L225.3 158.1 L243.0 163.8 L260.7 169.6 L278.4 175.3 L296.1 181.1 L313.8 186.8 L331.5 192.6 L349.2 198.3 L366.9 204.1 L384.6 209.8 L402.3 215.6 L420.0 221.3" style="fill:none;stroke:var(--c-danger);stroke-width:2"/>
<circle cx="154.5" cy="135.1" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="163.8" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="192.6" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="221.3" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="225.30236198079282" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">r = 0.5</text>
<text x="428.0" y="239.30236198079282" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">9.5×10⁻⁷</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Without the skip path the signal dies geometrically.</text>
<text x="16" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">With it, the signal can only grow, and LayerNorm reins that growth in.</text>
</svg>

The residual connection prevents gradient extinction. However, $(1 + 0.5)^{20} \approx 3325$ exposes another danger: **Exploding Gradients** and runaway variance accumulation.

The governor that tames this explosive growth across successive additions is the residual connection's essential partner: **Layer Normalization**.

---

## 5. Layer Normalization (LayerNorm): The mixing console of the orchestra

### Mathematical formulation and mixer analogy: Balancing internal activations

Repeated skip additions ($x + \text{Attn}(x) + \text{FFN}(x)$) cause activation norms and variances to scale up with depth.

**Layer Normalization (Ba, Kiros, & Hinton 2016)** stabilizes signals by normalizing each token's $d_{\text{model}}$ feature vector independently:

```
Vector input: x = [x_1, x_2, ..., x_d]
```

1. **Mean ($\mu$):**
$$\mu = \frac{1}{d} \sum_{i=1}^d x_i$$

2. **Variance ($\sigma^2$):**
$$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

3. **Standardization ($\hat{x}$):**
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

4. **Learnable Affine Scaling and Shift ($y$):**
$$y_i = \gamma_i \hat{x}_i + \beta_i$$

Here, $\epsilon$ ($10^{-5}$ or $10^{-6}$) prevents division by zero. Learnable parameters $\gamma$ (gain) and $\beta$ (bias) restore expressive capacity if the network requires non-zero means.

```
Analogy: Think of an audio mixing console. If one instrument (one feature channel)
spikes to 50 dB, the master amplifier distorts. LayerNorm acts as a dynamic leveler:
it measures the energy across channels for that single moment, centering it to 0 dB
and unit dynamic range.
```

### Why not BatchNorm? Variable sequence lengths and the batch=1 dilemma in NLP

Why was **Batch Normalization (BatchNorm)**, which transformed computer vision, abandoned in large language models?

```
BatchNorm (Vision): Aggregates statistics across the BATCH axis. (Same channel across images).
LayerNorm (NLP):    Aggregates statistics across the FEATURE axis. (Inside one token's vector).
```

| Criterion | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) |
|:---|:---|:---|
| **Axis of Statistics** | Across mini-batch samples | Across hidden dimensions of a single token |
| **Variable Sequence Length** | Padding tokens skew batch statistics | Entirely sequence-length agnostic |
| **Inference Behavior** | Relies on running population stats from training | Calculated on the fly per token |
| **Batch Size = 1 Scenario** | Variance is zero; crashes at single prompt | Operates identically with batch size 1 |

Modern architectures (Llama 3, Gemma 2) simplify LayerNorm into **RMSNorm** (Zhang & Sennrich 2019) by omitting the mean-centering step:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad \text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$

RMSNorm saves 10% to 50% GPU memory bandwidth by eliminating the extra memory pass needed to calculate and subtract the mean.

---

## 6. Step 1 after attention: Post-attention residual and LayerNorm numerical walkthrough

### Skip addition: Combining Z with the attention output

Recall our concrete tensors from Parts 4, 5, and 6. For $d_{\text{model}} = 4$ and token 1 (`"dog"`):

Input embedding vector:
$$z_1 = [0.210, \; 0.820, \; 0.130, \; 0.440]$$

Post-attention output vector:
$$o_1 = [1.005, \; 1.041, \; 1.012, \; 0.976]$$

**Step 1: Skip Addition ($r_1 = z_1 + o_1$):**

$$r_1[0] = 0.210 + 1.005 = 1.215$$

$$r_1[1] = 0.820 + 1.041 = 1.861$$

$$r_1[2] = 0.130 + 1.012 = 1.142$$

$$r_1[3] = 0.440 + 0.976 = 1.416$$

$$r_1 = [1.215, \; 1.861, \; 1.142, \; 1.416]$$

### Step-by-step LayerNorm arithmetic: Mean centering and variance scaling

Passing $r_1$ through LayerNorm ($\gamma = [1, 1, 1, 1]$, $\beta = [0, 0, 0, 0]$, $\epsilon = 10^{-5}$):

**1. Mean ($\mu$):**

$$\mu = \frac{1.215 + 1.861 + 1.142 + 1.416}{4} = \frac{5.634}{4} = 1.4085$$

**2. Squared deviations and Variance ($\sigma^2$):**

$$(1.215 - 1.4085)^2 = (-0.1935)^2 \approx 0.03744$$

$$(1.861 - 1.4085)^2 = (0.4525)^2 \approx 0.20476$$

$$(1.142 - 1.4085)^2 = (-0.2665)^2 \approx 0.07102$$

$$(1.416 - 1.4085)^2 = (0.0075)^2 \approx 0.00006$$

$$\sigma^2 = \frac{0.03744 + 0.20476 + 0.07102 + 0.00006}{4} = \frac{0.31328}{4} = 0.07832$$

Standard deviation denominator:

$$\sqrt{\sigma^2 + \epsilon} = \sqrt{0.07832 + 0.00001} = \sqrt{0.07833} \approx 0.27987$$

**3. Normalized Intermediate Representation ($h_1$):**

$$h_1[0] = \frac{1.215 - 1.4085}{0.27987} = \frac{-0.1935}{0.27987} \approx -0.691$$

$$h_1[1] = \frac{1.861 - 1.4085}{0.27987} = \frac{0.4525}{0.27987} \approx 1.617$$

$$h_1[2] = \frac{1.142 - 1.4085}{0.27987} = \frac{-0.2665}{0.27987} \approx -0.952$$

$$h_1[3] = \frac{1.416 - 1.4085}{0.27987} = \frac{0.0075}{0.27987} \approx 0.027$$

$$h_1 = [-0.691, \; 1.617, \; -0.952, \; 0.027]$$

This $h_1$ vector has zero mean and unit variance. Unchecked growth has been reined in, preparing the representation for the FFN.

---

## 7. The invisible boundary of attention: The Convex Hull barrier

### The geometric trap of Softmax: Why weighted averages hit a wall

A pivotal mathematical constraint often overlooked in Transformer tutorials is that **Self-Attention cannot synthesize new conceptual coordinates on its own.**

The bottleneck lies in Softmax:

$$A_{ij} \ge 0 \quad \text{and} \quad \sum_{j=1}^N A_{ij} = 1$$

These conditions define a **Convex Combination**:

$$\text{head}_i = \sum_{j=1}^N A_{ij} v_j$$

Geometrically, every output vector is trapped inside the **Convex Hull** formed by the input Value vectors $\{v_1, v_2, \dots, v_N\}$. It cannot generate points outside this boundary.

Here is that barrier with real numbers: the 2-dimensional value vectors and outputs of head 1 from Part 6:

<svg viewBox="0 0 560 300" role="img" aria-label="Convex hull of attention, drawn with real 2-dimensional numbers from Part 6, head 1. The value vectors v dog at 0.34, 0.95, v cat at 1.32, 0.53 and v chased at 1.20, 1.41 form a triangle. Head 1&#x27;s outputs, 0.870, 0.723 for cat and 1.064, 1.105 for chased, both fall inside it, because softmax weights are non-negative and sum to 1. A point such as 1.45, 0.25 lies outside and no attention weighting can produce it; the FFN, with its non-linearity, can." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="hl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<line x1="40" y1="270" x2="360" y2="270" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="40" y1="270" x2="40" y2="20" style="stroke:var(--c-border);stroke-width:1"/>
<text x="140.0" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">0.5</text>
<text x="32" y="189.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">0.5</text>
<text x="240.0" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1.0</text>
<text x="32" y="104.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">1.0</text>
<text x="340.0" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1.5</text>
<text x="32" y="19.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">1.5</text>
<polygon points="108.0,108.5 304.0,179.9 280.0,30.3" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.5"/>
<circle cx="108.0" cy="108.5" r="5" style="fill:var(--c-accent)"/>
<text x="108.0" y="126.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">v_dog</text>
<circle cx="304.0" cy="179.9" r="5" style="fill:var(--c-accent)"/>
<text x="304.0" y="197.9" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">v_cat</text>
<circle cx="280.0" cy="30.3" r="5" style="fill:var(--c-accent)"/>
<text x="280.0" y="20.3" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">v_chased</text>
<circle cx="214.0" cy="147.1" r="4.5" style="fill:var(--c-accent-2);stroke:var(--c-bg);stroke-width:1.5"/>
<text x="222.0" y="151.1" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px">o_cat</text>
<circle cx="252.8" cy="82.2" r="4.5" style="fill:var(--c-accent-2);stroke:var(--c-bg);stroke-width:1.5"/>
<text x="260.8" y="86.2" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px">o_chased</text>
<circle cx="330.0" cy="227.5" r="6" style="fill:none;stroke:var(--c-danger);stroke-width:1.8;stroke-dasharray:3 2"/>
<text x="320.0" y="231.5" text-anchor="end" style="fill:var(--c-danger);font-size:11px">unreachable</text>
<text x="384" y="40" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">a weighted average</text>
<text x="384" y="58" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">weights ≥ 0, sum to 1,</text>
<text x="384" y="73" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">so every output stays</text>
<text x="384" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">inside the triangle</text>
<text x="384" y="140" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:600">purple: real outputs</text>
<text x="384" y="158" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">head 1, Part 6:</text>
<text x="384" y="173" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">both land inside</text>
<text x="384" y="188" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">the hull of v</text>
<text x="384" y="226" text-anchor="start" style="fill:var(--c-success);font-size:13px;font-weight:600">FFN breaks out</text>
<text x="384" y="244" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">non-linearity reaches</text>
<text x="384" y="259" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">points no mixture can</text>
</svg>

### The paint-mixing analogy and the non-linear XOR dilemma

Consider an artist's palette:
- You have two tubes of paint: **Yellow** ($v_1$) and **Blue** ($v_2$).
- Attention mixes these paints in various proportions: $80\%$ blue $+ 20\%$ yellow $\to$ forest green; $30\%$ blue $+ 70\%$ yellow $\to$ lime green.
- Regardless of attention weight complexity, you can never produce **Red** if red is not on the palette.

A similar limitation applies to logical reasoning. The **XOR (Exclusive OR)** function is linearly inseparable. No convex combination of inputs can separate its classes.

Attention routes information between tokens, but synthesizing new concepts, recalling unmentioned facts, and performing non-linear logic requires an engine that shatters this boundary: **The Feed-Forward Network**.

---

## 8. Two-thirds of the parameters: The FFN as an associative key-value memory

### The parameter budget: Why 66% of weights live inside the FFN

Inspecting the configuration of modern LLMs reveals an eye-opening distribution: roughly **two-thirds ($66\%$)** of all parameters reside in the FFN rather than the attention mechanism.

Standard Transformer FFNs project activations up to four times the model dimension ($d_{\text{model}} \to 4d_{\text{model}}$) before projecting back down ($4d_{\text{model}} \to d_{\text{model}}$):

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

Matrix parameter counts:
- $W_1 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}} \implies 4 d_{\text{model}}^2$ weights.
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}} \implies 4 d_{\text{model}}^2$ weights.
- Total FFN parameters: $\mathbf{8 d_{\text{model}}^2}$ weights.

By comparison, the four Multi-Head Attention projection matrices ($W_Q, W_K, W_V, W^O$) total:
- Total MHA parameters: $\mathbf{4 d_{\text{model}}^2}$ weights.

The FFN contains double the parameter budget of the attention mechanism.

### Geva et al. 2021 discovery: W1 keys, W2 values, and factual recall

What purpose does this massive capacity serve? Geva et al. (2021, *Transformer Feed-Forward Layers Are Key-Value Memories*) demonstrated that FFNs operate as **Associative Key-Value Memories**:

1. **$W_1$ acts as KEYS:** Columns detect conceptual patterns in the input representation. For example, a neuron fires ($u_i > 0$) when detecting patterns like *"capital of France"* or a Python function definition.
2. **The non-linear activation $\sigma$ acts as a THRESHOLD FILTER:** Suppresses irrelevant keys (e.g., zeroing out negatives via ReLU), letting only active keys pass.
3. **$W_2$ acts as VALUES:** Columns corresponding to active keys inject stored factual information into the residual stream, such as adding the vector for *"Paris"*.

<svg viewBox="0 0 560 210" role="img" aria-label="The FFN as an associative key-value memory. The token&#x27;s vector is read from the residual stream; W1 projection matches it against keys; the activation acts as a threshold filter so only matched keys fire, in our walkthrough neurons 1, 2 and 4 of six; W2 projection injects the corresponding values, and the fact is added back into the residual stream." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="kv-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="kv-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="20" width="92" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="62.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">token</text>
<text x="62.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">input</text>
<line x1="108" y1="48" x2="125" y2="48" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="126" y="20" width="130" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="191.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">W₁ projection</text>
<text x="191.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">key matching</text>
<line x1="256" y1="48" x2="273" y2="48" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="274" y="20" width="130" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="339.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">activation</text>
<text x="339.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">threshold filter</text>
<line x1="404" y1="48" x2="421" y2="48" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="422" y="20" width="122" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="483.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">W₂ projection</text>
<text x="483.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">value injection</text>
<circle cx="296" cy="94" r="6" style="fill:var(--c-surface-2);fill-opacity:1;stroke:var(--c-border);stroke-width:1"/>
<circle cx="313" cy="94" r="6" style="fill:var(--c-warn);fill-opacity:.8;stroke:var(--c-border);stroke-width:1"/>
<circle cx="330" cy="94" r="6" style="fill:var(--c-warn);fill-opacity:.8;stroke:var(--c-border);stroke-width:1"/>
<circle cx="347" cy="94" r="6" style="fill:var(--c-surface-2);fill-opacity:1;stroke:var(--c-border);stroke-width:1"/>
<circle cx="364" cy="94" r="6" style="fill:var(--c-warn);fill-opacity:.8;stroke:var(--c-border);stroke-width:1"/>
<circle cx="381" cy="94" r="6" style="fill:var(--c-surface-2);fill-opacity:1;stroke:var(--c-border);stroke-width:1"/>
<text x="339" y="116" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">our walkthrough: 3 of 6 fire</text>
<rect x="16" y="138" width="528" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1"/>
<text x="280" y="155" text-anchor="middle" style="fill:var(--c-text);font-size:12px">residual stream</text>
<line x1="62" y1="138" x2="62" y2="78" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="483" y1="78" x2="483" y2="136" marker-end="url(#kv-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="491" y="112" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px">fact added</text>
<text x="16" y="192" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Keys decide which memories wake up; values decide what they write back.</text>
</svg>

Meng et al. (2022, ROME) exploited this exact key-value structure to surgically edit factual memories in GPT models, updating $W_1$ and $W_2$ to reassign facts like changing the Eiffel Tower's location from Paris to Rome.

---

## 9. Evolution of activation functions: ReLU, GELU, and SwiGLU

### Hard thresholds of ReLU and the dying neuron crisis

Breaking out of the convex hull requires non-linear activations. Classical deep learning relied on **ReLU (Rectified Linear Unit)**:

$$\text{ReLU}(x) = \max(0, x)$$

However, ReLU suffers from two major limitations:
1. **Non-differentiability at zero:** The derivative is discontinuous at $x = 0$.
2. **The Dying ReLU Problem:** If a neuron falls into the negative regime ($x < 0$), its gradient becomes strictly zero ($\frac{\partial y}{\partial x} = 0$). Its weights stop updating, permanently killing the neuron.

### GELU and SwiGLU: From probabilistic gating to bilinear gating

**GELU (Gaussian Error Linear Unit - Hendrycks & Gimpel 2016):** First-generation modern LLMs (BERT, GPT-2, GPT-3) replaced ReLU with GELU. GELU gates inputs stochastically by multiplying by the standard normal cumulative distribution function $\Phi(x)$:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x) = x \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt$$

In practice, models use a fast analytic approximation:

$$\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} (x + 0.044715 x^3) \right) \right)$$

GELU dips slightly below zero for $x \in (-0.75, 0)$ (minimum $\approx -0.17$). This smooth non-monotonic valley keeps gradients alive for small negative inputs, providing subtle regularization.

**SwiGLU (Swish Gated Linear Unit - Shazeer 2020):** Leading open-weight models (Llama 3, Mistral, Qwen 2) employ bilinear **SwiGLU** gating in their FFNs:

$$\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \otimes x W_{\text{up}} \right) W_{\text{down}}$$

Where $\text{Swish}(z) = z \cdot \text{sigmoid}(\beta z)$. SwiGLU uses three matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). To preserve the overall parameter count, the hidden dimension is scaled to approximately $\frac{8}{3}d_{\text{model}}$ instead of $4d_{\text{model}}$. This enables dynamic, multiplicative feature gating.

---

## 10. Step 2 through the FFN: Feed-forward and second residual numerical walkthrough

### W1 projection, ReLU gating, and returning to model dimension via W2

Taking our normalized vector from Step 1:

$$h_1 = [-0.691, \; 1.617, \; -0.952, \; 0.027] \in \mathbb{R}^4$$

Setting up an FFN with hidden dimension $d_{\text{ff}} = 6$ ($W_1 \in \mathbb{R}^{6 \times 4}$, $b_1 \in \mathbb{R}^6$):

$$W_1 = \begin{bmatrix} 0.2 & -0.1 & 0.3 & 0.4 \\ -0.5 & 0.2 & 0.1 & 0.0 \\ 0.1 & 0.3 & -0.2 & 0.2 \\ 0.4 & 0.1 & 0.0 & -0.3 \\ -0.2 & 0.5 & 0.4 & 0.1 \\ 0.3 & -0.4 & 0.2 & 0.6 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0.1 \\ 0.1 \\ 0.1 \\ 0.1 \\ 0.1 \\ 0.1 \end{bmatrix}$$

**1. Projection and Bias Addition ($u = W_1 h_1 + b_1$):**
- $u_0 = 0.2(-0.691) - 0.1(1.617) + 0.3(-0.952) + 0.4(0.027) + 0.1 \approx -0.475$
- $u_1 = -0.5(-0.691) + 0.2(1.617) + 0.1(-0.952) + 0.0(0.027) + 0.1 \approx 0.674$
- $u_2 = 0.1(-0.691) + 0.3(1.617) - 0.2(-0.952) + 0.2(0.027) + 0.1 \approx 0.712$
- $u_3 = 0.4(-0.691) + 0.1(1.617) + 0.0(-0.952) - 0.3(0.027) + 0.1 \approx -0.023$
- $u_4 = -0.2(-0.691) + 0.5(1.617) + 0.4(-0.952) + 0.1(0.027) + 0.1 \approx 0.669$
- $u_5 = 0.3(-0.691) - 0.4(1.617) + 0.2(-0.952) + 0.6(0.027) + 0.1 \approx -0.928$

$$u = [-0.475, \; 0.674, \; 0.712, \; -0.023, \; 0.669, \; -0.928]$$

**2. Non-linear Activation ($a = \text{ReLU}(u)$):**

Negative components are zeroed out, letting matching keys fire:

$$a = [0.000, \; 0.674, \; 0.712, \; 0.000, \; 0.669, \; 0.000]$$

Neurons 0, 3, and 5 are suppressed; neurons 1, 2, and 4 fire. Neuron 3 is a near miss: its pre-activation is only $-0.023$, and ReLU cuts it to exactly zero.

**3. Second Projection ($f = W_2 a + b_2$):**

Given $W_2 \in \mathbb{R}^{4 \times 6}$ and $b_2 = [0.05, \; 0.05, \; 0.05, \; 0.05]^T$:

$$W_2 = \begin{bmatrix} 0.2 & 0.1 & -0.3 & 0.4 & 0.2 & -0.1 \\ -0.2 & 0.3 & 0.2 & -0.1 & 0.5 & 0.4 \\ 0.1 & -0.4 & 0.3 & 0.2 & -0.2 & 0.1 \\ 0.5 & 0.2 & -0.1 & 0.3 & 0.1 & -0.3 \end{bmatrix}$$

Multiplying $a$ by $W_2$ and adding $b_2$:
- $f[0] = 0.2(0) + 0.1(0.674) - 0.3(0.712) + 0.4(0) + 0.2(0.669) - 0.1(0) + 0.05 \approx 0.038$
- $f[1] = -0.2(0) + 0.3(0.674) + 0.2(0.712) - 0.1(0) + 0.5(0.669) + 0.4(0) + 0.05 \approx 0.729$
- $f[2] = 0.1(0) - 0.4(0.674) + 0.3(0.712) + 0.2(0) - 0.2(0.669) + 0.1(0) + 0.05 \approx -0.140$
- $f[3] = 0.5(0) + 0.2(0.674) - 0.1(0.712) + 0.3(0) + 0.1(0.669) - 0.3(0) + 0.05 \approx 0.180$

$$f = [0.038, \; 0.729, \; -0.140, \; 0.180]$$

### Second residual addition and the final block output H2

**1. Second Skip Addition ($r_2 = h_1 + f$):**

$$r_2 = [-0.691 + 0.038, \; 1.617 + 0.729, \; -0.952 - 0.140, \; 0.027 + 0.180]$$

$$r_2 = [-0.653, \; 2.346, \; -1.092, \; 0.207]$$

**2. Second Layer Normalization ($H_2 = \text{LayerNorm}(r_2)$):**

- Mean: $\mu_2 = \frac{-0.653 + 2.346 - 1.092 + 0.207}{4} = \frac{0.808}{4} = 0.202$
- Variance: $\sigma_2^2 = \frac{(-0.855)^2 + (2.144)^2 + (-1.294)^2 + (0.005)^2}{4} = \frac{0.731 + 4.597 + 1.674 + 0.000}{4} \approx 1.751$
- Standard deviation: $\sqrt{1.751} \approx 1.323$

Standardized output values:

$$H_2[0] = \frac{-0.653 - 0.202}{1.323} \approx -0.646$$

$$H_2[1] = \frac{2.346 - 0.202}{1.323} \approx 1.621$$

$$H_2[2] = \frac{-1.092 - 0.202}{1.323} \approx -0.978$$

$$H_2[3] = \frac{0.207 - 0.202}{1.323} \approx 0.004$$

$$H_2 = [-0.646, \; 1.621, \; -0.978, \; 0.004]$$

This $H_2$ vector is the completed output of the Transformer block. Token 1 (`"dog"`) gathered context through attention, preserved its identity through the skip highway, stabilized via LayerNorm, broke through the convex hull barrier via the FFN, and is ready for the subsequent block.

The whole trip of token 1 through the block, on one page:

<svg viewBox="0 0 560 420" role="img" aria-label="Token 1, dog, through one transformer block, as nine 4-number strips with the values from the text. z1 0.210, 0.820, 0.130, 0.440 plus the attention output o1 1.005, 1.041, 1.012, 0.976 gives r1 1.215, 1.861, 1.142, 1.416. LayerNorm turns it into h1 -0.691, 1.617, -0.952, 0.027. The FFN widens to six pre-activations u -0.475, 0.674, 0.712, -0.023, 0.669, -0.928; ReLU zeroes neurons 0, 3 and 5. W2 brings it back to f 0.038, 0.729, -0.140, 0.180. The second skip gives r2 -0.653, 2.346, -1.092, 0.207, and LayerNorm gives the block output H2 -0.646, 1.621, -0.978, 0.004." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="tb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="tb-arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="16" y="27" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">z₁</text>
<text x="16" y="41" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">input</text>
<rect x="150" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.18"/>
<text x="171.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.210</text>
<rect x="196" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.35"/>
<text x="217.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.820</text>
<rect x="242" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.16"/>
<text x="263.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.130</text>
<rect x="288" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.24"/>
<text x="309.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.440</text>
<text x="16" y="61" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">+ o₁</text>
<text x="16" y="75" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">attention out</text>
<rect x="150" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.40"/>
<text x="171.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.005</text>
<rect x="196" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.41"/>
<text x="217.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.041</text>
<rect x="242" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.40"/>
<text x="263.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.012</text>
<rect x="288" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.39"/>
<text x="309.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.976</text>
<text x="16" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">r₁</text>
<text x="16" y="109" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">1st skip</text>
<rect x="150" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.46"/>
<text x="171.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.215</text>
<rect x="196" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.64"/>
<text x="217.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.861</text>
<rect x="242" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.44"/>
<text x="263.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.142</text>
<rect x="288" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.52"/>
<text x="309.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.416</text>
<text x="16" y="139" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">h₁</text>
<text x="16" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">LayerNorm</text>
<rect x="150" y="128" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.31"/>
<text x="171.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.691</text>
<rect x="196" y="128" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.57"/>
<text x="217.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.617</text>
<rect x="242" y="128" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.39"/>
<text x="263.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.952</text>
<rect x="288" y="128" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.13"/>
<text x="309.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.027</text>
<text x="16" y="183" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">u</text>
<text x="16" y="197" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">W₁h₁ + b₁ (×6)</text>
<rect x="150" y="172" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.25"/>
<text x="171.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.475</text>
<rect x="196" y="172" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="217.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.674</text>
<rect x="242" y="172" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.32"/>
<text x="263.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.712</text>
<rect x="288" y="172" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.13"/>
<text x="309.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.023</text>
<rect x="334" y="172" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="355.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.669</text>
<rect x="380" y="172" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.38"/>
<text x="401.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.928</text>
<text x="16" y="217" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">ReLU</text>
<text x="16" y="231" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">3 of 6 fire</text>
<rect x="150" y="206" width="43" height="22" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1;stroke-dasharray:3 2"/>
<text x="171.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.000</text>
<rect x="196" y="206" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="217.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.674</text>
<rect x="242" y="206" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.32"/>
<text x="263.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.712</text>
<rect x="288" y="206" width="43" height="22" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1;stroke-dasharray:3 2"/>
<text x="309.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.000</text>
<rect x="334" y="206" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="355.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.669</text>
<rect x="380" y="206" width="43" height="22" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1;stroke-dasharray:3 2"/>
<text x="401.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.000</text>
<text x="16" y="261" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">f</text>
<text x="16" y="275" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">W₂a + b₂</text>
<rect x="150" y="250" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.13"/>
<text x="171.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.038</text>
<rect x="196" y="250" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.32"/>
<text x="217.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.729</text>
<rect x="242" y="250" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.16"/>
<text x="263.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.140</text>
<rect x="288" y="250" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.17"/>
<text x="309.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.180</text>
<text x="16" y="305" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">r₂ = h₁ + f</text>
<text x="16" y="319" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">2nd skip</text>
<rect x="150" y="294" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.30"/>
<text x="171.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.653</text>
<rect x="196" y="294" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.75"/>
<text x="217.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2.346</text>
<rect x="242" y="294" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.43"/>
<text x="263.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-1.092</text>
<rect x="288" y="294" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.18"/>
<text x="309.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.207</text>
<text x="16" y="349" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">H₂</text>
<text x="16" y="363" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">block output</text>
<rect x="150" y="338" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.30"/>
<text x="171.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.646</text>
<rect x="196" y="338" width="43" height="22" rx="3" style="fill:var(--c-success);fill-opacity:0.57"/>
<text x="217.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1.621</text>
<rect x="242" y="338" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.39"/>
<text x="263.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0.978</text>
<rect x="288" y="338" width="43" height="22" rx="3" style="fill:var(--c-success);fill-opacity:0.12"/>
<text x="309.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0.004</text>
<path d="M430 16 H438 V106 H430" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<text x="446" y="57.0" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px">attention</text>
<text x="446" y="72.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">+ skip</text>
<path d="M430 128 H438 V228 H430" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="446" y="174.0" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px">FFN up</text>
<text x="446" y="189.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">+ ReLU</text>
<path d="M430 250 H438 V360 H430" style="fill:none;stroke:var(--c-success);stroke-width:1.4"/>
<text x="446" y="301.0" text-anchor="start" style="fill:var(--c-success);font-size:11.5px">FFN down,</text>
<text x="446" y="316.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">skip + norm</text>
<path d="M142 27 C 116 27, 116 95, 142 95" marker-end="url(#tb-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M142 139 C 110 139, 110 305, 142 305" marker-end="url(#tb-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="16" y="384" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Red = negative. Dashed arcs are the skip paths: each sub-layer only adds a delta to what</text>
<text x="16" y="400" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">it was given. Neuron 3 misses by 0.023, so ReLU silences it.</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Post-LN layout, as in the walkthrough; Llama-style blocks normalize before each sub-layer instead.</text>
</svg>

---

## 11. Block evolution in modern production models: Llama 3, Gemma 2, and DeepSeek-V3

Production models have evolved the standard 2017 block significantly:

### Meta Llama 3: Pre-RMSNorm, SwiGLU, and zero-bias design

Meta's Llama 3 series (8B and 70B) streamlines the block architecture:
1. **Pre-RMSNorm:** Replaces LayerNorm with compute-efficient RMSNorm.
2. **SwiGLU Activation:** Employs gated SwiGLU instead of classical MLPs.
3. **Zero Bias ($\text{bias} = \text{False}$):** Eliminates bias parameters across projections, FFN layers, and normalizations, maximizing GPU Tensor Core efficiency and mitigating overfitting.

### Gemma 2 and DeepSeek-V3: Dual normalization and Attention-FFN Disaggregation (AFD)

- **Google Gemma 2 (Dual Normalization):** To ensure stable training at 27B scale, Gemma 2 adds normalization both before (pre-norm) and after (post-norm) each sub-layer (**Dual Norm**), alongside a $30.0$ soft-capping factor on attention scores to prevent logit drift.
- **DeepSeek-V3 (Attention-FFN Disaggregation - AFD & MoE):** Converts the FFN into a **Mixture of Experts (MoE)** with 1 shared expert and 256 routed experts (selecting top-8 per token). In multi-node clusters, DeepSeek pioneers **Attention-FFN Disaggregation (AFD)**, running attention and FFN workloads on specialized, decoupled hardware tiers.

---

## 12. Two engineering lenses: Training vs inference dynamics

### Memory optimization in training: Activation checkpointing

During model training, the backward pass requires activations from the forward pass to evaluate parameter gradients:
- Attention score matrices ($N \times N$)
- Normalization inputs and running moments ($\mu, \sigma$)
- Massive $4d_{\text{model}}$ FFN intermediate activations ($a$)

At long sequence lengths, storing these tensors triggers GPU Out-of-Memory (OOM) errors. **Activation Checkpointing (Gradient Checkpointing)** solves this by discarding intermediate activations, retaining only the block input $H_0$. During backprop, the forward pass is **recomputed on the fly solely for that block (rematerialization)**. Trading roughly $20-30\%$ extra compute reduces activation memory footprint by up to $70\%$.

### Computational profile in inference: Attention memory bandwidth vs FFN compute

At inference time, the two sub-layers display opposing hardware bottlenecks:

```
Attention (Decode Phase): GEMV (Matrix-Vector Multiplication) ──> Memory Bandwidth Bound
FFN Sub-layer:            GEMM (Matrix-Matrix Multiplication) ──> Compute Bound
```

- **Attention is memory-bandwidth bound:** Generating a single token requires streaming gigabytes of past Key-Value states from high-bandwidth memory (HBM) into compute registers. Arithmetic intensity (FLOP/byte) is low.
- **The FFN is compute-bound:** Each token multiplies against large static weight matrices ($W_1, W_2$). Arithmetic intensity is high, fully saturating the GPU's Tensor Cores.

---

## 13. Step-by-step verification with PyTorch

### Single transformer block module and manual tensor verification

Run this executable PyTorch script to verify our manual arithmetic across skip additions, LayerNorm statistics, and FFN projections:

```python
import torch
import torch.nn as nn

# 1. Input and Attention Output (Tensors from Section 6)
z1 = torch.tensor([0.210, 0.820, 0.130, 0.440], dtype=torch.float32)
o1 = torch.tensor([1.005, 1.041, 1.012, 0.976], dtype=torch.float32)

# 2. First Residual Addition
r1 = z1 + o1
print(f"Residual 1 (r1): {[round(x, 4) for x in r1.tolist()]}")

# 3. First LayerNorm (gamma=1, beta=0, eps=1e-5)
ln1 = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
with torch.no_grad():
    ln1.weight.copy_(torch.ones(4))
    ln1.bias.copy_(torch.zeros(4))
h1 = ln1(r1)
print(f"LayerNorm 1 (h1): {[round(x, 4) for x in h1.tolist()]}")

# 4. FFN Weights and Forward Pass (Tensors from Section 10)
W1 = torch.tensor([
    [ 0.2, -0.1,  0.3,  0.4],
    [-0.5,  0.2,  0.1,  0.0],
    [ 0.1,  0.3, -0.2,  0.2],
    [ 0.4,  0.1,  0.0, -0.3],
    [-0.2,  0.5,  0.4,  0.1],
    [ 0.3, -0.4,  0.2,  0.6]
], dtype=torch.float32)
b1 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32)

W2 = torch.tensor([
    [ 0.2,  0.1, -0.3,  0.4,  0.2, -0.1],
    [-0.2,  0.3,  0.2, -0.1,  0.5,  0.4],
    [ 0.1, -0.4,  0.3,  0.2, -0.2,  0.1],
    [ 0.5,  0.2, -0.1,  0.3,  0.1, -0.3]
], dtype=torch.float32)
b2 = torch.tensor([0.05, 0.05, 0.05, 0.05], dtype=torch.float32)

# FFN computation
u = torch.matmul(W1, h1) + b1
a = torch.relu(u)
f = torch.matmul(W2, a) + b2
print(f"FFN Output (f): {[round(x, 4) for x in f.tolist()]}")

# 5. Second Residual Addition
r2 = h1 + f
print(f"Residual 2 (r2): {[round(x, 4) for x in r2.tolist()]}")

# 6. Second LayerNorm and Final Output
ln2 = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
with torch.no_grad():
    ln2.weight.copy_(torch.ones(4))
    ln2.bias.copy_(torch.zeros(4))
h2 = ln2(r2)
print(f"Final Block Output (H2): {[round(x, 4) for x in h2.tolist()]}")
```

### Numerical output and absolute error analysis

Executing the verification script produces the following terminal output:

```text
Residual 1 (r1): [1.215, 1.861, 1.142, 1.416]
LayerNorm 1 (h1): [-0.6914, 1.6168, -0.9522, 0.0268]
FFN Output (f): [0.0376, 0.7287, -0.1397, 0.1804]
Residual 2 (r2): [-0.6538, 2.3455, -1.0919, 0.2072]
Final Block Output (H2): [-0.6467, 1.6204, -0.9778, 0.0041]
```

The absolute error between manual calculations and PyTorch's 32-bit floating-point arithmetic is below $0.001$, confirming the mathematical integrity of the block walkthrough.

---

## The whole story in six lines

- **Attention alone is insufficient:** Attention is purely a dynamic router; Softmax limits outputs to convex combinations of input Value vectors, incapable of synthesizing new logical features or world knowledge.
- **The FFN shatters the convex hull:** Expanding dimensions to $4d_{\text{model}}$ and introducing non-linear activations (GELU/SwiGLU) allows the network to synthesize novel conceptual coordinates.
- **Two-thirds of parameters live in the FFN:** As proven by Geva et al. (2021), $W_1$ keys detect conceptual patterns and $W_2$ values inject factual world knowledge into the residual stream as an associative memory.
- **Residual connections form an identity highway:** The forward $y = x + f(x)$ yields an $\mathbf{I} + f'(x)$ backward gradient; even if sub-layers saturate, signals flow through the identity matrix ($\mathbf{I}$) across 100+ layers.
- **LayerNorm/RMSNorm governs variance growth:** Residual additions cause unbounded variance accumulation; LayerNorm standardizes signals per token, avoiding BatchNorm's failure with dynamic sequence lengths.
- **A balanced dual-engine architecture:** Attention drives sequence-level communication ($N \times N$), while the FFN drives channel-level computation ($d_{\text{model}} \to 4d_{\text{model}}$); stacking these blocks powers modern AI.

---

## Glossary

- **Transformer Block:** The modular, repeating architectural unit combining attention, residual connections, normalization, and an FFN.
- **Residual Connection (Skip Connection):** A parallel path adding layer inputs directly to outputs ($y = x + f(x)$) to maintain clean gradient flow.
- **Vanishing Gradient:** The decay of backpropagated error gradients toward zero across deep architectures due to chain rule multiplication of fractional Jacobians.
- **Identity Highway ($\mathbf{I} + f'(x)$):** The algebraic property of residual derivatives ensuring error signals propagate without attenuation even if layer derivatives collapse.
- **Layer Normalization (LayerNorm):** A normalization technique standardizing features across hidden dimensions for individual tokens to zero mean and unit variance.
- **RMSNorm (Root Mean Square Normalization):** A streamlined normalization variant omitting mean-centering to reduce memory bandwidth overhead.
- **Convex Hull:** The boundary enclosing all convex combinations of a set of points; defines the geometric limit of Softmax attention outputs.
- **Feed-Forward Network (FFN / MLP):** A position-wise fully connected network that expands and contracts feature dimensions to apply non-linear transformations.
- **Associative Memory (Key-Value Memory):** A framework where $W_1$ acts as pattern-matching keys and $W_2$ acts as stored factual values within the FFN.
- **GELU (Gaussian Error Linear Unit):** A smooth, probabilistic activation function that weights inputs by the Gaussian cumulative distribution function.
- **SwiGLU:** A bilinear gated activation combining Swish and linear projections, standard in modern frontier LLMs.

---

## To dive deeper

- **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) — Foundational paper introducing residual connections and ResNets.
- **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).** *Layer Normalization.* [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) — Mathematical formulation of LayerNorm.
- **Zhang, B., & Sennrich, R. (2019).** *Root Mean Square Layer Normalization.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) — RMSNorm efficiency optimization.
- **Xiong, R., et al. (2020).** *On Layer Normalization in the Transformer Architecture.* [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) — Pre-LN vs Post-LN stability analysis.
- **Vaswani, A., et al. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — The original Transformer architecture.
- **Hendrycks, D., & Gimpel, K. (2016).** *Gaussian Error Linear Units (GELUs).* [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) — Introduction of the GELU activation function.
- **Shazeer, N. (2020).** *GLU Variants Improve Transformer.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) — SwiGLU gating architecture.
- **Geva, M., et al. (2021).** *Transformer Feed-Forward Layers Are Key-Value Memories.* [arXiv:2012.14913](https://arxiv.org/abs/2012.14913) — Proving FFNs function as associative memories.
- **Dai, D., et al. (2022).** *Knowledge Neurons in Pretrained Transformers.* [arXiv:2104.08696](https://arxiv.org/abs/2104.08696) — Identifying factual knowledge neurons in deep networks.
- **Meng, K., et al. (2022).** *Locating and Editing Factual Associations in GPT (ROME).* [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) — Surgical editing of FFN weights.
- **Meta AI (2024).** *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) — Production-scale Pre-RMSNorm and SwiGLU design.
- **Google DeepMind (2024).** *Gemma 2: Improving Open Language Models at a Practical Size.* [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) — Dual normalization and logit soft-capping.
- **DeepSeek-AI (2024).** *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — MoE and Attention-FFN Disaggregation (AFD).
- **Hochreiter, S. (1991).** *Untersuchungen zu dynamischen neuronalen Netzen.* Diploma Thesis, TU Munich — First formal identification of the vanishing gradient problem.
- **Bengio, Y., Simard, P., & Frasconi, P. (1994).** *Learning long-term dependencies with gradient descent is difficult.* IEEE TNN — Gradient attenuation in recurrent and deep architectures.
