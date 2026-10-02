When you feed a prompt into a large language model, the foundational layers explored in previous installments of our series have already transformed raw text into geometry:
1. At the [Tokenization](post.html?slug=how-tokenization-works) desk, text is partitioned into discrete integers (`[1054, 492, 281]`).
2. The [Embedding Layer](post.html?slug=inside-the-embedding-layer) retrieves continuous representation vectors from the dictionary matrix, and Rotary Position Embeddings (RoPE) prepare them for sequence-aware geometry.
3. The [Semantic Vector Space](post.html?slug=how-embeddings-work) establishes the coordinate system where concepts acquire geometric meaning and proximity.
4. The [Self-Attention mechanism](post.html?slug=inside-self-attention) provides the dynamic engine where tokens project Query, Key, and Value vectors to attend to one another and produce contextualized representations.
5. The [Causal Attention mechanism](post.html?slug=inside-causal-attention) enforces the arrow of time with a lower-triangular mask, ensuring each token attends only to past tokens and cementing the mathematical foundation of the KV cache.

Recall the concrete three-token sentence we analyzed in Parts 4 and 5: **`["dog", "cat", "chased"]`** (subject, object, and predicate). Using a single set of projection weights ($W_Q, W_K, W_V$), every token cast its Query flashlight against every Key badge, yielding a predicate-centric attention distribution that was subsequently masked to obey causality in Part 5.

Yet here lies an insurmountable limitation of single-head attention: natural language never operates on just one isolated relationship at a time. Within that same sentence, multiple independent relational axes exist concurrently:
- Subject–verb dependency ("dog" $\to$ "chased")
- Object–verb dependency ("cat" $\to$ "chased")
- Local word order and adjacency ("dog" immediately preceding "cat")
- Syntactic agreement, grammatical case, and coreferences

In single-head attention, the entire model dimension $d_{\text{model}}$ is consumed by a single bilinear similarity matrix ($M = W_Q W_K^T$). Because a single matrix can only capture one geometric orientation at a time, it is trapped in an **"uncomfortable compromise"** when forced to represent conflicting relationships simultaneously: it either focuses on the verb and blinds itself to immediate word order, or preserves local word order and fails to capture the long-range predicate.

This article, Part 6 of our series, explains how **Multi-Head Attention (MHA)** shatters this compromise trap. We explore the intuition of splitting the hidden dimension into independent geometric subspaces ($d_k = d_{\text{model}} / H$) at zero additional parameter cost, trace a complete step-by-step numerical calculation with two distinct heads operating on our concrete 3-token tensor, examine the training dynamics of symmetry breaking and positive feedback, review seminal empirical head-pruning literature (Michel, Voita, Anthropic's induction heads), explain how GPUs compute multi-head projections with a single fused GEMM and zero-copy pointer reshapes, and analyze why the severe memory-bandwidth bottleneck of the multi-head KV cache forced modern architectures to evolve toward MQA and GQA.

**In this article**

- [1. From a single projector to multiple lenses: Multi-Head intuition and the single-head bottleneck](#1-from-a-single-projector-to-multiple-lenses-multi-head-intuition-and-the-single-head-bottleneck)
- [2. The mathematics of Multi-Head Attention: Subspaces, Concat, and WO projection](#2-the-mathematics-of-multi-head-attention-subspaces-concat-and-wo-projection)
- [3. Step-by-step numerical calculation: Two heads, two distinct perspectives](#3-step-by-step-numerical-calculation-two-heads-two-distinct-perspectives)
- [4. The hardware reality: Computing Multi-Head Attention on GPUs and tensor tricks](#4-the-hardware-reality-computing-multi-head-attention-on-gpus-and-tensor-tricks)
- [5. Training dynamics: From random initialization to symmetry breaking](#5-training-dynamics-from-random-initialization-to-symmetry-breaking)
- [6. Is specialization guaranteed? Head pruning and induction heads](#6-is-specialization-guaranteed-head-pruning-and-induction-heads)
- [7. Two engineering lenses: Training vs inference and the evolution of MHA](#7-two-engineering-lenses-training-vs-inference-and-the-evolution-of-mha)
- [8. Full step-by-step verification with PyTorch](#8-full-step-by-step-verification-with-pytorch)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. From a single projector to multiple lenses: Multi-Head intuition and the single-head bottleneck

### An intuitive everyday analogy: Two fundamental attention modes

Before diving into complex mathematical formulas, let us ground these two foundational attention architectures in an intuitive analogy drawn from everyday reading.

Consider analyzing a single sentence from a printed book:  
**`"Because it rained heavily yesterday, Ahmet grabbed his umbrella."`**

When processing this sentence, an artificial intelligence system operates under one of two fundamentally distinct operational regimes:

#### 1. Normal Multi-Head Attention (Bidirectional Attention)
This mechanism is deployed when we want to place an entire text on the table at once, reading and comprehending all parts simultaneously. It resides inside the **Encoder** block of the Transformer (exemplified by **BERT** and **RoBERTa**). It is also deployed in Encoder-Decoder architectures (the original 2017 Transformer and **T5**) inside the second attention layer (**Encoder-Decoder / Cross-Attention**), where the target sequence attends bidirectionally across the source sequence.

* **How It Works:** Every word in the sentence attends simultaneously to all preceding words and all succeeding words in a single, unconstrained bidirectional view.
* **Analogy:** The entire sentence is open and visible on the page before you. When your eyes land on the word *"umbrella"*, you glance backward to see *"Ahmet"* and *"rained"*, and simultaneously glance forward to observe that the action was *"grabbed"*. You resolve all semantic connections in a single pass.
* **Objective:** Form an exhaustive, holistic understanding of grammatical relationships and contextual representation (analysis, classification, and retrieval).

#### 2. Multi-Head Causal Self-Attention (Masked / Autoregressive Attention)
This mechanism is deployed when generating new text sequentially from scratch (writing word by word). It resides in the first attention layer of the Transformer's **Decoder** block (dominating virtually all modern frontier generative models: **GPT-4**, **LLaMA 3**, **Mistral**, **Gemma**).

* **How It Works:** When the model attempts to predict the next word, it is strictly forbidden from observing future words that have not yet been produced. Future token positions are sealed beneath a rigorous mathematical mask. Each word can only attend to words to its left (the past) and itself.
* **Analogy:** Predictive text autocomplete on a smartphone keyboard. You sequentially type *"Ahmet"*, *"yesterday"*, *"because"*. While predicting the next word, the predictive model cannot peek ahead to see what you intend to write on the right. Relying solely on the prefix on the left, it predicts the most probable next word: *"rained"*.
* **Objective:** Prevent the generative model from "peeking into the future and cheating," compelling it to construct coherent, logically sound continuations step by step.

#### 3. Multi-Head Cross-Attention
Present in dual-stack architectures like the original Transformer (Vaswani 2017) and T5 inside the Decoder's second attention layer. Here, Query vectors are generated by target tokens inside the Decoder, while Key and Value vectors are fed directly from the representations produced by the Encoder. As the model emits words in the target language (e.g., German), it attends bidirectionally across the entire unmasked source sentence (e.g., English).

The table below delineates the architectural boundaries between these attention mechanisms:

| Feature | Normal Multi-Head Attention (Bidirectional) | Masked (Causal) Multi-Head Attention | Multi-Head Cross-Attention |
| :--- | :--- | :--- | :--- |
| **Attention Direction** | Bidirectional (Past and Future) | Unidirectional (Past and Present Only) | Bidirectional (Full Source Sequence) |
| **Masking** | None (Padding mask only) | Active ($-\infty$ causal mask for $j > i$) | None (Entire encoder output is unmasked) |
| **Architectural Location** | Encoder block (All layers) | Decoder block (Input / First attention layer) | Decoder block (Second / Upper attention layer) |
| **Q, K, V Source** | $Q, K, V$ from same sequence (Self-Attention) | $Q, K, V$ from same sequence (Causal Self-Attention) | $Q$ from Decoder; $K, V$ from Encoder |
| **Primary Task** | Holistic comprehension (Analysis & Representation) | Autoregressive generation (Sequential writing) | Aligning generated text with source context (Translation/Summary) |
| **Example Models** | BERT, RoBERTa, DeBERTa, ViT | GPT-4, Llama 3, Mistral, Gemma | Original Transformer (Vaswani 2017), T5, BART |

### From a single projector to multiple lenses: The theater analogy and single-head bottleneck

Why, across both operational modes, can a language model not rely on a single attention head? Why is a **"Multi-Head"** architecture strictly necessary? To grasp why, imagine a theater stage or a photographic studio.

Suppose a single massive white floodlight hangs from the ceiling. When turned on, it floods the entire stage with uniform, average white illumination. You can point this floodlight directly at the lead actor; but when you do, the background details sink into deep, harsh shadows. If you widen the beam to illuminate both the actor and the background set simultaneously, the luminous intensity drops everywhere; neither the actor's subtle facial expressions stand out, nor the background textures appear crisp. With a single light source, you cannot simultaneously cast a high-contrast portrait key light on the actor, a soft ambient fill light across the stage, and a sharp rim light to separate the silhouette from the backdrop.

A professional stage does not solve this by attempting to bend a single bulb into impossible geometries. Instead, it mounts **multiple independent spotlights**, each with its own focal angle, spatial position, and color filter:
- **Spotlight 1:** A narrow, warm key beam locked tightly onto the actor's facial expressions.
- **Spotlight 2:** A cool, diffused fill light setting the overall atmospheric tone.
- **Spotlight 3:** A sharp back-rim light carving silhouettes out from the darkness.

<svg viewBox="0 0 560 262" role="img" aria-label="Single head versus multi-head. With one head, the input Z goes through a single bilinear map M = W_Q W_K transpose and ends in a compromise: blurred, averaged attention. With multiple heads, Z feeds four heads in parallel, head 1 on the verb or predicate, head 2 on local adjacency and syntax, head 3 on syntactic agreement, head 4 on long-range coreference; their outputs are concatenated and projected by W O into a rich, multi-layered representation." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bn-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bn-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="18" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Single-Head Attention</text>
<rect x="16" y="28" width="76" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="54.0" y="53.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">Z</text>
<line x1="92" y1="49" x2="110" y2="49" marker-end="url(#bn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="112" y="28" width="200" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="212.0" y="45.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Single Bilinear Map</text>
<text x="212.0" y="61.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">M = W_Q W_Kᵀ [d × d]</text>
<line x1="312" y1="49" x2="330" y2="49" marker-end="url(#bn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="332" y="28" width="212" height="42" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="438.0" y="45.5" text-anchor="middle" style="fill:var(--c-danger);font-size:12.5px;font-weight:600">Compromise Trap (Rank Collapse)</text>
<text x="438.0" y="61.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">Blurred, averaged attention profile</text>
<text x="16" y="104" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">Multi-Head Subspaces (H Independent Maps)</text>
<rect x="16" y="160" width="76" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="54.0" y="185.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">Z</text>
<path d="M92 181 C 102 181, 102 127, 112 127" marker-end="url(#bn-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="114" y="114" width="200" height="26" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="124" y="131" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Head 1: Subject-Verb Dependency</text>
<path d="M314 127 C 322 127, 322 181, 330 181" marker-end="url(#bn-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M92 181 C 102 181, 102 159, 112 159" marker-end="url(#bn-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="114" y="146" width="200" height="26" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="124" y="163" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Head 2: Local Adjacency / n-gram</text>
<path d="M314 159 C 322 159, 322 181, 330 181" marker-end="url(#bn-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M92 181 C 102 181, 102 191, 112 191" marker-end="url(#bn-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="114" y="178" width="200" height="26" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="124" y="195" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Head 3: Direct Object / Syntax</text>
<path d="M314 191 C 322 191, 322 181, 330 181" marker-end="url(#bn-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M92 181 C 102 181, 102 223, 112 223" marker-end="url(#bn-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="114" y="210" width="200" height="26" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="124" y="227" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Head 4: Long-Range Coreference</text>
<path d="M314 223 C 322 223, 322 181, 330 181" marker-end="url(#bn-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="332" y="150" width="96" height="62" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="380.0" y="177.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Concat</text>
<text x="380.0" y="193.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">× W_O</text>
<line x1="428" y1="181" x2="442" y2="181" marker-end="url(#bn-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="440" y="150" width="108" height="62" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="494.0" y="169.5" text-anchor="middle" style="fill:var(--c-accent-2);font-size:12.5px;font-weight:600">Rich &amp; Layered</text>
<text x="494.0" y="185.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Hierarchical Output</text>
<text x="494.0" y="201.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[B, T, d_model]</text>
</svg>

Every human sentence is an equally layered stage. Consider the concurrent webs of relationships:
- **Subject–verb dependency:** Identifying who initiated the action ("dog" $\to$ "chased").
- **Object–verb dependency:** Identifying who received the action ("cat" $\to$ "chased").
- **Local word order and adjacency:** Immediate syntactic bonds between neighboring words.
- **Syntactic agreement:** Gender, number, and case concord across word groups.
- **Semantic qualification:** Linking modifiers to their target nouns ("black" $\to$ "cat").
- **Long-range coreference:** Connecting a pronoun three sentences later back to the initial noun.

As proved in Part 4, the inner product between Query and Key is governed fundamentally by this bilinear map:

$$M = W_Q W_K^T \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

This $M$ matrix defines the geometric rule of interaction between tokens. However, a single $M$ matrix can only maintain a single dominant geometric orientation in space. If optimization tunes $W_Q$ and $W_K$ to excel at predicate retrieval, the dot product $q_i^T k_j$ peaks whenever $j$ is a verb. But the moment it does so, it loses the geometric freedom to measure whether $j$ is the token sitting immediately to its left. Cramming all syntactic and semantic axes into one $M$ matrix forces it into **"an uncomfortable compromise that fits everyone poorly and satisfies no one completely."**

Multi-Head Attention resolves this by equipping the model with $H$ independent bilinear maps:

$$M^{(h)} = W_Q^{(h)} \left(W_K^{(h)}\right)^T, \quad h \in \{1, 2, \dots, H\}$$

Just as a Convolutional Neural Network (CNN) does not use a single filter for an entire image — where filter 1 detects horizontal edges, filter 2 detects corners, and filter 3 detects textures — each attention head specializes in a distinct linguistic axis within its own private subspace.

> [!NOTE]
> **The Single-Lens Budget Paradox (The 131% Dilemma):** To fully contextualize the word *"chased"*, the artificial intelligence must simultaneously attend to two distinct targets: what the action is (to its own token: at $56.1\%$) and who received the action (to the immediate direct object, *"cat"*: at $75.0\%$). However, under the mathematical rules of the Softmax function, the total attention budget allocated across all tokens is strictly capped at exactly $100\%$. Because $56.1\% + 75.0\% = 131.1\%$, a single $100\%$ budget cannot possibly represent both strong relationships at once; the scores share the same denominator and dilute each other. Multi-Head Attention is the architecture designed to grant each semantic relationship its own independent $100\%$ budget.

---

## 2. The mathematics of Multi-Head Attention: Subspaces, Concat, and WO projection

The mathematical structure of Multi-Head Attention is remarkably elegant: it partitions the computational space into $H$ parallel subspaces without expanding total tensor dimensionality.

Let the input matrix containing $N$ tokens, each represented by a $d_{\text{model}}$-dimensional vector, be $Z \in \mathbb{R}^{N \times d_{\text{model}}}$. Rather than projecting $Z$ into a single $d_{\text{model}}$ space, independent projection weights are defined for each head.

For each head $h \in \{1, 2, \dots, H\}$:

$$Q_h = Z W_Q^{(h)}, \quad K_h = Z W_K^{(h)}, \quad V_h = Z W_V^{(h)}$$

Where the projection weight dimensions are:

$$W_Q^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_K^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_V^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

In standard Transformer architectures, the subspace dimension is obtained by dividing the model dimension equally by the number of heads:

$$d_k = d_v = \frac{d_{\text{model}}}{H}$$

Each head then computes its own attention matrix and context output independently:

$$\text{head}_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h \in \mathbb{R}^{N \times d_v}$$

Finally, all head outputs are concatenated along the feature dimension:

$$\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_H) \in \mathbb{R}^{N \times (H \cdot d_v)} = \mathbb{R}^{N \times d_{\text{model}}}$$

And transformed back into the residual stream space using the learned output projection matrix $W^O$:

$$\text{MultiHead}(Z) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_H) W^O$$

$$W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

### Mathematical masking matrix: -∞ and Softmax anatomy

The masking operation at the heart of Causal Attention is a mathematically rigorous zeroing mechanism that renders future lookahead strictly impossible.

Each attention head $h$ first computes raw scaled dot-product scores ($S$) in its subspace:

$$S_{ij} = \frac{q_i k_j^T}{\sqrt{d_k}}$$

A strict lower-triangular causal masking matrix ($M$) is subsequently added element-wise:

$$S^{\text{masked}}_{ij} = S_{ij} + M_{ij}, \quad \text{where } M_{ij} = \begin{cases} 0, & j \le i \text{ (past and present)} \\ -\infty, & j > i \text{ (future)} \end{cases}$$

When the Softmax operator is applied to this masked logit matrix, an exponential transformation takes place:

$$A_{ij} = \text{softmax}(S^{\text{masked}})_{ij} = \frac{\exp(S_{ij} + M_{ij})}{\sum_{k=1}^N \exp(S_{ik} + M_{ik})}$$

The core mathematical property lies in the asymptotic limit of the exponential function at negative infinity:

$$\lim_{x \to -\infty} \exp(x) = 0$$

For any future token position ($j > i$):
1. **The numerator collapses to zero:** $\exp(S_{ij} - \infty) = 0$.
2. **Future terms vanish from the denominator:** In the denominator sum, all future terms contribute an exact zero; the normalizer thus sums strictly across past and present tokens ($k=1 \dots i$):

$$A_{ij} = \begin{cases} \frac{\exp(S_{ij})}{\sum_{k=1}^i \exp(S_{ik})}, & j \le i \\ 0, & j > i \end{cases}$$

Consequently, each attention head produces an output that is strictly a convex combination of past Value vectors:

$$\text{head}_h = A^{(h)} V_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}} + M\right) V_h \in \mathbb{R}^{N \times d_v}$$

Notice the chronological execution sequence in hardware:

<svg viewBox="0 0 560 208" role="img" aria-label="Multi-head causal attention in six steps. Input Z is projected to Q, K and V and partitioned into H heads. For each head, in parallel: 1, scores S_h = Q_h K_h transpose over square root of d_k; 2, masking, S_h plus M with minus infinity for the future; 3, softmax gives A_h; 4, context, head_h = A_h V_h. Then 5, the heads are concatenated, and 6, multiplied by W O to give the output O." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="pl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="26" width="96" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="64.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Input Z [N, d]</text>
<line x1="112" y1="50" x2="123" y2="50" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="124" y="26" width="96" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="172.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Q, K, V Proj.</text>
<text x="172.0" y="62.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">W_Q, W_K, W_V</text>
<line x1="220" y1="50" x2="231" y2="50" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="232" y="26" width="96" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Split into Heads</text>
<text x="280.0" y="62.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">[H, N, d_k]</text>
<line x1="328" y1="50" x2="339" y2="50" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="340" y="26" width="96" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="388.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">1. Scores (Sₕ)</text>
<text x="388.0" y="62.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">QₕKₕᵀ / √d_k</text>
<line x1="436" y1="50" x2="447" y2="50" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="448" y="26" width="96" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="496.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">2. Causal Mask</text>
<text x="496.0" y="62.5" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-family:var(--font-mono)">Sₕ + M (−∞)</text>
<path d="M496.0 74 V92 H64.0 V108" marker-end="url(#pl-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="110" width="96" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="64.0" y="130.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">3. Softmax</text>
<text x="64.0" y="146.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Aₕ = σ(S̃ₕ)</text>
<line x1="112" y1="134" x2="123" y2="134" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="124" y="110" width="96" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="172.0" y="130.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">4. Context</text>
<text x="172.0" y="146.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">headₕ = AₕVₕ</text>
<line x1="220" y1="134" x2="231" y2="134" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="232" y="110" width="96" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="130.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">5. Concat</text>
<text x="280.0" y="146.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">[head₁…head_H]</text>
<line x1="328" y1="134" x2="339" y2="134" marker-end="url(#pl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="340" y="110" width="96" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="388.0" y="130.5" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">6. Linear Proj.</text>
<text x="388.0" y="146.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">Concat · W_O</text>
<rect x="16" y="170" width="12" height="12" rx="2" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="34" y="180" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">H heads execute independently and concurrently (in parallel).</text>
<text x="16" y="200" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Mask is applied in every head before Softmax; heads never influence each other until Concat.</text>
</svg>

The causal masking step is inserted **strictly between scaled dot-product scores and the Softmax function**. Because $e^{-\infty} = 0$, all future token logits vanish in the exponential, ensuring that zero future information can leak into the head outputs.
> [!IMPORTANT]
> **The Gradient Shield:** When the attention weight $A_{ij} \equiv 0$ for $j > i$, the chain rule dictates that the backpropagated gradient from future tokens into past representations is strictly zero ($\frac{\partial \mathcal{L}}{\partial S_{ij}} = 0$). Without this absolute algebraic barrier, models during training would simply copy future tokens directly, suffering **shortcut collapse** and entering infinite loops during autoregressive inference.

### Why modern LLMs converged exclusively on Causal MHA

When the Transformer was first introduced (Vaswani 2017), it was a dual-stack Encoder-Decoder. In 2018, Google pursued an Encoder-only path with BERT (Normal MHA), while OpenAI pursued a Decoder-only path with GPT-1 (Causal MHA). Today, virtually all frontier production LLMs (GPT-4, Llama 3, Mistral, Claude, DeepSeek) have standardized on **Decoder-only Causal MHA**. Three architectural and hardware pillars explain this convergence:

1. **A Unified Self-Supervised Objective (Next-Token Prediction):**
   BERT's Masked Language Modeling (randomly replacing 15% of tokens with `[MASK]`) is exceptional for sentence representation and classification, but fundamentally incapable of open-ended generation. Next-token prediction under Causal MHA scales effortlessly across trillions of unstructured tokens without human annotation. To accurately predict the next token, the network is compelled to acquire world knowledge, syntactic nuance, multi-step logical deduction, and in-context learning.

2. **Inference Latency and the Invariance of the KV Cache:**
   In bidirectional attention (BERT), appending a single token at the end retroactively modifies the representations of all preceding tokens. Consequently, generating text would require **recomputing the entire sequence from scratch at every step** ($O(N^2)$ latency blowout).  
   In Causal Attention, the arrow of time cannot be reversed: token $t$ can never alter the Key or Value representations of past tokens. Those past KV tensors freeze permanently. The model never recomputes the past; it reads existing keys and values directly from the KV cache and produces each new token with a single step ($O(1)$ projection compute and $O(N)$ memory bandwidth).

3. **Hardware Parallelism During Training (Teacher Forcing):**
   While autoregressive inference is necessarily sequential, training under Causal MHA is embarrassingly parallel. Thanks to the lower-triangular mask, all $N$ tokens of a training sequence are computed simultaneously in a single fused GEMM matrix multiplication on GPUs. The model computes loss across the entire document in a single forward pass without recurrent loops.

### Reading the symbols:

- $H$: Number of attention heads (e.g., 32 in LLaMA-3 8B, 128 in LLaMA-3 70B).
- $d_{\text{model}}$: Total hidden dimension of the model (e.g., 4,096 in LLaMA-3 8B, 8,192 in 70B).
- $d_k, d_v$: Subspace dimension per head (typically 128).
- $W^O$: The master mixer matrix that projects concatenated head features back into the model dimension.

### The parameter and FLOP neutrality paradox

At first glance, running $H$ parallel heads might seem to multiply parameter count and computational complexity by $H$. In reality, Multi-Head Attention consumes **virtually zero extra parameters and zero extra projection FLOPs** compared to single-head attention.

Let us compare the parameter mathematics side-by-side:

| Component | Single-Head Attention ($d_k = d_{\text{model}}$) | Multi-Head Attention ($H$ heads, $d_k = d_{\text{model}}/H$) |
| :--- | :--- | :--- |
| **Query Projections** | $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \implies d_{\text{model}}^2$ | $H \times \left(d_{\text{model}} \times \frac{d_{\text{model}}}{H}\right) = d_{\text{model}}^2$ |
| **Key Projections** | $W_K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \implies d_{\text{model}}^2$ | $H \times \left(d_{\text{model}} \times \frac{d_{\text{model}}}{H}\right) = d_{\text{model}}^2$ |
| **Value Projections** | $W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \implies d_{\text{model}}^2$ | $H \times \left(d_{\text{model}} \times \frac{d_{\text{model}}}{H}\right) = d_{\text{model}}^2$ |
| **Output Projection $W^O$** | $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \implies d_{\text{model}}^2$ | $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \implies d_{\text{model}}^2$ |
| **Total Projection Parameters** | $\mathbf{4 d_{\text{model}}^2}$ | $\mathbf{4 d_{\text{model}}^2}$ |

The total number of projection parameters on both sides is penny-for-penny identical at $4 d_{\text{model}}^2$.

The exact same neutrality applies to computational complexity: in single-head attention, $Q K^T$ requires $N \times d_{\text{model}} \times N = N^2 d_{\text{model}}$ operations. In multi-head attention, each head performs $N^2 (d_{\text{model}} / H)$ operations across dimension $d_k = d_{\text{model}} / H$. Summing over $H$ heads:

$$H \times \left(N^2 \frac{d_{\text{model}}}{H}\right) = N^2 d_{\text{model}}$$

Multi-Head Attention does not increase the computation budget. Instead, **it reorganizes the same parameter and compute budget into multiple independent, specialized perspectives.**

---

## 3. Step-by-step numerical calculation: Two heads, two distinct perspectives

Recall the paramount engineering principle established in Part 5 ([Inside Causal Attention](post.html?slug=inside-causal-attention)): **in an autoregressive language model, looking into the future is strictly forbidden.** If an autoregressive model sees future tokens during training, its attention projections degenerate into an identity shortcut collapse, causing inference to derail into repetitive gibberish.

Therefore, in production LLMs, Multi-Head Attention is never computed using unmasked, bidirectional matrices; rather, it operates as **Multi-Head Causal Attention**, where every single head independently enforces the lower-triangular causal mask ($M$). Let us now trace the complete step-by-step manual calculation across two causal heads ($H=2$) using the exact running tensors from Parts 4 and 5.

We retain our standard configuration from Parts 4 and 5:
- Sequence length: $N = 3$ (`["dog", "cat", "chased"]` — subject, object, and verb).
- Model dimension: $d_{\text{model}} = 4$.
- Number of heads: $H = 2$.
- Subspace dimension: $d_k = d_v = d_{\text{model}} / H = 4 / 2 = 2$.

Our input matrix $Z \in \mathbb{R}^{3 \times 4}$ remains identical to earlier installments:

$$Z = \begin{bmatrix} 0.21 & 0.82 & 0.13 & 0.44 \\ 0.95 & 0.16 & 0.37 & 0.28 \\ 0.19 & 0.40 & 1.01 & 0.82 \end{bmatrix} \quad \begin{matrix} \text{token 1 (dog — subject)} \\ \text{token 2 (cat — object)} \\ \text{token 3 (chased — verb)} \end{matrix}$$

### Head 1: The predicate-centric causal head

Recall the causal attention matrix ($A^{(1)}$) obtained in Part 5 by applying the lower-triangular causal mask ($M$) to scaled dot-product scores ($S$) prior to Softmax:

$$A^{(1)} = \begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.459 & 0.541 & 0.000 \\ 0.193 & 0.246 & 0.561 \end{bmatrix}$$

Let us read the geometry of this matrix:
- **Row 1 ("dog"):** Because no prior tokens exist in the sequence, it allocates $100\%$ of its budget strictly to itself; future tokens ("cat" and "chased") are blacked out via $-\infty$ ($0.000$).
- **Row 2 ("cat"):** Balanced between the preceding subject "dog" ($45.9\%$) and itself ($54.1\%$); leakage to the future verb is strictly zero ($0.000$).
- **Row 3 ("chased"):** It now observes the entire past and concentrates the bulk of its attention ($56.1\%$) on the action itself. Head 1 has specialized as a **predicate-centric causal head**.

Let Head 1's Value projection matrix $W_V^{(1)} \in \mathbb{R}^{4 \times 2}$ be the first two columns of the $W_V$ matrix from Part 4:

$$W_V^{(1)} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \\ 0.0 & 0.0 \end{bmatrix}$$

Multiplying $V_1 = Z W_V^{(1)} \in \mathbb{R}^{3 \times 2}$:
- Token 1: $[0.21(1) + 0.13(1), \; 0.82(1) + 0.13(1)] = [0.34, 0.95]$
- Token 2: $[0.95(1) + 0.37(1), \; 0.16(1) + 0.37(1)] = [1.32, 0.53]$
- Token 3: $[0.19(1) + 1.01(1), \; 0.40(1) + 1.01(1)] = [1.20, 1.41]$

```text
V_1 Matrix:
  token 1 (dog):    [0.340, 0.950]
  token 2 (cat):    [1.320, 0.530]
  token 3 (chased): [1.200, 1.410]
```

Now we compute the causal context representation for this subspace ($\text{head}_1 = A^{(1)} V_1$):
- Row 1: $1.000 [0.34, 0.95] = [0.340, 0.950]$
- Row 2: $0.459 [0.34, 0.95] + 0.541 [1.32, 0.53] = [0.870, 0.723]$
- Row 3: $0.193 [0.34, 0.95] + 0.246 [1.32, 0.53] + 0.561 [1.20, 1.41] = [1.064, 1.105]$

$$\text{head}_1 = \begin{bmatrix} 0.340 & 0.950 \\ 0.870 & 0.723 \\ 1.064 & 1.105 \end{bmatrix}$$

> [!NOTE]
> Examine these numbers closely: this $\text{head}_1$ matrix is **strictly identical down to the third decimal place** to the first two columns of the 4D output matrix ($O$) computed in Part 5 under single-head Causal Attention!

### Head 2: The local positional adjacency head

Now consider a second, independently initialized weight triplet $(W_Q^{(2)}, W_K^{(2)}, W_V^{(2)})$. During training, this second head specialized in an entirely different linguistic role: **local adjacency (attending strictly to the immediate predecessor / $i-1$ position).**

This head must also respect causality; therefore, its upper triangle is strictly zero:

$$A^{(2)} = \begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.800 & 0.200 & 0.000 \\ 0.050 & 0.750 & 0.200 \end{bmatrix}$$

Let us analyze this matrix:
- **"dog":** Lacking any prior context, it attends exclusively to itself ($100\%$). Future positions remain masked ($0.000$).
- **"cat":** Allocates an overwhelming $80.0\%$ of its attention to its immediate preceding subject ("dog")!
- **"chased":** Unlike Head 1, it does not fixate on the verb; instead, it locks **$75.0\%$** of its attention onto the immediate direct object ("cat")!

While Head 1 captures global predicate identity, Head 2 tracks immediate left-side syntactic dependencies and bigram adjacency ($i-1$).

Let Head 2's Value matrix $W_V^{(2)} \in \mathbb{R}^{4 \times 2}$ be the last two columns of $W_V$ from Part 4:

$$W_V^{(2)} = \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \\ 0.0 & 0.0 \\ 1.0 & 1.0 \end{bmatrix}$$

Computing $V_2 = Z W_V^{(2)} \in \mathbb{R}^{3 \times 2}$:
- Token 1: $[0.82(1) + 0.44(1), \; 0.21(1) + 0.44(1)] = [1.26, 0.65]$
- Token 2: $[0.16(1) + 0.28(1), \; 0.95(1) + 0.28(1)] = [0.44, 1.23]$
- Token 3: $[0.40(1) + 0.82(1), \; 0.19(1) + 0.82(1)] = [1.22, 1.01]$

```text
V_2 Matrix:
  token 1 (dog):    [1.260, 0.650]
  token 2 (cat):    [0.440, 1.230]
  token 3 (chased): [1.220, 1.010]
```

Computing context for Head 2 ($\text{head}_2 = A^{(2)} V_2$):
- Row 1: $1.000 [1.26, 0.65] = [1.260, 0.650]$
- Row 2: $0.800 [1.26, 0.65] + 0.200 [0.44, 1.23] = [1.008 + 0.088, \; 0.520 + 0.246] = [1.096, 0.766]$
- Row 3: $0.050 [1.26, 0.65] + 0.750 [0.44, 1.23] + 0.200 [1.22, 1.01] = [0.063 + 0.330 + 0.244, \; 0.033 + 0.923 + 0.202] = [0.637, 1.157]$

$$\text{head}_2 = \begin{bmatrix} 1.260 & 0.650 \\ 1.096 & 0.766 \\ 0.637 & 1.157 \end{bmatrix}$$

### Concatenation and the WO projection

Now we concatenate these two independent causal perspectives along the feature axis:

$$\text{Concat}(\text{head}_1, \text{head}_2) = \begin{bmatrix} 0.340 & 0.950 & 1.260 & 0.650 \\ 0.870 & 0.723 & 1.096 & 0.766 \\ 1.064 & 1.105 & 0.637 & 1.157 \end{bmatrix} \in \mathbb{R}^{3 \times 4}$$

Let the output projection matrix $W^O \in \mathbb{R}^{4 \times 4}$ be:

$$W^O = \begin{bmatrix} 1.0 & 0.0 & 0.5 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.5 \\ 0.5 & 0.0 & 1.0 & 0.0 \\ 0.0 & 0.5 & 0.0 & 1.0 \end{bmatrix}$$

Multiplying the concatenated matrix by $W^O$ produces the final output tensor ($O \in \mathbb{R}^{3 \times 4}$):

$$O = \text{Concat}(\text{head}_1, \text{head}_2) W^O$$

- **Token 1 ("dog"):**
  - Col 1: $0.340(1.0) + 1.260(0.5) = 0.340 + 0.630 = 0.970$
  - Col 2: $0.950(1.0) + 0.650(0.5) = 0.950 + 0.325 = 1.275$
  - Col 3: $0.340(0.5) + 1.260(1.0) = 0.170 + 1.260 = 1.430$
  - Col 4: $0.950(0.5) + 0.650(1.0) = 0.475 + 0.650 = 1.125$
- **Token 2 ("cat"):**
  - Col 1: $0.870(1.0) + 1.096(0.5) = 0.870 + 0.548 = 1.418$
  - Col 2: $0.723(1.0) + 0.766(0.5) = 0.723 + 0.383 = 1.106$
  - Col 3: $0.870(0.5) + 1.096(1.0) = 0.435 + 1.096 = 1.531$
  - Col 4: $0.723(0.5) + 0.766(1.0) = 0.3615 + 0.766 = 1.128$
- **Token 3 ("chased"):**
  - Col 1: $1.064(1.0) + 0.637(0.5) = 1.064 + 0.3185 = 1.382$
  - Col 2: $1.105(1.0) + 1.157(0.5) = 1.105 + 0.5785 = 1.683$
  - Col 3: $1.064(0.5) + 0.637(1.0) = 0.532 + 0.637 = 1.169$
  - Col 4: $1.105(0.5) + 1.157(1.0) = 0.5525 + 1.157 = 1.709$

$$O = \begin{bmatrix} 0.970 & 1.275 & 1.430 & 1.125 \\ 1.418 & 1.106 & 1.531 & 1.128 \\ 1.382 & 1.683 & 1.169 & 1.709 \end{bmatrix}$$

### What happened numerically? (The 131% budget paradox)

Observe the final output vector for the token "chased" ($O[3] = [1.382, \; 1.683, \; 1.169, \; 1.709]$). The structural elegance of Causal Multi-Head Attention lies in how these four numbers were formed:

**1. Where did the percentage allocations come from?**

Let us inspect row 3 ("chased") in both attention matrices:

- **Head 1 ($A^{(1)}$ row 3 — Predicate focus):**

$$A^{(1)}[3, :] = [\underbrace{0.193}_{\text{dog (19.3\%)}}, \; \underbrace{0.246}_{\text{cat (24.6\%)}}, \; \underbrace{\mathbf{0.561}}_{\text{chased (56.1\%)}}]$$

Head 1 allocates **$56.1\%$** of its budget to the action itself. Weighted against $V_1$, it yields $[1.064, \; 1.105]$, informing the model: *"What action took place?"*

- **Head 2 ($A^{(2)}$ row 3 — Direct object focus):**

$$A^{(2)}[3, :] = [\underbrace{0.050}_{\text{dog (5.0\%)}}, \; \underbrace{\mathbf{0.750}}_{\text{cat (75.0\%)}}, \; \underbrace{0.200}_{\text{chased (20.0\%)}}]$$

Operating in an independent subspace, Head 2 concentrates an immense **$75.0\%$** of its budget on the direct object ("cat"). Weighted against $V_2$, it yields $[0.637, \; 1.157]$, informing the model: *"Who was the immediate direct object of this action?"*

**2. Combining drawers via Concat and $W^O$:**

Because both heads operate in independent subspaces, these two signals do not cannibalize each other:

$$\text{Concat}(\text{head}_1, \text{head}_2)[3] = [\underbrace{1.064, \; 1.105}_{\text{Action Drawer}}, \quad \underbrace{0.637, \; 1.157}_{\text{Object Drawer}}]$$

$W^O$ subsequently mixes these drawers to create the composite vector $O[3] = [1.382, \; 1.683, \; 1.169, \; 1.709]$.

Let us formalize the resolution of the single-head limitation:

**1. The 131% Paradox (The Budget Bottleneck):**

To fully contextualize "chased", the model must attend concurrently to:
- The identity of the action (itself: **$56.1\%$**).
- The direct object of the action (the immediate predecessor "cat": **$75.0\%$**).

Under single-head attention, the Softmax axiom strictly dictates that the sum of attention weights across any row must equal **exactly $100\%$ ($1.0$)**:

$$\sum_{j=1}^N A_{i, j} = 1 \quad (100\%)$$

Summing the requirements of both dependencies:

$$56.1\% + 75.0\% = \mathbf{131.1\%} > 100\%$$

A single head cannot allocate $131.1\%$ across a single $100\%$ budget!

**2. Why Single-Head Attention Collapses (Denominator Cannibalization):**

If a single head attempts to assign high logits to both targets, consider the Softmax expression for the third token:

$$A_{3, j} = \frac{e^{s_j}}{e^{s_1} + e^{s_2} + e^{s_3}}$$

When the model tries to see both the action and object strongly ($s_2 \approx 2.5$ object, $s_3 \approx 2.5$ verb, and $s_1 \approx 0.5$ subject):
- Exponentials surge: $e^{0.5} \approx 1.65$, $e^{2.5} \approx 12.18$, $e^{2.5} \approx 12.18$.
- The denominator doubles:

$$\text{Denominator} = 1.65 + \underbrace{12.18}_{\text{object}} + \underbrace{12.18}_{\text{action}} = 26.01$$

Both high scores share the same denominator, directly diluting each other's percentage (known in literature as **denominator cannibalization**).
- Both weights collapse to an uninformative compromise:

$$A_{3, 2} = \frac{12.18}{26.01} \approx \mathbf{46.8\%}, \quad A_{3, 3} = \frac{12.18}{26.01} \approx \mathbf{46.8\%}$$

Neither dependency is sharply resolved; the representation blurs.

**3. The Multi-Head Causal Attention Solution:**

- **Independent Budgets:** Each head receives its own dedicated $100\%$ budget. Head 1 dedicates **$56.1\%$** uncompromised to the action; Head 2 dedicates **$75.0\%$** uncompromised to the object ($A^{(2)}[3, 2] = 0.750$).
- **Dedicated Drawers (`Concat`):** Information is stored in separate feature slices:

$$\text{Concat}(\text{head}_1, \text{head}_2)[3] = [\underbrace{1.064, \; 1.105}_{\text{Drawer 1: Action (56.1\%)}}, \quad \underbrace{0.637, \; 1.157}_{\text{Drawer 2: Object (75.0\%)}}]$$

- **Harmonization ($W^O$):** Projections are blended into the final output vector $O[3] = [1.382, \; 1.683, \; 1.169, \; 1.709]$.
- **Zero Future Leakage:** Crucially, because both heads strictly honor the lower-triangular causal mask ($M$), zero future information has leaked into past representations.

The whole argument on one picture: one budget versus two, and where each head's result lands:

<svg viewBox="0 0 560 268" role="img" aria-label="Where the chased row&#x27;s attention budget goes. A single head that wants both the verb and its object ends up with 6.3 percent on dog and 46.8 percent each on cat and chased, a blur. Head 1 spends 19.3, 24.6 and 56.1 percent, mostly on the action. Head 2 spends 5.0, 75.0 and 20.0 percent, mostly on the object. Their outputs sit in separate drawers of the concatenated vector, 1.064 and 1.105 from head 1, 0.637 and 1.157 from head 2, and W O mixes them into 1.382, 1.683, 1.169, 1.709." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">"chased" Row Attention Budget (100%)</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Independent Softmax normalization per head</text>
<text x="16" y="70" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:600">Single Head</text>
<rect x="76.0" y="54" width="12.2" height="24" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="89.2" y="54" width="97.3" height="24" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="138.37" y="70" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">46.8%</text>
<rect x="187.5" y="54" width="97.3" height="24" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="236.65" y="70" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">46.8%</text>
<text x="16" y="110" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:600">Head 1</text>
<rect x="76.0" y="94" width="39.5" height="24" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="96.265" y="110" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">19.3%</text>
<rect x="116.5" y="94" width="50.7" height="24" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="142.36" y="110" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">24.6%</text>
<rect x="168.2" y="94" width="116.8" height="24" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="227.095" y="110" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">56.1%</text>
<text x="16" y="150" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:600">Head 2</text>
<rect x="76.0" y="134" width="9.5" height="24" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="86.5" y="134" width="156.5" height="24" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="165.25" y="150" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">75.0%</text>
<rect x="244.0" y="134" width="41.0" height="24" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="265.0" y="150" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">20.0%</text>
<rect x="76" y="178" width="11" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="92" y="188" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">dog</text>
<rect x="136" y="178" width="11" height="11" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="152" y="188" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">cat</text>
<rect x="192" y="178" width="11" height="11" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="208" y="188" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">chased</text>
<text x="16" y="214" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px">Single head: 56.1% (action) + 75.0% (object) = 131.1% needed!</text>
<text x="16" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Shared denominator flattens both to 46.8%; 2 heads provide 2 full budgets.</text>
<text x="316" y="22" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Concat → W_O Projection</text>
<text x="316" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Each head writes to its own subspace</text>
<rect x="316" y="54" width="52" height="28" rx="3" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.4"/>
<text x="342" y="73" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.064</text>
<rect x="372" y="54" width="52" height="28" rx="3" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.4"/>
<text x="398" y="73" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.105</text>
<rect x="428" y="54" width="52" height="28" rx="3" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="454" y="73" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0.637</text>
<rect x="484" y="54" width="52" height="28" rx="3" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="510" y="73" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.157</text>
<text x="370" y="98" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:600">Head 1 (Action)</text>
<text x="482" y="98" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">Head 2 (Object)</text>
<line x1="426" y1="106" x2="426" y2="136" marker-end="url(#bg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="436" y="125" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">× W_O</text>
<rect x="316" y="140" width="52" height="28" rx="3" style="fill:var(--c-text-mute);fill-opacity:.15;stroke:var(--c-border);stroke-width:1"/>
<text x="342" y="159" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.382</text>
<rect x="372" y="140" width="52" height="28" rx="3" style="fill:var(--c-text-mute);fill-opacity:.15;stroke:var(--c-border);stroke-width:1"/>
<text x="398" y="159" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.683</text>
<rect x="428" y="140" width="52" height="28" rx="3" style="fill:var(--c-text-mute);fill-opacity:.15;stroke:var(--c-border);stroke-width:1"/>
<text x="454" y="159" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.169</text>
<rect x="484" y="140" width="52" height="28" rx="3" style="fill:var(--c-text-mute);fill-opacity:.15;stroke:var(--c-border);stroke-width:1"/>
<text x="510" y="159" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.709</text>
<text x="426" y="186" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">O[2]: Two rich semantic signals unified</text>
<text x="16" y="260" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Independent budgets prevent competing dependencies from diluting a single Softmax denominator.</text>
</svg>

> [!IMPORTANT]
> **Core Takeaway:** Single-head attention forces competing semantic dependencies into a zero-sum Softmax trade-off. Multi-Head Causal Attention allocates an independent $100\%$ budget to each relational axis while strictly preserving the arrow of time, allowing modern LLMs to capture complex syntax and semantics without leaking future context.

---

## 4. The hardware reality: Computing Multi-Head Attention on GPUs and tensor tricks

Looking at the textbook formulas, one might naively implement MHA like this:

```python
# Naive theoretical loop: terrible GPU performance!
outputs = []
for h in range(num_heads):
    q_h = torch.matmul(Z, W_q[h])
    k_h = torch.matmul(Z, W_k[h])
    v_h = torch.matmul(Z, W_v[h])
    att_h = torch.softmax(torch.matmul(q_h, k_h.T) / math.sqrt(d_k), dim=-1)
    outputs.append(torch.matmul(att_h, v_h))
output = torch.matmul(torch.cat(outputs, dim=-1), W_o)
```

Running such a loop on a modern GPU is an engineering disaster. Launching $3 \times H$ tiny CUDA kernels sequentially triggers massive kernel launch overhead, starves Tensor Cores, and thrashes memory bandwidth.

In production engines (PyTorch, Hugging Face, vLLM), Multi-Head Attention is executed via **a single fused GEMM and zero-copy tensor view/transpose** manipulations:

<svg viewBox="0 0 560 250" role="img" aria-label="How a GPU computes multi-head attention. Input Z of shape B, N, d_model goes through one fused GEMM with W_qkv of shape d_model by 3 d_model, giving packed QKV of shape B, N, 3, H, d_k. It is split and permuted to B, H, N, d_k. A batched GEMM computes Q times K transpose, softmax, times V for all heads at once, giving head outputs of shape B, H, N, d_k. Transpose and view bring it back to B, N, d_model, and a final GEMM with W_o gives the output of shape B, N, d_model." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="gp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="10" width="256" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="27.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Input Z (Embedding Tensor)</text>
<text x="144.0" y="43.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[B, N, d_model]</text>
<line x1="144.0" y1="52" x2="144.0" y2="65" marker-end="url(#gp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="66" width="256" height="42" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="144.0" y="83.5" text-anchor="middle" style="fill:var(--c-accent);font-size:12.5px;font-weight:600">Single Fused GEMM (Projection)</text>
<text x="144.0" y="99.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">W_qkv: [d_model, 3·d_model]</text>
<line x1="144.0" y1="108" x2="144.0" y2="121" marker-end="url(#gp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="122" width="256" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="139.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Packed QKV Tensor</text>
<text x="144.0" y="155.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[B, N, 3, H, d_k]</text>
<line x1="144.0" y1="164" x2="144.0" y2="177" marker-end="url(#gp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="178" width="256" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="195.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Zero-Copy Permute &amp; Split</text>
<text x="144.0" y="211.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Q, K, V each: [B, H, N, d_k]</text>
<rect x="288" y="10" width="256" height="42" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416.0" y="27.5" text-anchor="middle" style="fill:var(--c-accent-2);font-size:12.5px;font-weight:600">Batched GEMM (Attention)</text>
<text x="416.0" y="43.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">σ(Q Kᵀ / √d_k + M) · V</text>
<line x1="416.0" y1="52" x2="416.0" y2="65" marker-end="url(#gp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="288" y="66" width="256" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="83.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Concurrent Head Outputs</text>
<text x="416.0" y="99.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[B, H, N, d_k]</text>
<line x1="416.0" y1="108" x2="416.0" y2="121" marker-end="url(#gp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="288" y="122" width="256" height="42" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="139.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">transpose(1, 2).contiguous().view</text>
<text x="416.0" y="155.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[B, N, H·d_k = d_model]</text>
<line x1="416.0" y1="164" x2="416.0" y2="177" marker-end="url(#gp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="288" y="178" width="256" height="42" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="416.0" y="195.5" text-anchor="middle" style="fill:var(--c-accent);font-size:12.5px;font-weight:600">Final GEMM (W_O Projection)</text>
<text x="416.0" y="211.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Output O: [B, N, d_model]</text>
<path d="M144.0 220 V230 H280 V6 H416.0 V9" marker-end="url(#gp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="244" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Two large GEMMs and one batched attention call: no Python loop over heads anywhere.</text>
</svg>

### Four critical tensor tricks in production:

1. **Fused QKV GEMM Matrix:**
   Rather than maintaining $H$ separate matrices for Query, Key, and Value, all projections are packed into one single master matrix:
   $$W_{QKV} \in \mathbb{R}^{d_{\text{model}} \times (3 \cdot d_{\text{model}})}$$
   The GPU multiplies input $Z \in \mathbb{R}^{B \times N \times d_{\text{model}}}$ in a single hardware GEMM call, saturating Tensor Cores.

2. **Zero-Copy View and Transpose:**
   No memory is allocated to split heads. The engine simply adjusts memory strides:
   $$(B, N, 3 \cdot d_{\text{model}}) \xrightarrow{\text{view}} (B, N, 3, H, d_k)$$
   Then permutes dimensions to place the head axis before sequence length:
   $$\xrightarrow{\text{permute}} (3, B, H, N, d_k)$$
   Now slices $Q, K, V$ have shape $(B, H, N, d_k)$. Attention is computed simultaneously across all heads and batches with a single batched matrix multiplication (`torch.bmm`):
   $$S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times N \times N}$$

3. **Free Concatenation via Stride Flattening:**
   After multiplying attention scores by $V$, head outputs have shape $(B, H, N, d_k)$. Recombining them requires no extra memory allocation; dimensions are permuted and flattened:
   $$\text{tensor}.\text{transpose}(1, 2).\text{contiguous}().\text{view}(B, N, d_{\text{model}})$$
   This aligns head coordinates side-by-side in memory and feeds directly into the $W^O$ projection.

4. **Zero-Overhead Causal Mask Broadcasting:**
   The lower-triangular causal mask is allocated as a single master copy of shape $(1, 1, N, N)$ in memory. PyTorch and CUDA kernels apply this mask to the $(B, H, N, N)$ attention score tensor $S$ using hardware stride broadcasting, avoiding allocating $B \times H$ duplicate mask copies across memory.

In state-of-the-art inference engines, **FlashAttention** (Dao et al., 2022; 2023) pushes this optimization further: it loads $Q, K, V$ tiles directly into fast GPU SRAM ($192\text{ KB}$ per Streaming Multiprocessor), computes softmax online, and never materializes the massive $N \times N$ attention matrix in slow HBM memory.

---

## 5. Training dynamics: From random initialization to symmetry breaking

How do Head 1 and Head 2 settle into such completely different roles? Why don't all heads converge to the exact same redundant function?

This specialization rests on one of deep learning's foundational principles: **symmetry breaking** followed by a **positive feedback** loop.

### The symmetry trap: Why random initialization is mandatory

Suppose we conducted an experiment and initialized all attention heads with identical weights:

$$W_Q^{(1)} = W_Q^{(2)} = \dots = W_Q^{(H)}$$

$$W_K^{(1)} = W_K^{(2)} = \dots = W_K^{(H)}$$

$$W_V^{(1)} = W_V^{(2)} = \dots = W_V^{(H)}$$

In this scenario, the model is trapped in mathematical symmetry:
1. Every head receives the exact same input $Z$.
2. With identical weights, they generate identical Query, Key, and Value vectors ($Q_1 = Q_2$, $K_1 = K_2$, $V_1 = V_2$).
3. They produce identical attention matrices and head outputs ($\text{head}_1 = \text{head}_2$).
4. When backpropagation flows backward through $W^O$, every head receives the exact same gradient:
   $$\frac{\partial \mathcal{L}}{\partial W_Q^{(1)}} = \frac{\partial \mathcal{L}}{\partial W_Q^{(2)}} = \dots = \frac{\partial \mathcal{L}}{\partial W_Q^{(H)}}$$
5. Gradient descent updates each head by the identical amount in the identical direction.

Because identical weights receiving identical gradients remain identical forever, the heads will never differentiate. **Independent random initialization (Xavier/He/Gaussian) is mandatory to break this symmetry from step zero.**

### Distinct computational paths and gradient divergence

Even tiny random variations ensure that each head computes a slightly different output:

$$\text{Concat}(\text{head}_1, \dots, \text{head}_H)$$

Because each head's output feeds into a different sub-block of $W^O$, their parameters influence loss through distinct computational paths:

$$\frac{\partial \mathcal{L}}{\partial W_Q^{(1)}} \neq \frac{\partial \mathcal{L}}{\partial W_Q^{(2)}}$$

### Snowball effect: Positive feedback

From this point on, gradient descent triggers a self-reinforcing dynamic:
- If Head 1's random initial state gives it a tiny accidental correlation with verbs, that direction reduces loss slightly.
- Gradient descent rewards this correlation by updating Head 1 further along that direction.
- In the next iteration, Head 1 captures verbs even more clearly, increasing gradients along that trajectory.
- Meanwhile, Head 2 is pushed toward other unexplained loss components, such as local word order or syntactic concord.

This process mirrors biological morphogenesis: microscopic fluctuations break initial symmetry, after which self-reinforcing feedback drives cells into specialized organs.

---

## 6. Is specialization guaranteed? Head pruning and induction heads

While symmetry breaking encourages specialization, gradient descent offers no mathematical guarantee that every head will learn a unique and useful role. The loss function contains no explicit "diversity bonus" penalty.

Empirical research in Transformer interpretability has unveiled two surprising truths:

### Empirical evidence: Are sixteen heads really better than one?

Seminal papers in deep learning interpretability delivered a startling finding:
- **Michel et al. (2019) — *Are Sixteen Heads Really Better than One?*:** The authors tested pruning attention heads at test time. In many Transformer layers, up to **$80\%$ of heads could be removed completely** with negligible drop in BLEU or accuracy!
- **Voita et al. (2019) — *Analyzing Multi-Head Self-Attention*:** They categorized heads into three functional groups: *syntactic heads* (tracking grammar), *positional heads* (tracking adjacent tokens), and *rare word heads*. The vast majority of remaining heads were redundant or inactive.

Why over-parameterize with so many heads if most can be pruned? Because during training, redundant heads provide multiple optimization lottery tickets; once the network finds a stable representation, only a subset does the heavy lifting.

### Anthropic's induction heads: The circuit of in-context learning

In 2022, Anthropic researchers (Olsson et al.) discovered one of the most critical functional head types in modern LLMs: **induction heads**.

An induction head executes an algorithmic copy-paste pattern:

$$\text{Context: } \dots [A][B] \dots [A] \longrightarrow \text{Predict: } [B]$$

Anthropic proved that induction heads do not operate in a single layer. They form a **two-layer circuit**:
1. **Layer 1 (Previous-Token Head):** Encodes information about the token that immediately preceded the current one ($[A] \to [B]$).
2. **Layer 2 (Induction Head):** Searches earlier context for previous occurrences of $[A]$, identifies what token followed it ($[B]$), and copies $[B]$ to the current prediction.

This two-head circuit is the mechanical engine behind few-shot prompting, in-context pattern matching, and algorithmic reasoning in modern LLMs.

---

## 7. Two engineering lenses: Training vs inference and the evolution of MHA

An architecture optimized for training throughput may face catastrophic bottlenecks during inference. Multi-Head Attention is the textbook example of this architectural divergence.

| Engineering Dimension | Training Lens | Inference Lens (Autoregressive Generation) |
| :--- | :--- | :--- |
| **Primary Memory Bottleneck** | Weights + Gradients + Optimizer States ($16\text{ B}$/param) | GPU HBM Memory **KV Cache footprint** |
| **Hardware Compute Regime** | **Compute-bound** (Large GEMMs saturate Tensor Cores) | Production stage **Memory-bandwidth-bound** (GEMV) |
| **Head Parallelism Advantage** | Full hardware parallelism across sequence length $N$ | Low arithmetic intensity ($1$ token generation per step) |
| **Communication Overhead** | Inter-GPU Tensor Parallel all-reduce ($8\text{ B}$/token) | KV cache transfer across distributed serving nodes |
| **Dominant Failure Modes** | Loss spikes, gradient collapse, dead heads | KV Cache Out-of-Memory (OOM), tail latency blowout |

### The KV Cache crisis in Multi-Head Attention

During inference, predicting each new token requires attending to all prior tokens. To avoid recomputing Key and Value projections for past tokens, their tensors are stored in GPU memory as the **KV Cache**.

In standard Multi-Head Attention, every head stores its own Key and Value tensors:

$$\text{Memory per Token} = 2 \times 2 \times n_{\text{layers}} \times d_{\text{model}} \quad \text{bytes}$$

Let us calculate this for a model of LLaMA-3 70B scale:
- $n_{\text{layers}} = 80$
- $d_{\text{model}} = 8{,}192$
- FP16 precision ($2\text{ bytes}$ per parameter)

$$\text{Memory per Token} = 2 \times 2 \times 80 \times 8{,}192 = 2{,}621{,}440 \text{ bytes} \approx 2.62 \text{ MB/token!}$$

For a concurrency of 100 users with a context of 4,096 tokens:

$$\text{Total KV Cache} = 100 \times 4{,}096 \times 2.62 \text{ MB} \approx 1{,}073 \text{ GB} = 1.07 \text{ TB of VRAM!}$$

More than a terabyte of high-bandwidth GPU memory is consumed just holding past token badges. Even worse, at each step the GPU must load this terabyte across its memory bus to compute a single token, dropping arithmetic intensity to near zero.

### The architectural evolution: From MHA to MQA, GQA, and MLA

Storing independent Key and Value vectors for every single attention head works flawlessly during training, but runs directly into the most unforgiving hardware bottleneck in modern inference: **HBM memory bandwidth.**

During the autoregressive decode phase, generation proceeds one token at a time. The underlying computational kernel is GEMV (Matrix-Vector multiplication), where arithmetic intensity is minimal ($\approx 1\text{ FLOP/byte}$). The GPU spends over 90% of its execution time stalled, waiting for massive historical KV tensors to transfer across the memory bus from HBM into ultra-fast on-chip SRAM (**memory-bandwidth-bound GEMV regime**).

To dismantle this memory wall, the AI research community engineered a four-generation architectural revolution, transitioning from standard MHA to MQA, GQA, and ultimately MLA:

<svg viewBox="0 0 560 300" role="img" aria-label="Four attention layouts, drawn with 8 query heads each. MHA gives every query head its own key-value head, ratio 32 to 32 to 32, 100 percent of the KV cache. MQA connects all query heads to a single shared key-value head, 32 to 1 to 1, 3.1 percent. GQA shares one key-value head per group of four query heads, 32 to 8 to 8, 25 percent. MLA keeps every head but caches one compressed latent vector of 512 numbers plus a 64-number RoPE key per token, about 1.8 percent of an MHA with the same 128 heads, roughly 57 times smaller." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="8" width="126" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="30" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">MHA</text>
<text x="26" y="46" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Vaswani 2017</text>
<rect x="25" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="39" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="53" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="67" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="81" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="95" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="109" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="123" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<line x1="31" y1="74" x2="31.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="25.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="45" y1="74" x2="45.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="39.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="59" y1="74" x2="59.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="53.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="73" y1="74" x2="73.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="67.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="87" y1="74" x2="87.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="81.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="101" y1="74" x2="101.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="95.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="115" y1="74" x2="115.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="109.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="129" y1="74" x2="129.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="123.0" y="112" width="12" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="25" y="154" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Q:32 · KV:32</text>
<rect x="25" y="166" width="108" height="10" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:.8"/>
<rect x="25" y="166" width="108.0" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.8"/>
<text x="26" y="204" text-anchor="start" style="fill:var(--c-text);font-size:18px;font-weight:700">100%</text>
<text x="26" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">KV cache footprint</text>
<text x="26" y="237" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPT-3, Llama 1</text>
<rect x="150" y="8" width="126" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="160" y="30" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">MQA</text>
<text x="160" y="46" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Shazeer 2019</text>
<rect x="159" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="173" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="187" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="201" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="215" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="229" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="243" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="257" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<line x1="165" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="179" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="193" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="207" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="221" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="235" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="249" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="263" y1="74" x2="214.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="201.0" y="112" width="26" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="160" y="154" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Q:32 · K:1 · V:1</text>
<rect x="159" y="166" width="108" height="10" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:.8"/>
<rect x="159" y="166" width="3.4" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.8"/>
<text x="160" y="204" text-anchor="start" style="fill:var(--c-text);font-size:18px;font-weight:700">3.1%</text>
<text x="160" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">32× smaller</text>
<text x="160" y="237" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PaLM, Falcon</text>
<rect x="284" y="8" width="126" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="294" y="30" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">GQA</text>
<text x="294" y="46" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Ainslie 2023</text>
<rect x="293" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="307" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="321" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="335" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="349" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="363" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="377" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="391" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<line x1="299" y1="74" x2="320.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="313" y1="74" x2="320.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="327" y1="74" x2="320.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="341" y1="74" x2="320.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="308.0" y="112" width="24" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<line x1="355" y1="74" x2="376.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="369" y1="74" x2="376.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="383" y1="74" x2="376.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="397" y1="74" x2="376.0" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="364.0" y="112" width="24" height="22" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="294" y="154" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Q:32 · K:8 · V:8</text>
<rect x="293" y="166" width="108" height="10" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:.8"/>
<rect x="293" y="166" width="27.0" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.8"/>
<text x="294" y="204" text-anchor="start" style="fill:var(--c-text);font-size:18px;font-weight:700">25%</text>
<text x="294" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">4× smaller</text>
<text x="294" y="237" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 2/3, Mistral</text>
<rect x="418" y="8" width="126" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="428" y="30" text-anchor="start" style="fill:var(--c-accent-2);font-size:15px;font-weight:700;letter-spacing:.06em">MLA</text>
<text x="428" y="46" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">DeepSeek 2024</text>
<rect x="427" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="441" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="455" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="469" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="483" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="497" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="511" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="525" y="58" width="12" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="427" y="112" width="110" height="22" rx="3" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<text x="482" y="127" text-anchor="middle" style="fill:var(--c-text);font-size:11px">latent c_t</text>
<line x1="433" y1="74" x2="433" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="447" y1="74" x2="447" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="461" y1="74" x2="461" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="475" y1="74" x2="475" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="489" y1="74" x2="489" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="503" y1="74" x2="503" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="517" y1="74" x2="517" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="531" y1="74" x2="531" y2="112" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="426" y="154" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">c:512 + k_R:64</text>
<rect x="427" y="166" width="108" height="10" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:.8"/>
<rect x="427" y="166" width="2.0" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.8"/>
<text x="428" y="204" text-anchor="start" style="fill:var(--c-text);font-size:18px;font-weight:700">1.8%</text>
<text x="428" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">≈57× smaller</text>
<text x="428" y="237" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DeepSeek-V2/V3</text>
<rect x="16" y="254" width="12" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="34" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Query heads (Q)</text>
<rect x="148" y="254" width="12" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="166" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Cached Key/Value (KV)</text>
<text x="16" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Simplified with 8 query heads; ratios reflect a 32-head baseline (128 heads for MLA).</text>
</svg>

#### 1. MHA (Multi-Head Attention - Vaswani et al., 2017)
* **Core Philosophy:** "Assign every Query head its own dedicated Key head and Value head."
* **Architecture:** $H$ Query heads matched symmetrically to $H$ Key heads and $H$ Value heads (a strict $1:1:1$ ratio).
* **KV Footprint Per Token:** Each layer stores $2 \times H \times d_k$ parameters:
$$\text{KV}_{\text{MHA}} = 2 \times H \times d_k$$
* **Intuitive Analogy:** A research institute with 32 specialized investigators (Query heads). Each investigator has their own private filing cabinet (Key/Value heads) inside their private office. For 32 investigators, the institute maintains 32 complete filing cabinets.
* **Strength:** Uncompromised representational expressiveness. Each head independently tracks distinct grammatical and semantic axes (subject-verb, coreferences, local syntax).
* **Fatal Flaw:** Serving a 70B model to just 100 concurrent users requires over $1\text{ TB}$ of VRAM strictly for the KV cache. Decode throughput chokes on memory bandwidth.

#### 2. MQA (Multi-Query Attention - Noam Shazeer, 2019)
* **Core Philosophy:** "Force all Query heads to share a single, unified Key and Value head."
* **Architecture:** Preserves all $H$ Query heads, but collapses Key and Value projections into **a single head** across the layer ($n_{\text{kv\_heads}} = 1$), yielding an $H:1:1$ ratio.
* **KV Footprint Per Token:** Slashes the KV Cache footprint by exactly $H\times$ (e.g., a $32\times$ or $96.9\%$ reduction in a 32-head model):
$$\text{KV}_{\text{MQA}} = 2 \times 1 \times d_k$$
* **Intuitive Analogy:** All 32 investigators continue working, but all their private filing cabinets are removed. A **single shared filing cabinet** is placed in the center hallway. All 32 investigators must query the exact same files and extract the exact same values.
* **Strength:** Drastically cuts KV memory bandwidth; generation speed and concurrent serving capacity surge dramatically (pioneered in Google PaLM, Falcon, and StarCoder).
* **The Trade-Off (Capacity Bottleneck):** Because all Query heads are forced to attend over identical Key/Value projections, model performance on multi-hop reasoning, in-context learning, and fine-grained association tasks suffers measurable degradation.

#### 3. GQA (Grouped-Query Attention - Ainslie et al., 2023)
* **Core Philosophy:** "The golden mean: Strike the optimal balance between MHA's expressive power and MQA's speed."
* **Architecture:** Partitions $H$ Query heads into $G$ groups. Each group of $H/G$ queries shares a single Key head and Value head ($H:G:G$ ratio).
* **KV Footprint Per Token:** Achieves an $H/G$-fold reduction (e.g., $4\times$ to $8\times$ smaller than MHA):
$$\text{KV}_{\text{GQA}} = 2 \times G \times d_k$$
* **Intuitive Analogy:** The 32 investigators are organized into 8 departmental teams of 4. Each 4-person team shares a single departmental filing cabinet. Instead of 32 cabinets, the facility maintains only 8.
* **Industry Standard:** Standardized by LLaMA-2/3, Mistral, and Gemma 2. Retains $>99\%$ of MHA's empirical benchmark accuracy while slashing KV cache bandwidth and VRAM by $4\times$ to $8\times$.

#### 4. MLA (Multi-Head Latent Attention - DeepSeek-V2 / V3, 2024)
* **Core Philosophy:** "Stop pruning heads! Keep all heads intact, but compress Keys and Values into a shared low-rank latent subspace."
* **Architecture:** Rather than reducing head counts, MLA maintains a massive head count ($H = 128$). Instead of caching wide Key/Value tensors, it projects hidden states into a compact latent vector ($c_t^{KV} \in \mathbb{R}^{d_c}$) via low-rank down-projection:
$$c_t^{KV} = X_t W_{DKV} \in \mathbb{R}^{d_c}, \quad \text{KV}_{\text{MLA}} = d_c + d_R$$
* **Decoupled RoPE:** Rotary Position Embeddings cannot be straightforwardly compressed into a low-rank bottleneck without corrupting spatial rotation. DeepSeek resolves this by decoupling positional geometry from content: $c_t^{KV}$ ($512$ dimensions) carries pure content, while a tiny $64$-dimensional shared decoupled RoPE key ($k_t^R$) is cached alongside it.
* **Zero Inference Overhead:** Thanks to matrix associativity ($Q (W_{UK} c_t) = (Q W_{UK}) c_t$), the up-projection matrix $W_{UK}$ can be pre-multiplied into Query projections during inference. The GPU never materializes full 128 Key heads in HBM or spends extra compute decompressing them!
* **Intuitive Analogy:** Rather than stacking bulky paper encyclopedias in cabinets, convert all text into ultra-high-density microfilms / QR codes ($c_t^{KV}$). Investigators stream microfilms into fast on-chip memory and read them in a single flash.
* **The Result:** Achieves a **$93.3\%$ to $98.2\%$ KV Cache reduction** compared to MHA (in DeepSeek-V2/V3). It consumes even less memory than MQA while preserving the full expressive capacity of 128 attention heads.

The table below summarizes the architectural characteristics across these four evolutionary generations:

| Feature | MHA (Vaswani 2017) | MQA (Shazeer 2019) | GQA (Ainslie 2023) | MLA (DeepSeek 2024) |
| :--- | :--- | :--- | :--- | :--- |
| **Head Ratio ($Q : K : V$)** | $H : H : H$ ($32 : 32 : 32$) | $H : 1 : 1$ ($32 : 1 : 1$) | $H : G : G$ ($32 : 8 : 8$) | $H : H : H$ (Low-rank compressed via $c_t^{KV}$) |
| **Memory Per Token** | $2 \times H \times d_k$ ($1\times$ baseline) | $2 \times 1 \times d_k$ ($H\times$ smaller) | $2 \times G \times d_k$ ($4\times - 8\times$ smaller) | $d_c + d_R$ ($\approx 57\times$ smaller, $98\%$ reduction) |
| **Expressive Quality** | Maximum baseline ($100\%$) | Measurable degradation in reasoning | Nearly identical to MHA ($99+\%$) | Matches or exceeds MHA baseline |
| **Hardware Bottleneck** | Memory bandwidth bound | Maximum decoding throughput | Balanced, high throughput | Ultra-low bandwidth + massive context ($128\text{K}+$) |
| **RoPE Integration** | Standard (Per-head RoPE) | Standard (Single-head RoPE) | Standard (Grouped RoPE) | Resolved via decoupled RoPE key ($k_t^R$) |
| **Flagship Models** | Original Transformer, GPT-3 | PaLM, Falcon, StarCoder | LLaMA 2/3, Mistral, Gemma 2 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 |

---

## 8. Full step-by-step verification with PyTorch

Having manually walked through the two-head Causal Multi-Head Attention arithmetic on paper in Section 3, we now construct the entire pipeline tensor-by-tensor in PyTorch. The goal of this section is to show how abstract formulas and pencil-and-paper derivations translate directly into executable code, and to observe granularly how tensor shapes, causal masking, and subspace drawers function in real memory.

We break the computation down into 6 focused steps, followed by a complete, self-contained Python script ready for single-click execution.

### Step 1: Input tensor and subspace projection matrices

We first define the input matrix $Z \in \mathbb{R}^{3 \times 4}$ representing our three tokens (`["dog", "cat", "chased"]`) with model dimension $d_{\text{model}} = 4$.

In Multi-Head Attention, the overall dimension $d_{\text{model}}$ is split evenly across the $H = 2$ heads:

$$\text{shape}(Z) = (3, 4), \quad \text{shape}(W_{V1}) = (4, 2), \quad \text{shape}(V_1) = (3, 2)$$

Each head's Value projection matrix ($W_{V1}$ and $W_{V2}$) projects the 4-dimensional representation into an independent 2-dimensional semantic subspace ($d_v = 2$). The distinct learned weights in these matrices allow each head to extract fundamentally different semantic features from the sentence:

```python
import torch

# Formatting settings: display clean floats with 3 decimal places
torch.set_printoptions(precision=3, sci_mode=False)

# 1. Input tensor Z (3 tokens, d_model = 4)
Z = torch.tensor([
    [0.21, 0.82, 0.13, 0.44],  # dog (t=0)
    [0.95, 0.16, 0.37, 0.28],  # cat (t=1)
    [0.19, 0.40, 1.01, 0.82]   # chased (t=2)
], dtype=torch.float32)

# 2. Subspace Value projection matrices (d_model = 4 -> d_v = 2)
W_V1 = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.0, 0.0]
], dtype=torch.float32)

W_V2 = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

# Compute Value tensors (V = Z @ W_V)
V1 = torch.matmul(Z, W_V1)
V2 = torch.matmul(Z, W_V2)

print("V1 (Head 1 Value Tensor - Action Space):\n", V1)
print("\nV2 (Head 2 Value Tensor - Object Space):\n", V2)
```

Console output:

```text
V1 (Head 1 Value Tensor - Action Space):
 tensor([[0.340, 0.950],
         [1.320, 0.530],
         [1.200, 1.410]])

V2 (Head 2 Value Tensor - Object Space):
 tensor([[1.260, 0.650],
         [0.440, 1.230],
         [1.220, 1.010]])
```

**What did we observe?** Each token now holds two distinct 2-dimensional identities: for example, `"dog"` is projected to $[0.340, \; 0.950]$ in Head 1's subspace, and to $[1.260, \; 0.650]$ in Head 2's subspace. These two coordinate systems never overwrite or pollute each other.

### Step 2: Raw attention scores and causal masking

Projecting Queries and Keys ($Q K^T / \sqrt{d_k}$) produces a $3 \times 3$ scaled dot-product similarity matrix ($S_1$ and $S_2$) for each head. Where do the exact numbers inside the $S_1$ and $S_2$ matrices originate? In the attention mechanism, no value is arbitrary:

1. **The Origin of Matrix $S_1$ (The Bridge to Parts 4 & 5):**
   Head 1 is the predicate-centric causal head. Input matrix $Z$ is projected using the $W_Q$ and $W_K$ weight matrices from Part 4 to yield $Q = Z W_Q$ and $K = Z W_K$. The dot products are then evaluated and scaled by $\sqrt{d_k} = \sqrt{4} = 2.0$:
   - For token 3 ("chased"), $q_2 \cdot k_0 / 2 = 1.900$ (dog), $q_2 \cdot k_1 / 2 = 2.141$ (cat), and $q_2 \cdot k_2 / 2 = 2.968$ (chased).
   - When Softmax processes these scores, $e^{2.968} = 19.453$ claims the dominant share, driving Head 1's focus to precisely $56.1\%$ on the verb itself.

2. **The Origin of Matrix $S_2$ (Logit Inversion and Object-Centric Geometry):**
   Head 2 is an independently trained subspace specializing in local adjacency and direct object resolution ($i-1$ target). The decimals in this matrix ($1.386$ and $2.708$) derive with mathematical rigor from logit inversion ($s_j - s_k = \ln(A_j / A_k)$):
   - In row 2 ("cat"), the model targets an $80\%$ to $20\%$ focus between previous subject ("dog") and self. With an odds ratio of $0.80 / 0.20 = 4$, the required logit difference is $\ln(4) \approx 1.386$ ($s_{10} = 1.386, s_{11} = 0.0$).
   - In row 3 ("chased"), the model targets $75\%$ on the immediate direct object ("cat"), $20\%$ on self ("chased"), and $5\%$ on the distant subject ("dog"). With relative odds $15 : 4 : 1$, the logit levels are $\ln(15) \approx 2.708$, $\ln(4) \approx 1.386$, and $\ln(1) = 0.000$.
   - Consequently, when $Q_2 K_2^T / \sqrt{d_k}$ computes these projections, Head 2 achieves its uncompromising $75.0\%$ direct-object focus.

However, an existential danger remains: **at this stage, the upper triangle is still exposed!** The first token `"dog"` ($t=0$) holds positive raw affinity scores ($1.389$ and $1.871$) toward future tokens `"cat"` ($t=1$) and `"chased"` ($t=2$). If we fed this matrix directly to Softmax, the model would cheat by inspecting the future.

We therefore apply our causal mask ($M$) to add $-\infty$ to all upper-triangular positions before Softmax:

$$S_{\text{masked}} = S_{\text{scaled}} + M$$

```python
# Head 1 Projection Weights (Predicate-centric head from Parts 4 & 5)
W_Q1 = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0]
], dtype=torch.float32)

W_K1 = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0]
], dtype=torch.float32)

# Head 2 Projection Weights (Direct object and local adjacency i-1 head)
W_Q2 = torch.tensor([
    [ 3.118, -2.194,  0.015, 0.0],
    [ 1.883, -0.148, -0.428, 0.0],
    [-1.270,  3.834,  2.015, 0.0],
    [-0.076,  2.463,  1.104, 0.0]
], dtype=torch.float32)

W_K2 = torch.tensor([
    [-0.015,  1.136, -0.402, 0.0],
    [ 1.238, -0.214, -0.256, 0.0],
    [-0.613, -0.016,  0.821, 0.0],
    [ 0.154, -0.139,  0.426, 0.0]
], dtype=torch.float32)

# Generate Query and Key tensors (Q = Z @ W_Q, K = Z @ W_K)
Q1 = torch.matmul(Z, W_Q1)
K1 = torch.matmul(Z, W_K1)
Q2 = torch.matmul(Z, W_Q2)
K2 = torch.matmul(Z, W_K2)

# Compute raw scores via dot-product and scale by sqrt(d_k) = 2.0
S1_scaled = torch.matmul(Q1, K1.T) / 2.0
S2_scaled = torch.matmul(Q2, K2.T) / 2.0

# Upper-triangular mask with -inf (diagonal=1: above main diagonal)
mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)

# Masking: all future-looking positions are banished to -inf
S1_masked = S1_scaled + mask
S2_masked = S2_scaled + mask

print("Masked Scores Head 1 (S1_masked):\n", S1_masked)
print("\nMasked Scores Head 2 (S2_masked):\n", S2_masked)
```

Console output:

```text
Masked Scores Head 1 (S1_masked):
 tensor([[1.339,  -inf,  -inf],
         [1.391, 1.554,  -inf],
         [1.900, 2.141, 2.968]])

Masked Scores Head 2 (S2_masked):
 tensor([[1.000,  -inf,  -inf],
         [1.386, 0.000,  -inf],
         [0.000, 2.708, 1.386]])
```

**What did we observe?** Every entry where $j > i$ is strictly converted to `-inf`. Future information channels are mathematically locked down.

### Step 3: Softmax normalization and the 131% dilemma in tensors

We pass the masked scores through the Softmax function along the final axis (`dim=-1`). Because $e^{-\infty} = 0.0$, all masked upper-triangular entries collapse to exactly $0.000$:

$$A = \text{Softmax}(S_{\text{masked}})$$

```python
# Softmax normalization: e^-inf = 0.000, row sums = 1.0 (100%)
A1 = torch.softmax(S1_masked, dim=-1)
A2 = torch.softmax(S2_masked, dim=-1)

print("Head 1 Attention Weights (A1 - Action Focused):\n", A1)
print("\nHead 2 Attention Weights (A2 - Object Focused):\n", A2)
```

Console output:

```text
Head 1 Attention Weights (A1 - Action Focused):
 tensor([[1.000, 0.000, 0.000],
         [0.459, 0.541, 0.000],
         [0.193, 0.246, 0.561]])

Head 2 Attention Weights (A2 - Object Focused):
 tensor([[1.000, 0.000, 0.000],
         [0.800, 0.200, 0.000],
         [0.050, 0.750, 0.200]])
```

**The 131% Dilemma in the Tensors:**
These two printed matrices embody the foundational thesis of Multi-Head Attention. Inspect row 3 (`"chased"`, $t=2$):
- **Head 1 ($A_1[2]$):** Allocates exactly **$56.1\%$** ($0.561$) of its attention to the action itself.
- **Head 2 ($A_2[2]$):** Allocates an enormous **$75.0\%$** ($0.750$) of its attention to the direct object (`"cat"`).
- **Why Single-Head Fails:** In a single-head layer, row probabilities must sum to $100\%$. Accommodating both demands would require $56.1\% + 75.0\% = \mathbf{131.1\%} > 100\%$, forcing an awkward compromise. By providing two independent $100\%$ budgets, Multi-Head Attention captures both syntactic axes with pristine clarity.

### Step 4: Value aggregation and head output generation (head = AV)

Each head now multiplies its attention distribution matrix ($A$) against its corresponding Value matrix ($V$):

$$\text{head}_1 = A_1 V_1, \quad \text{head}_2 = A_2 V_2$$

This operation is a standard matrix multiplication (`(3, 3) @ (3, 2) -> (3, 2)`). Each token gathers a weighted average of prior token Value vectors within its dedicated 2-dimensional subspace:

```python
# Compute independent head outputs (head = A @ V)
head1 = torch.matmul(A1, V1)
head2 = torch.matmul(A2, V2)

print("Head 1 Output (head1 - Drawer 1: Action Context):\n", head1)
print("\nHead 2 Output (head2 - Drawer 2: Object Context):\n", head2)
```

Console output:

```text
Head 1 Output (head1 - Drawer 1: Action Context):
 tensor([[0.340, 0.950],
         [0.870, 0.723],
         [1.064, 1.105]])

Head 2 Output (head2 - Drawer 2: Object Context):
 tensor([[1.260, 0.650],
         [1.096, 0.766],
         [0.637, 1.157]])
```

**What did we observe?** The tensor values match Section 3's hand-calculated results bit-for-bit:
- For token `"chased"`: Head 1 produces $[1.064, \; 1.105]$ (predicate context).
- For token `"chased"`: Head 2 produces $[0.637, \; 1.157]$ (direct object context).

### Step 5: Drawer concatenation (Concat)

To preserve both rich, independent representations without losing information, the model joins the two head outputs side-by-side along the feature dimension (`dim=-1`):

$$\text{Concat} = [\text{head}_1 \; \| \; \text{head}_2] \in \mathbb{R}^{3 \times 4}$$

```python
# Concatenate heads along feature dimension (Concat)
concat_heads = torch.cat([head1, head2], dim=-1)

print("Concatenated Heads (Concat):\n", concat_heads)
```

Console output:

```text
Concatenated Heads (Concat):
 tensor([[0.340, 0.950, 1.260, 0.650],
         [0.870, 0.723, 1.096, 0.766],
         [1.064, 1.105, 0.637, 1.157]])
```

**The Drawer Metaphor in Code:**
Every row in this concatenated tensor contains 4 elements:
- The first two columns ($[:, 0:2]$) hold the pure action information from Head 1 (`[1.064, 1.105]`).
- The last two columns ($[:, 2:4]$) hold the pure direct object information from Head 2 (`[0.637, 1.157]`).
No numbers have been averaged, overwritten, or diluted. Both drawers are placed side-by-side on the table.

### Step 6: Output projection (WO) and final multi-head representation (O)

The final step is to mix these adjacent drawers back into a single unified representation via the learned projection matrix $W^O \in \mathbb{R}^{4 \times 4}$:

$$O = \text{Concat} \times W^O \in \mathbb{R}^{3 \times 4}$$

```python
# Output projection matrix W^O (4 x 4)
W_O = torch.tensor([
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0, 0.0],
    [0.0, 0.5, 0.0, 1.0]
], dtype=torch.float32)

# Final Multi-Head Attention output (O = Concat @ W_O)
O = torch.matmul(concat_heads, W_O)

print("Final MultiHead(Z) Output (O):\n", O)
```

Console output:

```text
Final MultiHead(Z) Output (O):
 tensor([[0.970, 1.275, 1.430, 1.125],
         [1.418, 1.106, 1.531, 1.128],
         [1.382, 1.683, 1.169, 1.709]])
```

**Exact Agreement with Section 3:**
Examine the final context vector for token 3 (`"chased"`):
$$O[2] = [1.382, \quad 1.683, \quad 1.169, \quad 1.709]$$
It matches our step-by-step arithmetic from Section 3 down to the third decimal place! $W^O$ blends both the action drawer and the object drawer into a balanced, 4-dimensional contextual embedding.

### Step 7: Complete self-contained executable PyTorch script

Here is the complete, self-contained Python script uniting all steps, ready to copy and run directly in any terminal or Jupyter notebook:

```python
import torch

# Formatting settings
torch.set_printoptions(precision=3, sci_mode=False)

# 1. Input tensor Z (3 tokens, d_model = 4)
Z = torch.tensor([
    [0.21, 0.82, 0.13, 0.44],  # dog (t=0)
    [0.95, 0.16, 0.37, 0.28],  # cat (t=1)
    [0.19, 0.40, 1.01, 0.82]   # chased (t=2)
], dtype=torch.float32)

# 2. Subspace Value projection matrices (d_model = 4 -> d_v = 2)
W_V1 = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.0, 0.0]
], dtype=torch.float32)

W_V2 = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

# Value tensors (V = Z @ W_V)
V1 = torch.matmul(Z, W_V1)
V2 = torch.matmul(Z, W_V2)

# 3. Query and Key projection weights
W_Q1 = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0]
], dtype=torch.float32)

W_K1 = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0]
], dtype=torch.float32)

W_Q2 = torch.tensor([
    [ 3.118, -2.194,  0.015, 0.0],
    [ 1.883, -0.148, -0.428, 0.0],
    [-1.270,  3.834,  2.015, 0.0],
    [-0.076,  2.463,  1.104, 0.0]
], dtype=torch.float32)

W_K2 = torch.tensor([
    [-0.015,  1.136, -0.402, 0.0],
    [ 1.238, -0.214, -0.256, 0.0],
    [-0.613, -0.016,  0.821, 0.0],
    [ 0.154, -0.139,  0.426, 0.0]
], dtype=torch.float32)

# Generate Query and Key tensors (Q = Z @ W_Q, K = Z @ W_K)
Q1 = torch.matmul(Z, W_Q1)
K1 = torch.matmul(Z, W_K1)
Q2 = torch.matmul(Z, W_Q2)
K2 = torch.matmul(Z, W_K2)

# Compute raw scores via dot-product and scale by sqrt(d_k) = 2.0
S1_scaled = torch.matmul(Q1, K1.T) / 2.0
S2_scaled = torch.matmul(Q2, K2.T) / 2.0

# 4. Causal masking
mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
S1_masked = S1_scaled + mask
S2_masked = S2_scaled + mask

# 5. Softmax normalization
A1 = torch.softmax(S1_masked, dim=-1)
A2 = torch.softmax(S2_masked, dim=-1)

# 6. Head output calculation (head = A @ V)
head1 = torch.matmul(A1, V1)
head2 = torch.matmul(A2, V2)

# 7. Drawer concatenation (Concat)
concat_heads = torch.cat([head1, head2], dim=-1)

# 8. Output projection (O = Concat @ W_O)
W_O = torch.tensor([
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0, 0.0],
    [0.0, 0.5, 0.0, 1.0]
], dtype=torch.float32)

O = torch.matmul(concat_heads, W_O)

# Print verification results
print("=== Causal Multi-Head Attention Verification ===")
print("\nA1 (Head 1 Attention Weights):\n", A1)
print("\nA2 (Head 2 Attention Weights):\n", A2)
print("\nhead1 (Head 1 Output):\n", head1)
print("\nhead2 (Head 2 Output):\n", head2)
print("\nConcat (Combined Drawers):\n", concat_heads)
print("\nO (Final Multi-Head Output):\n", O)
```

Console output:

```text
=== Causal Multi-Head Attention Verification ===

A1 (Head 1 Attention Weights):
 tensor([[1.000, 0.000, 0.000],
         [0.459, 0.541, 0.000],
         [0.193, 0.246, 0.561]])

A2 (Head 2 Attention Weights):
 tensor([[1.000, 0.000, 0.000],
         [0.800, 0.200, 0.000],
         [0.050, 0.750, 0.200]])

head1 (Head 1 Output):
 tensor([[0.340, 0.950],
         [0.870, 0.723],
         [1.064, 1.105]])

head2 (Head 2 Output):
 tensor([[1.260, 0.650],
         [1.096, 0.766],
         [0.637, 1.157]])

Concat (Combined Drawers):
 tensor([[0.340, 0.950, 1.260, 0.650],
         [0.870, 0.723, 1.096, 0.766],
         [1.064, 1.105, 0.637, 1.157]])

O (Final Multi-Head Output):
 tensor([[0.970, 1.275, 1.430, 1.125],
         [1.418, 1.106, 1.531, 1.128],
         [1.382, 1.683, 1.169, 1.709]])
```

---

## The whole story in six lines

- **The Single-Head Bottleneck:** A single $W_Q W_K^T$ matrix can only point along one geometric direction, forcing conflicting linguistic relationships into an averaged compromise.
- **Subspace Partitioning:** Multi-Head Attention divides the model dimension into $H$ parallel subspaces ($d_k = d_{\text{model}} / H$) without adding parameters or computational cost.
- **The 131% Paradox:** A single attention row has a $100\%$ softmax budget; when a verb demands $56.1\%$ on itself and $75.0\%$ on its object, a single head fails ($131.1\% > 100\%$), whereas MHA grants two independent $100\%$ budgets.
- **Hardware Execution:** Production engines never use loops; they compute all heads with a single fused GEMM and zero-copy pointer views/transposes.
- **Symmetry Breaking:** Independent random initialization is mandatory to break mathematical symmetry so gradient descent can specialize heads.
- **Inference Evolution:** Because multi-head KV caches consume terabytes of VRAM during inference, modern architectures transitioned to GQA and MLA.

---

## Glossary

- **Multi-Head Attention (MHA):** Attention mechanism that computes multiple attention representations in parallel subspaces ($d_k = d_{\text{model}} / H$) and concatenates them.
- **Single-Head Attention:** An attention layer that computes a single Query, Key, and Value vector using the entire model dimension.
- **Subspace ($d_k$):** The reduced dimensionality ($d_{\text{model}} / H$) in which an individual attention head operates.
- **Concatenation (Concat):** Joining vectors side-by-side along the feature dimension without altering their coordinates.
- **Output Projection ($W^O$):** The learned matrix that mixes concatenated subspace outputs back into the residual stream dimension.
- **Symmetry Breaking:** Microscopic initial differences that prevent identical parameter updates across parallel heads.
- **Fused GEMM:** Combining multiple matrix multiplications into a single hardware kernel call on the GPU.
- **KV Cache:** Storing past Key and Value activations in GPU VRAM to avoid recomputing them during autoregressive generation.
- **Head Pruning:** Removing uninformative or redundant attention heads at inference time without retraining.
- **Induction Head:** A multi-layer attention circuit ($[A][B] \dots [A] \to [B]$) that enables in-context pattern matching and few-shot reasoning.
- **Multi-Query Attention (MQA):** An architecture using multiple Query heads but sharing a single Key and Value head to minimize KV cache.
- **Grouped-Query Attention (GQA):** An architecture grouping Query heads to share Key and Value heads in clusters.
- **Multi-Head Latent Attention (MLA):** DeepSeek's low-rank compressed attention architecture that compresses the KV cache by over $90\%$.

---

## Going deeper

- Vaswani et al. (2017) — *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*: The seminal paper introducing the Transformer and Multi-Head Attention.
- Michel et al. (2019) — *[Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)*: Empirical proof of attention head redundancy and pruning.
- Voita et al. (2019) — *[Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting](https://arxiv.org/abs/1905.09418)*: Analysis of syntactic, positional, and rare-word heads.
- Olsson et al. / Anthropic (2022) — *[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)*: Discovery and mechanistic analysis of the induction head circuit.
- Shazeer (2019) — *[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)*: Introduction of Multi-Query Attention (MQA).
- Ainslie et al. (2023) — *[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)*: Introduction of Grouped-Query Attention (GQA).
- DeepSeek-AI (2024) — *[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)*: Technical details of Multi-Head Latent Attention (MLA).
- On this blog: [A Prompt's Journey (1): Tokenization](post.html?slug=how-tokenization-works), [A Prompt's Journey (2): The Embedding Layer](post.html?slug=inside-the-embedding-layer), [A Prompt's Journey (3): Semantic Vectors](post.html?slug=how-embeddings-work), [A Prompt's Journey (4): Inside Self-Attention](post.html?slug=inside-self-attention), and [A Prompt's Journey (5): Inside Causal Attention](post.html?slug=inside-causal-attention).
