Imagine designing a state-of-the-art deep neural network with 100 stacked dense layers, allocating billions of floating-point parameters across high-bandwidth GPU memory. You compute the forward pass, orchestrate parallel tensor contractions across thousands of CUDA cores, and train on petabytes of data. Yet, regardless of training duration, the model fails to learn anything beyond a flat, linear decision boundary—collapsing entirely when tasked with predicting something as elementary as a 2-bit XOR gate.

The underlying failure is neither a software bug nor numerical divergence. It is fundamental linear algebra: matrix multiplication is associative. Without a non-linear activation function slotted between layers, every sequence of successive matrix projections $W_L \dots W_2 W_1 x$ collapses mathematically into a single composite matrix $W_{\text{net}} x$. Your 100-layer deep network is functionally identical to a single-layer linear regression model, incapable of bending, folding, or carving high-dimensional geometric representations.

Activation functions are the sole mathematical component that grants neural networks the capacity to act as universal function approximators. Yet, for decades, the choice of activation plagued deep learning with vanishing gradients, dead neurons, and training instability. This article examines the geometric necessity of non-linear activations, traces their evolution from the saturating sigmoids of early connectionism to the probabilistic gating of GELU, analyzes why modern large language models (LLMs) like LLaMA 3 and DeepSeek-V3 universally rely on SwiGLU, and walks through exact manual tensor arithmetic verified with runnable code.

**In this article**

- [1. The core intuition: Why linearity collapses without activation](#1-the-core-intuition-why-linearity-collapses-without-activation)
- [2. The classical era: Sigmoid, Tanh, and the calculus of saturation](#2-the-classical-era-sigmoid-tanh-and-the-calculus-of-saturation)
- [3. The deep learning renaissance: ReLU and the dying neuron pathology](#3-the-deep-learning-renaissance-relu-and-the-dying-neuron-pathology)
- [4. The probabilistic gating paradigm: GELU](#4-the-probabilistic-gating-paradigm-gelu)
- [5. The modern LLM standard: Gated Linear Units and SwiGLU](#5-the-modern-llm-standard-gated-linear-units-and-swiglu)
- [6. Step-by-step manual tensor walkthrough](#6-step-by-step-manual-tensor-walkthrough)
- [7. Two engineering lenses: Training vs inference dynamics](#7-two-engineering-lenses-training-vs-inference-dynamics)
- [8. Python and PyTorch verification](#8-python-and-pytorch-verification)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [To dive deeper](#to-dive-deeper)

---

## 1. The core intuition: Why linearity collapses without activation

### The Origami Principle of Representation

To understand what an activation function does geometrically, consider a flat sheet of paper. 

If you are only permitted to rotate, translate, stretch, or shear the paper—the geometric equivalents of affine transformations $W x + b$—any two points that lie on opposite sides of a straight line can never be brought together without moving all other collinear points along with them. You cannot isolate a central region from its surrounding perimeter using only linear cuts.

Non-linear activation functions provide the **fold**. By bending, clipping, or twisting the continuous coordinate space, non-linearities allow the network to fold distant regions of the input space into close proximity or carve complex non-convex decision boundaries.

<svg viewBox="0 0 560 300" role="img" aria-label="Why depth needs non-linearity. Left, linear collapse: input x passes through layer 1 with W1, layer 2 with W2, up to layer L with W_L; by associativity the output is (W_L ... W1) x, which collapses into a single matrix W_net, rank-bounded by the bottleneck layer. Right, non-linear folding: affine transformations are interleaved with activation folds sigma; by universal approximation, the network can model arbitrary non-convex manifolds." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="af-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="af-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="8" width="256" height="284" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="30" text-anchor="start" style="fill:var(--c-danger);font-size:13.5px;font-weight:700">Linear collapse (Rank bounded)</text>
<rect x="40" y="44" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="62.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">input: x ∈ ℝᵈ</text>
<line x1="144" y1="72" x2="144" y2="81" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="82" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">layer 1: W₁x</text>
<line x1="144" y1="110" x2="144" y2="119" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="120" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="138.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">layer 2: W₂h₁</text>
<line x1="144" y1="148" x2="144" y2="157" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="158" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="176.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">layer L: W_L h_{L-1}</text>
<line x1="144" y1="186" x2="144" y2="236" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="30" y="238" width="228" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="144" y="256" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">(W_L … W₁) x = W_net x</text>
<text x="144" y="273" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">associativity: collapses to 1 layer</text>
<rect x="288" y="8" width="256" height="284" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="302" y="30" text-anchor="start" style="fill:var(--c-accent-2);font-size:15px;font-weight:700;letter-spacing:.06em">Non-linear folding</text>
<rect x="312" y="44" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="62.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">input: x ∈ ℝᵈ</text>
<line x1="416" y1="72" x2="416" y2="81" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="82" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">affine: z₁ = W₁x + b₁</text>
<line x1="416" y1="110" x2="416" y2="119" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="120" width="208" height="28" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416.0" y="138.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">fold: h₁ = σ(z₁)</text>
<line x1="416" y1="148" x2="416" y2="157" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="158" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="176.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">affine: z₂ = W₂h₁ + b₂</text>
<line x1="416" y1="186" x2="416" y2="195" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="196" width="208" height="28" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416.0" y="214.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">fold: h₂ = σ(z₂)</text>
<line x1="416" y1="224" x2="416" y2="236" marker-end="url(#af-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="302" y="238" width="228" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="256" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">universal approximation manifold</text>
<text x="416" y="273" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">space bends: non-convex boundaries</text>
</svg>

### The Matrix Collapse Proof

Mathematically, let an $L$-layer feed-forward network without activation functions be defined as:

$$h_1 = W_1 x + b_1$$

$$h_2 = W_2 h_1 + b_2 = W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$$

By induction across all $L$ layers:

$$y = W_L (W_{L-1} \dots (W_1 x + b_1) \dots + b_{L-1}) + b_L = W_{\text{net}} x + b_{\text{net}}$$

Where:

$$W_{\text{net}} = \prod_{l=L}^{1} W_l, \quad b_{\text{net}} = b_L + \sum_{i=1}^{L-1} \left( \prod_{j=L}^{i+1} W_j \right) b_i$$

Because the set of linear transformations is closed under composition, $W_{\text{net}}$ is simply another single matrix of dimensions $d_{\text{out}} \times d_{\text{in}}$. Furthermore, by the rank inequality:

$$\text{rank}(W_{\text{net}}) \le \min \left( \text{rank}(W_1), \text{rank}(W_2), \dots, \text{rank}(W_L) \right)$$

No matter how wide or deep the intermediate hidden layers are, intermediate representations cannot expand the expressive rank beyond the bottleneck dimension. More critically, linear mappings cannot compute the logical XOR function, let alone parse syntax, recognize visual hierarchies, or reason over linguistic semantics.

---

## 2. The classical era: Sigmoid, Tanh, and the calculus of saturation

During the initial rise of artificial neural networks, biological plausibility dominated architectural decisions. Biological neurons fire action potentials when membrane voltage crosses a threshold, motivating smooth, S-shaped continuous squashing functions.

### The Logistic Sigmoid and Gradient Decay

The standard logistic sigmoid maps the real number line $(-\infty, \infty)$ strictly into the open probability interval $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Its first derivative exhibits an elegant algebraic property:

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z) \left( 1 - \sigma(z) \right)$$

Evaluating the derivative reveals an immediate architectural vulnerability:
* When $z = 0$, $\sigma(0) = 0.5$, and $\sigma'(0) = 0.5 \times 0.5 = 0.25$.
* As $|z| \to \infty$, $\sigma(z) \to 0$ or $1$, causing $\sigma'(z) \to 0$.

The maximum possible derivative of the sigmoid function is strictly **$0.25$**. In an $L$-layer network, the backpropagation chain rule requires multiplying the local Jacobians backwards:

$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial z_L} \prod_{l=1}^{L-1} \left( W_{l+1}^T \text{diag}(\sigma'(z_l)) \right)$$

Even if the weight matrices are initialized with spectral norm near $1$, the upper bound on the gradient propagation through $L$ activation layers shrinks exponentially:

$$\left\| \prod_{l=1}^{L} \sigma'(z_l) \right\| \le (0.25)^L$$

Across a modest 10-layer network, $(0.25)^{10} \approx 9.53 \times 10^{-7}$. The error gradient transmitted to the earliest layers is diminished by a factor of nearly one million. As a consequence, weights in early layers receive virtually zero updates, freezing the network in its random initial state.

### The Non-Zero-Centered Dilemma

Sigmoid suffers from a secondary structural flaw: its outputs are strictly positive ($\sigma(z) > 0$). 

For any hidden neuron whose output serves as input to layer $l+1$, $h^{(l)} > 0$. The gradient of the loss with respect to weight vector $w_i^{(l+1)}$ is:

$$\frac{\partial \mathcal{L}}{\partial w_i^{(l+1)}} = \frac{\partial \mathcal{L}}{\partial z_i^{(l+1)}} h^{(l)}$$

Because $h^{(l)}$ is strictly positive, all elements of $\frac{\partial \mathcal{L}}{\partial w_i^{(l+1)}}$ must share the identical sign as the upstream scalar derivative $\frac{\partial \mathcal{L}}{\partial z_i^{(l+1)}}$. The gradient vector is constrained to point exclusively into all-positive or all-negative orthants, forcing gradient descent into inefficient, zig-zagging trajectories during parameter optimization.

### Hyperbolic Tangent (Tanh)

To eliminate the non-zero-centered bias, practitioners turned to the hyperbolic tangent function ($\tanh$), which maps $(-\infty, \infty)$ symmetrically to $(-1, 1)$:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

Its derivative is:

$$\tanh'(z) = 1 - \tanh^2(z)$$

While $\tanh'(0) = 1.0$ (alleviating the $0.25$ ceiling of sigmoid at the origin), $\tanh(z)$ still saturates rapidly for $|z| > 2$. Once a neuron's pre-activation drifts into the saturated tails ($|z| \ge 3 \implies \tanh'(z) \le 0.01$), the local gradient extinguishes, rendering deep networks fragile during pre-training.

| Activation | Mathematical Form | Output Range | Max Derivative | Zero-Centered? | Saturation Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $(0, 1)$ | $0.25$ ($z=0$) | No | Severe ($\vert z \vert > 3$) |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | $1.00$ ($z=0$) | Yes | Severe ($\vert z \vert > 2$) |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | $1.00$ ($z>0$) | No | None for $z>0$; Absolute for $z<0$ |
| **GELU** | $z \Phi(z)$ | $[-0.17, \infty)$ | $\approx 1.12$ ($z=0.75$) | Near-zero | Minimal (smooth tail) |
| **SiLU (Swish)** | $z \sigma(z)$ | $[-0.278, \infty)$ | $\approx 1.10$ ($z=1.28$) | Near-zero | Minimal (smooth tail) |

---

## 3. The deep learning renaissance: ReLU and the dying neuron pathology

In 2010, Vinod Nair and Geoffrey Hinton introduced the Rectified Linear Unit (ReLU), an architectural pivot that made the deep learning revolution of the 2010s computationally viable:

$$\text{ReLU}(z) = \max(0, z)$$

Its derivative is piecewise constant:

$$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$

### Why ReLU Unlocked Deep Networks

1. **Non-Saturating Gradient Highway:** For any positive activation ($z > 0$), the local derivative is exactly $1.0$. Gradients flow backward across dozens of layers without geometric damping or exponentiation decay.
2. **Computational Frugality:** Evaluating $\max(0, z)$ requires no transcendental floating-point operations ($e^z$, division). On hardware, modern ALUs compute ReLU via a single binary branch or SIMD bitwise mask (`z & ~(z >> 31)`), consuming a single clock cycle.
3. **Biological Sparsity:** In biological brains, only 1–4% of neurons fire simultaneously. ReLU naturally yields sparse representations: negative features produce an exact zero, clearing memory buffers and eliminating unnecessary feature interactions.

### The "Dying ReLU" Pathology

Despite its elegance, ReLU possesses an operational fatal flaw: **directional irreversibility**.

Suppose a neuron with weights $w$ and bias $b$ receives an unusually large gradient update during backpropagation—often caused by a transient learning rate spike or a high-variance mini-batch. The updated parameter vector $w_{\text{new}}$ shifts the operating point such that:

$$w_{\text{new}}^T x + b_{\text{new}} \le 0 \quad \forall x \in \mathcal{D}_{\text{train}}$$

Because the pre-activation is now negative across the entire data manifold, the output is identically $0$. Consequently, the backward derivative is identically $0$:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \cdot x = 0 \cdot x = \mathbf{0}$$

No gradient can ever flow into this neuron again. Its weights can never be updated, its bias can never change, and it becomes permanently inactive: a **dead neuron**. In poorly initialized or aggressively tuned deep networks, 10% to 40% of all ReLU neurons can die permanently before the first training epoch concludes, permanently burning parameter capacity.

### Structural Remedies: Leaky ReLU and PReLU

To prevent complete gradient cutoff, Andrew Maas et al. (2013) proposed **Leaky ReLU**:

$$\text{LeakyReLU}(z) = \max(\alpha z, z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \le 0 \end{cases}$$

Where $\alpha$ is a fixed small constant, typically $\alpha = 0.01$. The backward gradient for negative activations becomes $\alpha$, guaranteeing that even inactive neurons retain a non-zero recovery pathway. Kaiming He et al. (2015) parameterized this slope into **Parametric ReLU (PReLU)**, treating $\alpha$ as a learnable parameter optimized via backpropagation.

---

## 4. The probabilistic gating paradigm: GELU

While ReLU solved vanishing gradients and LeakyReLU patched dead neurons, both functions remain piece-wise linear approximations with an abrupt, non-differentiable singularity at $z = 0$.

In 2016, Dan Hendrycks and Kevin Gimpel introduced the **Gaussian Error Linear Unit (GELU)**, reframing neuron activation from a deterministic threshold into a **stochastic gating mechanism**.

### The Conceptual Leap: Stochastic Regularization Meets Non-Linearity

In modern architectures, two operations were traditionally decoupled:
1. **Deterministic Non-Linearity:** ReLU ($\max(0, x)$) zeros out negative inputs.
2. **Stochastic Regularization:** Dropout randomly zeros out activations with probability $p$.

GELU unifies these concepts. Rather than dropping inputs uniformly at random or gating them based on a rigid binary rule, GELU scales an input $x$ by the probability that a standard Gaussian variable $X \sim \mathcal{N}(0, 1)$ does not exceed $x$:

$$\text{GELU}(x) = x \cdot P(X \le x) = x \cdot \Phi(x)$$

Where $\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution:

$$\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-\frac{t^2}{2}} dt = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$

Here, $\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt$ represents the Gauss error function.

<svg viewBox="0 0 560 196" role="img" aria-label="GELU as a gate. The input x is scaled by the probability that a standard normal variable is at most x, producing y = x · Phi(x). The Gaussian CDF Phi rises smoothly from 0 to 1: large negative inputs have their gradients suppressed, while positive inputs pass through via an identity mapping." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="ge-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="20" width="96" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="64.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">input: x</text>
<rect x="150" y="14" width="260" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="38.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Stochastic Gaussian gate: x · Φ(x)</text>
<text x="280.0" y="54.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Φ(x) = P(X ≤ x),  X ~ 𝒩(0, 1)</text>
<rect x="448" y="20" width="96" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="496.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">y = GELU(x)</text>
<line x1="112" y1="42" x2="148" y2="42" marker-end="url(#ge-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="410" y1="42" x2="446" y2="42" marker-end="url(#ge-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="180" y="94" width="200" height="98" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="280" y1="94" x2="280" y2="72" marker-end="url(#ge-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M190 170 H370" style="stroke:var(--c-border);stroke-width:1"/>
<path d="M190.0 169.9 L193.0 169.9 L196.0 169.8 L199.0 169.8 L202.0 169.7 L205.0 169.6 L208.0 169.5 L211.0 169.4 L214.0 169.2 L217.0 168.9 L220.0 168.6 L223.0 168.3 L226.0 167.8 L229.0 167.3 L232.0 166.7 L235.0 166.0 L238.0 165.2 L241.0 164.2 L244.0 163.1 L247.0 161.9 L250.0 160.5 L253.0 159.0 L256.0 157.3 L259.0 155.5 L262.0 153.5 L265.0 151.5 L268.0 149.3 L271.0 147.1 L274.0 144.8 L277.0 142.4 L280.0 140.0 L283.0 137.6 L286.0 135.2 L289.0 132.9 L292.0 130.7 L295.0 128.5 L298.0 126.5 L301.0 124.5 L304.0 122.7 L307.0 121.0 L310.0 119.5 L313.0 118.1 L316.0 116.9 L319.0 115.8 L322.0 114.8 L325.0 114.0 L328.0 113.3 L331.0 112.7 L334.0 112.2 L337.0 111.7 L340.0 111.4 L343.0 111.1 L346.0 110.8 L349.0 110.6 L352.0 110.5 L355.0 110.4 L358.0 110.3 L361.0 110.2 L364.0 110.2 L367.0 110.1 L370.0 110.1" style="fill:none;stroke:var(--c-accent);stroke-width:2"/>
<text x="194" y="114" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Φ(x)</text>
<text x="370" y="186" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">Gaussian CDF: x ∈ [-3, 3]</text>
<text x="16" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">x ≪ 0: Φ(x) → 0</text>
<text x="16" y="135" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradient damped, gate shut</text>
<text x="544" y="120" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">x ≫ 0: Φ(x) → 1</text>
<text x="544" y="135" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">identity: GELU(x) ≈ x</text>
</svg>

Geometrically, this confers profound advantages:
* When $x$ is large and positive ($x > 2$), $\Phi(x) \to 1$, and $\text{GELU}(x) \to x$ (approaching the linear identity of ReLU).
* When $x$ is large and negative ($x < -2$), $\Phi(x) \to 0$, and $\text{GELU}(x) \to 0$ (approaching zero like ReLU).
* In the critical transitional zone around zero, GELU is **smooth, non-monotonic, and infinitely differentiable** ($C^\infty$). It dips slightly below zero, reaching a minimum value of $\approx -0.170$ at $x \approx -0.7517$.

### Analytical Derivative

Differentiating $\text{GELU}(x)$ with respect to $x$ using the product rule:

$$\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \cdot \Phi'(x) = \Phi(x) + x \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$

Notice that at $x = 0$:
$$\frac{d}{dx}\text{GELU}(0) = \Phi(0) + 0 = 0.5000$$

Unlike ReLU, whose subgradient at zero is discontinuous and arbitrarily set to $0$ or $1$, GELU exhibits a smooth gradient slope of exactly $0.5$ at the origin, providing gentle curvature that accelerates convergence in Transformer architectures.

### The Fast Tanh Approximation

Evaluating the error function $\text{erf}(x)$ on early GPU architectures incurred significant transcendental compute penalties. Hendrycks and Gimpel derived a highly accurate polynomial approximation based on hyperbolic tangents:

$$\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)$$

This formulation was adopted as the foundational standard in original BERT, GPT-2, GPT-3, and ViT (Vision Transformer). In modern hardware, while native error function units exist, frameworks like Hugging Face and PyTorch continue to support `gelu_pytorch_tanh` (seen in models such as Gemma 2 and Gemma 3).

---

## 5. The modern LLM standard: Gated Linear Units and SwiGLU

In 2017, Google Brain researchers (Prajit Ramachandran, Barret Zoph, Quoc V. Le) conducted an automated reinforcement learning architecture search over thousands of mathematical candidates to discover new activation functions. The search yielded **Swish** (identically known as **SiLU** — Sigmoid Linear Unit):

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

For $\beta = 1$:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

SiLU shares the smooth, non-monotonic curvature of GELU, with a slight negative dip to $-0.278$ at $x \approx -1.28$, but replaces the Gaussian CDF with the computationally simpler logistic sigmoid.

### The Bilinear Gating Revolution (GLU)

Concurrently, Yann Dauphin et al. (2016) demonstrated that convolutional networks for NLP benefited dramatically from **Gated Linear Units (GLU)**.

In a standard Multi-Layer Perceptron (MLP/FFN), inputs are linearly projected, passed through an element-wise activation, and projected back:

$$\text{FFN}_{\text{standard}}(x) = \sigma(x W_1 + b_1) W_2 + b_2$$

GLU replaces this single projection with two parallel linear projections whose outputs interact **multiplicatively**:

$$\text{GLU}(x, W, V) = \sigma(x W) \odot (x V)$$

Where $\odot$ represents the Hadamard (element-wise) product. 
* The first branch ($x W$) acts as a continuous, soft **gate**, outputting values between $0$ and $1$.
* The second branch ($x V$) provides the **linear signal (amplitude)**.

### Shazeer's SwiGLU Formulation

In 2020, Noam Shazeer published *"GLU Variants Improve Transformer"*, systematically replacing the sigmoid gate in GLU with modern non-linearities: ReGLU (using ReLU), GeGLU (using GELU), and **SwiGLU** (using Swish/SiLU):

$$\text{SwiGLU}(x, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}) = \left( \text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$

Where:
* $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$: Projects input into the gating subspace.
* $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$: Projects input into the unconstrained feature subspace.
* $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$: Contracts the gated representation back to model dimension.

<svg viewBox="0 0 560 196" role="img" aria-label="The SwiGLU feed-forward block. The input tensor x of size d_model is projected twice: W_gate feeds a SiLU/Swish activation to generate a dynamic gate; W_up carries the linear feature signal. The two branches meet at an element-wise Hadamard product, and W_down projects the representation back to d_model." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="70" width="92" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="62.0" y="90.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">x</text>
<text x="62.0" y="106.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">(d_model)</text>
<rect x="140" y="14" width="110" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="195.0" y="36.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W_gate</text>
<rect x="274" y="14" width="120" height="36" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="334.0" y="36.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">SiLU / Swish</text>
<rect x="140" y="138" width="254" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="267.0" y="160.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W_up (linear feature)</text>
<path d="M108 94 H124 V32 H138" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M124 94 V156 H138" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="250" y1="32" x2="272" y2="32" marker-end="url(#sw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="436" cy="94" r="18" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="436" y="100" text-anchor="middle" style="fill:var(--c-text);font-size:18px">⊙</text>
<path d="M394 32 H436 V74" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M394 156 H436 V114" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="460" y="70" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hadamard product</text>
<rect x="476" y="76" width="68" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="510.0" y="98.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W_down</text>
<line x1="454" y1="94" x2="474" y2="94" marker-end="url(#sw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="476" y="134" width="68" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="510.0" y="154.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">y</text>
<text x="510.0" y="170.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">(d_model)</text>
<line x1="510" y1="112" x2="510" y2="132" marker-end="url(#sw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="140" y="76" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gate branch: Swish(x W_gate)</text>
<text x="140" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">linear value: x W_up</text>
</svg>

### The Parameter Budget Equivalence ($\frac{8}{3} d_{\text{model}}$)

A standard 2-layer FFN contains two matrices:
* $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
* $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
* Total parameters (omitting biases) $= 2 \times d_{\text{model}} \times d_{\text{ff}}$.
* In the original Transformer (Vaswani et al., 2017), $d_{\text{ff}} = 4 d_{\text{model}}$, yielding $8 d_{\text{model}}^2$ parameters.

SwiGLU, however, introduces **three** projection matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). If $d_{\text{ff}}$ remained at $4 d_{\text{model}}$, SwiGLU would consume $3 \times d_{\text{model}} \times 4 d_{\text{model}} = 12 d_{\text{model}}^2$ parameters—a 50% increase in FLOPs and parameter count.

To maintain strict parameter and compute equivalence with standard architectures, Shazeer adjusted the intermediate hidden dimension:

$$3 \times (d_{\text{model}} \times d_{\text{ff}}) \approx 2 \times (d_{\text{model}} \times 4 d_{\text{model}})$$

$$d_{\text{ff}} = \frac{8}{3} d_{\text{model}} \approx 2.667 d_{\text{model}}$$

In production models, this intermediate size is typically rounded to the nearest multiple of 256 or 64 to optimize memory tile alignment for GPU Tensor Cores. For instance:
* **LLaMA 3 8B:** $d_{\text{model}} = 4096 \implies \frac{8}{3} \times 4096 \approx 10922.6 \to \mathbf{14336}$ (explicitly scaled for extended capacity).
* **Mistral 7B:** $d_{\text{model}} = 4096 \to d_{\text{ff}} = \mathbf{14336}$.
* **DeepSeek-V3:** Relies on SwiGLU across all Routed MoE and Shared Experts.

Why does SwiGLU decisively outperform standard GELU and ReLU across language benchmarks? Because multiplicative gating provides **first-order control over feature relevance**. Rather than statically compressing features through an additive non-linearity, one pathway dynamically suppresses irrelevant dimensions while amplifying salient activations based directly on the current token context.

---

## 6. Step-by-step manual tensor walkthrough

To demystify these mathematical formulations, let us walk through exact, deterministic numerical evaluations across two distinct scenarios:
1. **Part A:** 1D element-wise evaluations and exact derivatives across a three-point spectrum.
2. **Part B:** A complete 2D SwiGLU layer forward pass using small-scale toy projection matrices.

### Part A: 1D Element-wise Scalar Walk

Consider an input vector testing negative, zero, and positive regimes:

$$x = \begin{bmatrix} -1.5 & 0.0 & 2.0 \end{bmatrix}$$

Using the exact mathematical formulations:

#### Point 1: $x = -1.5$
* **Sigmoid:** $\sigma(-1.5) = \frac{1}{1 + e^{1.5}} = \frac{1}{1 + 4.481689} = \mathbf{0.182426}$
  * Derivative: $\sigma'(-1.5) = 0.182426 \times (1 - 0.182426) = \mathbf{0.149146}$
* **Tanh:** $\tanh(-1.5) = \frac{e^{-1.5} - e^{1.5}}{e^{-1.5} + e^{1.5}} = \frac{0.223130 - 4.481689}{0.223130 + 4.481689} = \mathbf{-0.905148}$
  * Derivative: $1 - (-0.905148)^2 = 1 - 0.819293 = \mathbf{0.180707}$
* **ReLU:** $\max(0, -1.5) = \mathbf{0.000000}$
  * Derivative: $\mathbf{0.000000}$ *(Dying neuron regime)*
* **LeakyReLU ($\alpha=0.01$):** $0.01 \times (-1.5) = \mathbf{-0.015000}$
  * Derivative: $\mathbf{0.010000}$
* **GELU (Exact):** $\Phi(-1.5) = 0.5 \times [1 + \text{erf}(-1.5 / \sqrt{2})] = 0.5 \times [1 - 0.866386] = 0.066807$
  * Value: $-1.5 \times 0.066807 = \mathbf{-0.100211}$
  * Derivative: $\Phi(-1.5) + (-1.5) \times \frac{1}{\sqrt{2\pi}} e^{-(-1.5)^2/2} = 0.066807 - 1.5 \times 0.129518 = \mathbf{-0.127469}$
* **SiLU (Swish):** $-1.5 \times \sigma(-1.5) = -1.5 \times 0.182426 = \mathbf{-0.273638}$
  * Derivative: $0.182426 \times [1 - 1.5 \times (1 - 0.182426)] = 0.182426 \times [1 - 1.226361] = \mathbf{-0.041294}$

#### Point 2: $x = 0.0$
* **Sigmoid:** $\sigma(0) = \mathbf{0.500000}$ | Derivative: $0.5 \times 0.5 = \mathbf{0.250000}$
* **Tanh:** $\tanh(0) = \mathbf{0.000000}$ | Derivative: $1 - 0^2 = \mathbf{1.000000}$
* **ReLU:** $\max(0, 0) = \mathbf{0.000000}$ | Derivative: $\mathbf{0.000000}$
* **GELU (Exact):** $0 \times \Phi(0) = \mathbf{0.000000}$ | Derivative: $\Phi(0) + 0 = \mathbf{0.500000}$
* **SiLU (Swish):** $0 \times \sigma(0) = \mathbf{0.000000}$ | Derivative: $\sigma(0) + 0 = \mathbf{0.500000}$

#### Point 3: $x = 2.0$
* **Sigmoid:** $\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.135335} = \mathbf{0.880797}$ | Derivative: $0.880797 \times (1 - 0.880797) = \mathbf{0.104994}$
* **Tanh:** $\tanh(2) = \mathbf{0.964028}$ | Derivative: $1 - (0.964028)^2 = \mathbf{0.070651}$
* **ReLU:** $\max(0, 2) = \mathbf{2.000000}$ | Derivative: $\mathbf{1.000000}$
* **GELU (Exact):** $\Phi(2) = 0.977250 \implies 2 \times 0.977250 = \mathbf{1.954500}$ | Derivative: $0.977250 + 2 \times 0.053991 = \mathbf{1.085232}$
* **SiLU (Swish):** $2 \times \sigma(2) = 2 \times 0.880797 = \mathbf{1.761594}$ | Derivative: $0.880797 \times [1 + 2 \times 0.119203] = \mathbf{1.090784}$

| Input ($x$) | Activation Function | Forward Value ($y$) | Backward Gradient ($\frac{dy}{dx}$) | Key Architectural Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **$-1.5$** | Sigmoid | $0.182426$ | $0.149146$ | Positive bias, suppressed gradient |
| | Tanh | $-0.905148$ | $0.180707$ | Strong saturation onset |
| | ReLU | $0.000000$ | $0.000000$ | **Dead neuron (gradient = 0)** |
| | LeakyReLU | $-0.015000$ | $0.010000$ | Preserves 1% recovery gradient |
| | GELU (Exact) | $-0.100211$ | $-0.127469$ | Smooth negative dip, active gradient |
| | SiLU / Swish | $-0.273638$ | $-0.041294$ | Smooth non-monotonic minimum |
| **$0.0$** | Sigmoid | $0.500000$ | $0.250000$ | Maximum gradient is only 0.25 |
| | Tanh | $0.000000$ | $1.000000$ | Zero-centered, unitary slope |
| | ReLU | $0.000000$ | $0.000000$ | Boundary non-differentiability |
| | GELU (Exact) | $0.000000$ | $0.500000$ | Smooth 0.5 slope |
| | SiLU / Swish | $0.000000$ | $0.500000$ | Smooth 0.5 slope |
| **$2.0$** | Sigmoid | $0.880797$ | $0.104994$ | Saturated; gradient collapses |
| | Tanh | $0.964028$ | $0.070651$ | Deep saturation; gradient vanishes |
| | ReLU | $2.000000$ | $1.000000$ | Perfect identity gradient highway |
| | GELU (Exact) | $1.954500$ | $1.085232$ | Near-identity with slight curvature boost |
| | SiLU / Swish | $1.761594$ | $1.090784$ | Near-identity with dynamic amplification |

```text
Scalar Derivatives at x = -1.5:
  ReLU:      0.000000  <-- Signal is completely erased.
  LeakyReLU: 0.010000  <-- Retains tiny 0.01 trickle.
  GELU:     -0.127469  <-- Informative negative gradient preserves feature dynamics.
  SiLU:     -0.041294  <-- Negative curvature gradient informs backward pass.
```

---

### Part B: Complete SwiGLU 2D Layer Walkthrough

Now let us execute a complete, step-by-step SwiGLU layer forward pass on a 2-dimensional vector with toy projection matrices ($d_{\text{in}} = 2, d_{\text{ff}} = 2, d_{\text{out}} = 2$):

$$x = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix}$$

Weights:

$$W_{\text{gate}} = \begin{bmatrix} 1.0 & -1.0 \\ 0.5 & 2.0 \end{bmatrix}, \quad W_{\text{up}} = \begin{bmatrix} 0.5 & 1.0 \\ -1.0 & 0.5 \end{bmatrix}, \quad W_{\text{down}} = \begin{bmatrix} 1.0 & 0.5 \\ 0.0 & 1.0 \end{bmatrix}$$

#### Step 1: Compute Gate and Up Linear Projections

$$z_{\text{gate}} = x W_{\text{gate}} = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix} \begin{bmatrix} 1.0 & -1.0 \\ 0.5 & 2.0 \end{bmatrix}$$

$$z_{\text{gate}}[0] = (1.0)(1.0) + (-0.5)(0.5) = 1.0 - 0.25 = \mathbf{0.75}$$

$$z_{\text{gate}}[1] = (1.0)(-1.0) + (-0.5)(2.0) = -1.0 - 1.0 = \mathbf{-2.00}$$

$$z_{\text{gate}} = \begin{bmatrix} 0.75 & -2.00 \end{bmatrix}$$

Similarly for $z_{\text{up}}$:

$$z_{\text{up}} = x W_{\text{up}} = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix} \begin{bmatrix} 0.5 & 1.0 \\ -1.0 & 0.5 \end{bmatrix}$$

$$z_{\text{up}}[0] = (1.0)(0.5) + (-0.5)(-1.0) = 0.5 + 0.5 = \mathbf{1.00}$$

$$z_{\text{up}}[1] = (1.0)(1.0) + (-0.5)(0.5) = 1.0 - 0.25 = \mathbf{0.75}$$

$$z_{\text{up}} = \begin{bmatrix} 1.00 & 0.75 \end{bmatrix}$$

#### Step 2: Apply SiLU Activation to Gate Projection

$$\text{SiLU}(z) = z \cdot \sigma(z)$$

For $z_{\text{gate}}[0] = 0.75$:
$$\sigma(0.75) = \frac{1}{1 + e^{-0.75}} = \frac{1}{1 + 0.472367} = 0.679179$$
$$\text{SiLU}(0.75) = 0.75 \times 0.679179 = \mathbf{0.509384}$$

For $z_{\text{gate}}[1] = -2.00$:
$$\sigma(-2.00) = \frac{1}{1 + e^{2.00}} = \frac{1}{1 + 7.389056} = 0.119203$$
$$\text{SiLU}(-2.00) = -2.00 \times 0.119203 = \mathbf{-0.238406}$$

$$\text{gate\_act} = \begin{bmatrix} 0.509384 & -0.238406 \end{bmatrix}$$

#### Step 3: Hadamard (Element-wise) Gating Interaction

$$h = \text{gate\_act} \odot z_{\text{up}}$$

$$h[0] = 0.509384 \times 1.00 = \mathbf{0.509384}$$

$$h[1] = -0.238406 \times 0.75 = \mathbf{-0.178804}$$

$$h = \begin{bmatrix} 0.509384 & -0.178804 \end{bmatrix}$$

Notice how dimension 1 is dynamically attenuated and inverted by the negative gate activation, while dimension 0 passes through with moderate attenuation.

#### Step 4: Down-Projection

$$\text{out} = h W_{\text{down}} = \begin{bmatrix} 0.509384 & -0.178804 \end{bmatrix} \begin{bmatrix} 1.0 & 0.5 \\ 0.0 & 1.0 \end{bmatrix}$$

$$\text{out}[0] = (0.509384)(1.0) + (-0.178804)(0.0) = \mathbf{0.509384}$$

$$\text{out}[1] = (0.509384)(0.5) + (-0.178804)(1.0) = 0.254692 - 0.178804 = \mathbf{0.075888}$$

$$\text{out} = \begin{bmatrix} 0.509384 & 0.075888 \end{bmatrix}$$

```text
Toy SwiGLU Forward Summary:
  Input Vector x:           [ 1.000000, -0.500000]
  Gate Projection z_gate:   [ 0.750000, -2.000000]
  Up Projection z_up:       [ 1.000000,  0.750000]
  SiLU Gate Activation:     [ 0.509384, -0.238406]
  Hadamard Product h:       [ 0.509384, -0.178804]
  Final Output Vector:      [ 0.509384,  0.075888]
```

---

## 7. Two engineering lenses: Training vs inference dynamics

When evaluating activation functions in production systems, architectural elegance must be balanced against concrete hardware constraints across both training and serving regimes.

| Engineering Dimension | Training Regime (Pre-training / Fine-tuning) | Inference Regime (Serving / Generation) |
| :--- | :--- | :--- |
| **Dominant Workload** | **Compute & Activation Memory Bound** | **Memory Bandwidth Bound (during Decode)** |
| **Activation Memory** | Requires saving intermediate tensors ($z_{\text{gate}}, z_{\text{up}}, \text{act}$) for backward pass | Ephemeral; intermediate SRAM registers freed immediately |
| **Arithmetic Intensity** | High FLOP/byte ratio due to large batch size ($B \times S \times D$) | Prefill: GEMM (Compute bound); Decode: GEMV (Memory bandwidth bound) |
| **Numerical Format** | Bfloat16 / FP32 master weights (avoids underflow in gradient chains) | FP16, BF16, or FP8/INT4 quantized matrix weights |
| **Hardware Bottleneck** | HBM VRAM capacity & Distributed inter-node all-reduce bandwidth | HBM-to-SRAM memory bandwidth during weight fetching |

### Training Lens: The Activation Memory Squeeze

In standard ReLU or GELU MLPs, the forward pass requires caching two tensors per token per layer for backward gradient calculation:
1. The pre-activation tensor $z_1 = x W_1$ (needed to compute $\sigma'(z_1)$).
2. The layer input $x$ (needed to compute $\frac{\partial \mathcal{L}}{\partial W_1} = x^T dZ_1$).

Under **SwiGLU**, the backward pass must compute derivatives across a Hadamard product:

$$\frac{\partial \mathcal{L}}{\partial z_{\text{up}}} = dH \odot \text{SiLU}(z_{\text{gate}})$$

$$\frac{\partial \mathcal{L}}{\partial z_{\text{gate}}} = \left( dH \odot z_{\text{up}} \right) \odot \text{SiLU}'(z_{\text{gate}})$$

This demands that the runtime retain **three distinct intermediate activations** in GPU VRAM for every token:
1. $z_{\text{gate}}$ (or its activated form $\text{SiLU}(z_{\text{gate}})$).
2. $z_{\text{up}}$.
3. The upstream projection input $x$.

In an 8B parameter model trained across a 4096-token sequence with batch size 32, this expands activation memory by **$1.5\times$** relative to a traditional 2-matrix FFN. To avoid out-of-memory (OOM) failures, engineers must employ **activation checkpointing** (recomputing SwiGLU forward passes during the backward phase) or implement **fused CUDA/Triton kernels** that combine linear projection, SiLU activation, and Hadamard multiplication in SRAM without round-tripping through high-bandwidth memory (HBM).

### Inference Lens: Prefill vs. Decode Dynamics

During inference, intermediate activations are strictly ephemeral. Once `out` is computed, $z_{\text{gate}}$ and $z_{\text{up}}$ are overwritten in GPU shared memory (SRAM), consuming zero persistent VRAM.

However, the hardware operational profile bifurcates cleanly between generation phases:
1. **The Prefill Phase (Prompt Ingestion):**
   * Processes all prompt tokens simultaneously ($N_{\text{ctx}} \times d_{\text{model}}$).
   * This is a General Matrix Multiplication (**GEMM**), exhibiting high arithmetic intensity. The extra matrix in SwiGLU executes with near-peak compute saturation on Tensor Cores.
2. **The Decode Phase (Autoregressive Generation):**
   * Generates a single token at a time ($N = 1$).
   * This collapses matrix multiplication into a General Matrix-Vector product (**GEMV**).
   * The GPU must read all three parameter matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) from HBM into on-chip cache just to process a single token vector. At $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$, SwiGLU requires transferring the exact same number of parameter bytes as a classical $4 d_{\text{model}}$ FFN, maintaining identical memory-bandwidth-bound latency.

---

## 8. Python and PyTorch verification

To verify the mathematical derivations and scalar values calculated in Section 6, the following standalone script implements each activation function from first mathematical principles using Python's standard library and validates the SwiGLU tensor pass against native PyTorch primitives:

```python
import math
import torch
import torch.nn.functional as F

# 1. First-principles implementations using standard math library
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh_fn(x):
    return math.tanh(x)

def tanh_grad(x):
    t = math.tanh(x)
    return 1.0 - t * t

def relu(x):
    return max(0.0, x)

def relu_grad(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_grad(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def phi_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def gelu_exact(x):
    return x * phi_cdf(x)

def gelu_exact_grad(x):
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)
    return phi_cdf(x) + x * pdf

def silu(x):
    return x * sigmoid(x)

def silu_grad(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

# Scalar verification
test_points = [-1.5, 0.0, 2.0]
print("=== 1D ELEMENT-WISE ACTIVATION VERIFICATION ===")
for x in test_points:
    print(f"\n--- Evaluation at x = {x:.1f} ---")
    print(f"Sigmoid:       val = {sigmoid(x):.6f}, grad = {sigmoid_grad(x):.6f}")
    print(f"Tanh:          val = {tanh_fn(x):.6f}, grad = {tanh_grad(x):.6f}")
    print(f"ReLU:          val = {relu(x):.6f}, grad = {relu_grad(x):.6f}")
    print(f"LeakyReLU:     val = {leaky_relu(x):.6f}, grad = {leaky_relu_grad(x):.6f}")
    print(f"GELU (exact):  val = {gelu_exact(x):.6f}, grad = {gelu_exact_grad(x):.6f}")
    print(f"SiLU / Swish:  val = {silu(x):.6f}, grad = {silu_grad(x):.6f}")

# 2. SwiGLU forward pass verification
print("\n=== TOY SWIGLU NUMERICAL VERIFICATION ===")
x_vec = [1.0, -0.5]
W_gate = [[1.0, -1.0], [0.5, 2.0]]
W_up   = [[0.5,  1.0], [-1.0, 0.5]]
W_down = [[1.0,  0.5], [0.0, 1.0]]

# Manual step 1: Linear projections
z_gate = [x_vec[0]*W_gate[0][0] + x_vec[1]*W_gate[1][0],
          x_vec[0]*W_gate[0][1] + x_vec[1]*W_gate[1][1]]
z_up   = [x_vec[0]*W_up[0][0] + x_vec[1]*W_up[1][0],
          x_vec[0]*W_up[0][1] + x_vec[1]*W_up[1][1]]

# Manual step 2: Gating
gate_act = [silu(z_gate[0]), silu(z_gate[1])]
h = [gate_act[0] * z_up[0], gate_act[1] * z_up[1]]

# Manual step 3: Down-projection
out = [h[0]*W_down[0][0] + h[1]*W_down[1][0],
       h[0]*W_down[0][1] + h[1]*W_down[1][1]]

print(f"Computed z_gate:   [{z_gate[0]:.6f}, {z_gate[1]:.6f}]")
print(f"Computed z_up:     [{z_up[0]:.6f}, {z_up[1]:.6f}]")
print(f"Computed gate_act: [{gate_act[0]:.6f}, {gate_act[1]:.6f}]")
print(f"Computed h:        [{h[0]:.6f}, {h[1]:.6f}]")
print(f"Computed out:      [{out[0]:.6f}, {out[1]:.6f}]")

# PyTorch autograd equivalence check
x_tensor = torch.tensor([[1.0, -0.5]], dtype=torch.float32, requires_grad=True)
W_g = torch.tensor([[1.0, -1.0], [0.5, 2.0]], dtype=torch.float32)
W_u = torch.tensor([[0.5,  1.0], [-1.0, 0.5]], dtype=torch.float32)
W_d = torch.tensor([[1.0,  0.5], [0.0, 1.0]], dtype=torch.float32)

torch_gate = x_tensor @ W_g
torch_up = x_tensor @ W_u
torch_h = F.silu(torch_gate) * torch_up
torch_out = torch_h @ W_d
print(f"\nPyTorch Tensor Output: {torch_out.detach().numpy().round(6).tolist()}")
```

Executing the verification script produces exact numerical parity across all decimal places:

```text
=== 1D ELEMENT-WISE ACTIVATION VERIFICATION ===

--- Evaluation at x = -1.5 ---
Sigmoid:       val = 0.182426, grad = 0.149146
Tanh:          val = -0.905148, grad = 0.180707
ReLU:          val = 0.000000, grad = 0.000000
LeakyReLU:     val = -0.015000, grad = 0.010000
GELU (exact):  val = -0.100211, grad = -0.127469
SiLU / Swish:  val = -0.273638, grad = -0.041294

--- Evaluation at x = 0.0 ---
Sigmoid:       val = 0.500000, grad = 0.250000
Tanh:          val = 0.000000, grad = 1.000000
ReLU:          val = 0.000000, grad = 0.000000
LeakyReLU:     val = 0.000000, grad = 0.010000
GELU (exact):  val = 0.000000, grad = 0.500000
SiLU / Swish:  val = 0.000000, grad = 0.500000

--- Evaluation at x = 2.0 ---
Sigmoid:       val = 0.880797, grad = 0.104994
Tanh:          val = 0.964028, grad = 0.070651
ReLU:          val = 2.000000, grad = 1.000000
LeakyReLU:     val = 2.000000, grad = 1.000000
GELU (exact):  val = 1.954500, grad = 1.085232
SiLU / Swish:  val = 1.761594, grad = 1.090784

=== TOY SWIGLU NUMERICAL VERIFICATION ===
Computed z_gate:   [0.750000, -2.000000]
Computed z_up:     [1.000000, 0.750000]
Computed gate_act: [0.509384, -0.238406]
Computed h:        [0.509384, -0.178804]
Computed out:      [0.509384, 0.075888]

PyTorch Tensor Output: [[0.509384, 0.075888]]
```

---

## The whole story in six lines

- Without non-linear activations, multi-layer neural networks collapse algebraically into a single linear regression matrix by associativity.
- Sigmoid and Tanh functions enforce bounded outputs but inflict severe vanishing gradients due to exponential saturation in the tails.
- ReLU enabled deep networks with unitary gradients and near-zero compute, but suffers from irreversible dead neuron failure modes.
- GELU introduced probabilistic gating via the Gaussian CDF, creating a smooth, non-monotonic curve standard in BERT, GPT, and ViT.
- Modern LLMs (LLaMA 3, Gemma 2, DeepSeek-V3) standardize on SwiGLU, using bilinear multiplicative gating for dynamic feature filtration.
- SwiGLU compensates for its three projection matrices by scaling intermediate dimensions to $\frac{8}{3} d_{\text{model}}$, equalizing total parameter and FLOP budgets.

---

## Glossary

- **Affine Transformation** — A linear mapping ($W x$) combined with a translation bias ($+ b$), preserving collinearity and flat geometric structures.
- **Vanishing Gradient** — The exponential decay of error signals across deep backpropagation chains caused by repeated multiplication of fractional derivatives ($\le 0.25$).
- **Dead Neuron** — A state where a ReLU unit receives large negative bias updates, rendering its activation and derivative permanently zero across all training examples.
- **Universal Approximation Theorem** — The mathematical proof that a feed-forward network with a single non-linear hidden layer can approximate any continuous function on compact subsets of $\mathbb{R}^n$.
- **GELU (Gaussian Error Linear Unit)** — An activation scaling inputs by their standard Gaussian cumulative probability, combining stochastic regularization with smooth non-linearity.
- **Hadamard Product ($\odot$)** — The element-wise multiplication of two identically shaped tensors, central to bilinear gating mechanisms in GLU and SwiGLU.
- **SwiGLU** — A Gated Linear Unit variant utilizing the Swish/SiLU non-linearity across parallel linear projections, standard across modern state-of-the-art LLMs.
- **Activation Checkpointing** — A memory-optimization technique that discards intermediate activations during the forward pass and recomputes them during backpropagation to prevent GPU VRAM exhaustion.

---

## To dive deeper

- [Nair & Hinton (ICML 2010): Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) — The seminal paper introducing ReLU to modern machine learning.
- [Hendrycks & Gimpel (arXiv:1606.08415, 2016): Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — The foundational paper on probabilistic gating.
- [Ramachandran, Zoph, & Le (arXiv:1710.05941, 2017): Searching for Activation Functions](https://arxiv.org/abs/1710.05941) — Google Brain's reinforcement learning discovery of Swish/SiLU.
- [Noam Shazeer (arXiv:2002.05202, 2020): GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — The architectural origin of SwiGLU, GeGLU, and modern Transformer FFN gating.
- On this blog: [A Prompt's Journey (7): The Transformer Block](post.html?slug=inside-the-transformer-block) — How SwiGLU and FFNs form associative key-value memory banks inside production language models.
