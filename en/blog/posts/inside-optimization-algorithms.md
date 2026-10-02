Imagine attempting to train a 100-layer transformer using vanilla Stochastic Gradient Descent (SGD). You initialize the parameters, launch the first training batch, and set a modest learning rate of $\eta = 0.01$. Within twenty steps, the loss violently diverges to `NaN`. You intervene, cautiously lowering the learning rate by a factor of one hundred to $\eta = 0.0001$. The divergence ceases, but training grinds to an imperceptible crawl: after an entire weekend of compute, the training loss has barely moved.

Why does vanilla gradient descent fail to find a viable middle ground in deep networks?

The failure is rooted in the **anisotropic geometry of high-dimensional loss landscapes**. Deep neural networks do not optimize over smooth, spherical bowls; they navigate treacherous, elongated **ravines**. Along the transverse canyon walls, curvature is violently steep: the largest eigenvalue of the Hessian matrix $\lambda_{\max}$ is immense. Along the canyon floor—the sole direction pointing toward the global minimum—the landscape is virtually flat ($\lambda_{\min} \approx 0$). In this ill-conditioned regime, any learning rate large enough to make progress down the canyon floor causes SGD to ricochet uncontrollably between the canyon walls.

Optimizers are the navigation algorithms that steer gradient descent through this geometric terrain. For decades, training relied on physical analogies of momentum and the scalar adaptive moments of Adam and AdamW. Today, large-scale LLM pre-training is undergoing its biggest optimization revolution in a decade with the advent of **Muon**—an optimizer that treats weight matrices not as arbitrary lists of independent scalars, but as 2D linear operators orthogonalized via Newton-Schulz iterations. This article deconstructs the geometry of ill-conditioned ravines, analyzes why AdamW was necessary to fix the L2 regularization paradox, explores the spectral mechanics of Muon, and walks through exact manual tensor arithmetic verified with runnable code.

**In this article**

- [1. The core intuition: The ravine pathology and why gradients oscillate](#1-the-core-intuition-the-ravine-pathology-and-why-gradients-oscillate)
- [2. From physical momentum to adaptive moments: Polyak, Nesterov, and RMSprop](#2-from-physical-momentum-to-adaptive-moments-polyak-nesterov-and-rmsprop)
- [3. The AdamW revolution: Decoupled weight decay vs L2 regularization](#3-the-adamw-revolution-decoupled-weight-decay-vs-l2-regularization)
- [4. The 2024-2026 pre-training breakthrough: The Muon optimizer](#4-the-2024-2026-pre-training-breakthrough-the-muon-optimizer)
- [5. Learning rate dynamics: Warmup, Cosine Annealing, and WSD schedules](#5-learning-rate-dynamics-warmup-cosine-annealing-and-wsd-schedules)
- [6. Step-by-step manual tensor walkthrough](#6-step-by-step-manual-tensor-walkthrough)
- [7. Two engineering lenses: Training vs inference dynamics](#7-two-engineering-lenses-training-vs-inference-dynamics)
- [8. Python and PyTorch verification](#8-python-and-pytorch-verification)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [To dive deeper](#to-dive-deeper)

---

## 1. The core intuition: The ravine pathology and why gradients oscillate

### The Ill-Conditioned Canyon

To understand why simple gradient descent fails in deep architectures, consider minimizing an anisotropic quadratic bowl:

$$f(w_1, w_2) = \frac{1}{2} w_1^2 + \frac{L}{2} w_2^2 \quad (L \gg 1)$$

The gradient vector at any arbitrary point $(w_1, w_2)$ is:

$$\nabla f(w) = \begin{bmatrix} w_1 \\ L w_2 \end{bmatrix}$$

The curvature of this surface is dictated by its Hessian matrix:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 \\ 0 & L \end{bmatrix}$$

The **condition number** $\kappa$ of the Hessian is the ratio of its largest to smallest eigenvalues:

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{L}{1} = L$$

In deep neural networks, condition numbers regularly exceed $\kappa \ge 10^4 \text{ to } 10^6$.

<svg viewBox="0 0 560 330" role="img" aria-label="The ill-conditioned ravine, with condition number kappa far above 10 to the 4. Along the wall direction, lambda max, curvature is steep and plain SGD bounces in violent transverse oscillations. Along the floor direction, lambda min, curvature is flat and forward progress is near zero. SGD is stuck: with a step size above 2 over lambda max it diverges; below it, it crawls along the floor. Three algorithmic fixes: momentum cancels the oscillations, since alternating gradients sum to about zero, and accumulates velocity along the floor; AdamW normalizes the step size per coordinate with an RMS; Muon orthogonalizes the whole 2D update matrix with Newton-Schulz iterations." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="rv-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="rv-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
</defs>
<text x="16" y="20" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700;letter-spacing:.06em">The ravine dilemma (κ ≫ 10⁴)</text>
<ellipse cx="230" cy="100" rx="200" ry="62" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<ellipse cx="230" cy="100" rx="140" ry="43" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<ellipse cx="230" cy="100" rx="80" ry="25" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<ellipse cx="230" cy="100" rx="24" ry="7.5" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<path d="M60 48 L73 150 L86 53 L99 144 L112 58 L125 140 L138 63 L151 134 L164 68" marker-end="url(#rv-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.6"/>
<circle cx="230" cy="100" r="3" style="fill:var(--c-text)"/>
<text x="238" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">minimum</text>
<path d="M448 60 V140" marker-start="url(#rv-arr)" marker-end="url(#rv-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="458" y="84" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">wall · λ_max</text>
<text x="458" y="99" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">steep curvature:</text>
<text x="458" y="113" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">violent oscillations</text>
<path d="M150 178 H310" marker-start="url(#rv-arr)" marker-end="url(#rv-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="230" y="196" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">floor · λ_min · flat curvature: near-zero forward progress</text>
<rect x="16" y="206" width="528" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280" y="225.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px">SGD: η &gt; 2/λ_max → diverges;  η &lt; 2/λ_max → crawls along the floor</text>
<text x="16" y="258" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">Algorithmic solutions</text>
<rect x="16" y="268" width="168" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="100.0" y="284.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Momentum</text>
<text x="100.0" y="300.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">cancels oscillations</text>
<text x="100.0" y="316.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">(Σ ±g ≈ 0), builds speed</text>
<rect x="196" y="268" width="168" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="280.0" y="284.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">AdamW</text>
<text x="280.0" y="300.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">normalizes step size</text>
<text x="280.0" y="316.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">per-coordinate RMS</text>
<rect x="376" y="268" width="168" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="460.0" y="284.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Muon</text>
<text x="460.0" y="300.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">orthogonalizes the 2D</text>
<text x="460.0" y="316.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">matrix (Newton–Schulz)</text>
</svg>

### The Stability Ceiling of Vanilla SGD

Under standard gradient descent with learning rate $\eta$, parameter updates follow:

$$w_1^{(t+1)} = (1 - \eta) w_1^{(t)}, \quad w_2^{(t+1)} = (1 - \eta L) w_2^{(t)}$$

For the iteration to remain numerically stable and converge, the contraction factor along every coordinate must satisfy $|1 - \eta \lambda_i| < 1$:

$$\eta < \frac{2}{\lambda_{\max}} = \frac{2}{L}$$

If the engineer sets $\eta \ge \frac{2}{L}$, updates along $w_2$ amplify exponentially: $w_2^{(t)} \to \pm \infty$ (training diverged).

However, if the engineer respects this stability boundary and sets $\eta = \frac{1}{L}$ to prevent wall oscillations, look at what happens along the $w_1$ canyon floor:

$$w_1^{(t+1)} = \left( 1 - \frac{1}{L} \right) w_1^{(t)}$$

When $L = 10{,}000$, after 1,000 full training steps, the model has only reduced its distance along the canyon floor by:

$$\left( 1 - 10^{-4} \right)^{1000} \approx 0.9048$$

The model has traversed less than $10\%$ of the distance to the minimum. The network is paralyzed by the steepest direction in the parameter space.

---

## 2. From physical momentum to adaptive moments: Polyak, Nesterov, and RMSprop

To conquer the ravine pathology, optimization evolved through three distinct paradigms: physical momentum, adaptive coordinate scaling, and moment estimation.

### Polyak Momentum (The Heavy Ball Method)

In 1964, Soviet mathematician Boris Polyak proposed modeling optimization as a physical particle with mass rolling down a potential well:

$$v_t = \beta v_{t-1} + g_t$$

$$\theta_t = \theta_{t-1} - \eta v_t$$

Where $v_t$ is the velocity vector, and $\beta \in [0, 1)$ represents the friction coefficient (typically $\beta = 0.9$).

Unrolling velocity across time reveals an Exponentially Weighted Moving Average (EWMA):

$$v_t = \sum_{\tau=0}^{t} \beta^{t-\tau} g_\tau$$

Geometrically, this confers an enormous advantage:
* **Across canyon walls ($w_2$):** The gradient alternates signs consecutively ($+L, -L, +L, -L$). In the velocity summation, these opposing vectors cancel out: $\sum \pm g \approx 0$.
* **Along the canyon floor ($w_1$):** The gradient points persistently in the same direction. The velocities accumulate coherently, accelerating by an amplification factor of:

$$\text{Effective Speedup} = \frac{1}{1 - \beta} = \frac{1}{1 - 0.9} = \mathbf{10\times}$$

Momentum dampens high-frequency oscillations while magnifying steady, low-frequency progress.

### Nesterov Accelerated Gradient (NAG)

In 1983, Yurii Nesterov refined Polyak's heavy ball by computing the gradient not at the current position, but **at the projected look-ahead position**:

$$g_t = \nabla f(\theta_{t-1} - \eta \beta v_{t-1})$$

$$v_t = \beta v_{t-1} + g_t$$

$$\theta_t = \theta_{t-1} - \eta v_t$$

By evaluating the slope where the ball's current momentum will carry it, NAG senses when the slope begins rising ahead of time, acting as a predictive brake that prevents overshooting sharp turns.

### The Adaptive Coordinate Era: AdaGrad and RMSprop

While momentum accelerates consistent directions, it still applies a single uniform learning rate scalar across all coordinates.

In 2011, John Duchi, Elad Hazan, and Yoram Singer introduced **AdaGrad**, scaling each coordinate inversely by the square root of its historical squared gradients:

$$G_t = G_{t-1} + g_t^2, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t$$

AdaGrad allowed rare, infrequent features to take larger steps. However, because $G_t = \sum_{\tau=1}^t g_\tau^2$ accumulates non-negative terms monotonically, $G_t \to \infty$, driving the effective learning rate $\frac{\eta}{\sqrt{G_t}} \to 0$. In deep networks, AdaGrad freezes long before converging.

In 2012, Geoffrey Hinton solved AdaGrad's premature freezing in his Coursera lectures by replacing the infinite monotonic sum with an Exponential Moving Average (EMA), creating **RMSprop**:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t$$

By setting $\beta_2 = 0.99$, RMSprop remembers only recent gradient magnitudes, ensuring the denominator tracks local curvature without decaying to zero.

### Adam: Adaptive Moment Estimation

In 2014, Diederik Kingma and Jimmy Ba unified momentum and RMSprop into **Adam**, tracking both the first moment (mean velocity) and second moment (uncentered variance):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{First Moment: Momentum})$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{Second Moment: RMSprop})$$

#### The Mathematics of Bias Correction
Because moments are typically initialized to zero ($m_0 = \mathbf{0}, v_0 = \mathbf{0}$), early estimates are heavily biased toward zero:
At $t = 1$: $m_1 = (1 - \beta_1) g_1 = (1 - 0.9) g_1 = \mathbf{0.1 g_1}$.

To prove the exact unbiasing factor, unroll $m_t$:

$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

Taking expectations under the assumption that true gradients $g_i$ come from a stationary distribution with mean $\mathbb{E}[g_i] = \mathbb{E}[g_t]$:

$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] = \mathbb{E}[g_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g_t] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g_t] \left( 1 - \beta_1^t \right)$$

Dividing by $1 - \beta_1^t$ recovers an unbiased estimator:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The final Adam parameter update is:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 3. The AdamW revolution: Decoupled weight decay vs L2 regularization

For years after Adam's publication, practitioners observed an empirical anomaly: **Standard SGD with momentum consistently outperformed Adam on competitive benchmarks like ImageNet.**

Researchers assumed Adam suffered from poor generalization. In 2017, Ilya Loshchilov and Frank Hutter identified the true culprit: **a catastrophic mathematical conflation of L2 regularization and Weight Decay.**

### The Equivalence in SGD

In standard SGD, adding an $L_2$ weight penalty $\frac{\lambda}{2} \|\theta\|_2^2$ to the objective function $\mathcal{L}(\theta)$ yields:

$$\nabla_{\theta} \left( \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2 \right) = g_t + \lambda \theta_{t-1}$$

The resulting parameter update is:

$$\theta_t = \theta_{t-1} - \eta (g_t + \lambda \theta_{t-1}) = (1 - \eta \lambda) \theta_{t-1} - \eta g_t$$

In vanilla SGD, $L_2$ regularization is strictly mathematically identical to **Weight Decay** (multiplying the weights by a constant factor $1 - \eta \lambda$ at each step).

### How L2 Regularization Breaks in Adam

When deep learning frameworks implemented $L_2$ regularization inside Adam, they simply added $\lambda \theta_{t-1}$ to the incoming gradient before passing it to the optimizer:

$$\tilde{g}_t = g_t + \lambda \theta_{t-1}$$

Now trace what occurs inside Adam's moments:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) (g_t + \lambda \theta_{t-1})$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t + \lambda \theta_{t-1})^2$$

The unrolled parameter update becomes:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \hat{m}_t + \frac{\lambda \theta_{t-1}}{1 - \beta_1^t} \right)$$

Look carefully at the regularizer term: **the weight penalty is divided by $\sqrt{\hat{v}_t}$!**

This causes an inverted regularization failure:
1. **Frequent features / Large gradients:** Weights that receive large, frequent updates have large $\hat{v}_t$. Their regularization penalty is divided by a huge denominator, **suppressing their weight decay**.
2. **Rare features / Small gradients:** Weights that receive tiny or sparse updates have near-zero $\hat{v}_t$. Their regularization penalty is divided by $\epsilon$, **massively amplifying their weight decay**.

Instead of applying uniform regularization, Adam with $L_2$ penalized rare weights disproportionately and let frequent weights grow unchecked, destroying generalization.

### The AdamW Fix: Decoupled Weight Decay

Loshchilov and Hutter restored the original formulation of weight decay by **decoupling** it completely from the gradient moments:

$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

<svg viewBox="0 0 560 230" role="img" aria-label="Adam with L2 regularization versus AdamW. In Adam with L2, the decay term lambda times w is added to the gradient, fed into the first and second moments, and divided by the square root of v, so weights with large gradient history are barely decayed. In AdamW, decoupled weight decay, the plain gradient g goes through the moments to a standard Adam step, and the weight decay, minus eta times lambda times w, bypasses the moments and is subtracted from the weights directly." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="aw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="8" width="528" height="96" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="29" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700;letter-spacing:.06em">Adam + L2 regularization</text>
<text x="530" y="29" text-anchor="end" style="fill:var(--c-danger);font-size:12px;font-weight:600">decay distorted</text>
<rect x="16" y="120" width="528" height="100" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="141" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">AdamW (decoupled weight decay)</text>
<text x="530" y="141" text-anchor="end" style="fill:var(--c-success);font-size:12px;font-weight:600">decay intact</text>
<rect x="30" y="40" width="150" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="105.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g + λ·w</text>
<rect x="206" y="40" width="150" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="281.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">1st &amp; 2nd moments</text>
<rect x="382" y="40" width="148" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="456.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">divided by √v</text>
<line x1="180" y1="57" x2="204" y2="57" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="356" y1="57" x2="380" y2="57" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="30" y="92" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">the decay term is rescaled per coordinate along with the gradient</text>
<rect x="30" y="152" width="150" height="30" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="105.0" y="171.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g</text>
<rect x="206" y="152" width="150" height="30" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="281.0" y="171.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">moments m, v</text>
<rect x="382" y="152" width="148" height="30" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="456.0" y="171.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">standard step</text>
<line x1="180" y1="167" x2="204" y2="167" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="356" y1="167" x2="380" y2="167" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="30" y="190" width="150" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="105.0" y="209.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">−η·λ·w</text>
<rect x="382" y="190" width="148" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="456.0" y="209.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">subtract from w</text>
<line x1="180" y1="205" x2="380" y2="205" marker-end="url(#aw-arr)" style="stroke:var(--c-success);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="280" y="199" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bypasses the moments</text>
</svg>

In AdamW, every weight decays by $(1 - \eta \lambda)$ uniformly, regardless of whether its gradient magnitude is $10^3$ or $10^{-6}$. This single fix restored Adam's generalization, establishing AdamW as the universal default optimizer for modern transformer architectures (BERT, GPT-3, LLaMA, Mistral).

---

## 4. The 2024-2026 pre-training breakthrough: The Muon optimizer

Despite the success of AdamW, an intrinsic architectural compromise remained: **AdamW treats 2D weight matrices as flat lists of independent 1D scalar coordinates.**

In a transformer block, hidden layer weights are linear operators: $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. They act on multidimensional vector spaces by rotating and scaling coordinate axes according to their **singular value spectrum** ($W = U \Sigma V^T$). By flattening matrices into 1D vectors, coordinate-wise optimizers ignore matrix rank, singular vectors, and spectral norms.

In late 2024, researcher Keller Jordan and collaborators introduced **Muon (MomentUm Orthogonalized by Newton-Schulz)**, igniting the biggest pre-training optimizer revolution since AdamW.

### The Core Concept: Steepest Descent in the Spectral Norm

AdamW performs coordinate-wise normalization by dividing by $\sqrt{\hat{v}_t}$, which can be viewed as steepest descent under the $\ell_\infty$ norm (clamping update steps into hypercubes).

Muon asks a deeper geometric question: **What is the steepest descent direction for a 2D linear mapping under the spectral norm (operator norm)?**

The mathematical answer is the **nearest orthogonal matrix**:

$$O = \arg\min_{Q^T Q = \mathbf{I}} \|G - Q\|_F = U V^T$$

Where $G = U \Sigma V^T$ is the Singular Value Decomposition (SVD) of the momentum matrix.

Replacing the raw gradient matrix $G$ with its orthogonal polar factor $U V^T$ effectively **equalizes all singular values to $1.0$**:

$$\Sigma_{\text{updated}} = \text{diag}(1, 1, \dots, 1)$$

* The primary singular directions (which dominate standard gradients) are bounded.
* The secondary and tail singular directions (which represent subtle, nuanced linguistic representations) are amplified to unit spectral magnitude.

### How to Compute Orthogonality Without SVD: Newton-Schulz Iterations

Computing the exact SVD of a $4096 \times 4096$ matrix costs $O(d^3)$ FLOPs and cannot be efficiently parallelized on modern GPU Tensor Cores.

Muon resolves this computational bottleneck using **Newton-Schulz iterations**—an iterative algorithm that computes the polar decomposition using purely matrix-matrix multiplications (GEMMs):

1. **Normalize Matrix Energy:**
   $$X_0 = \frac{G}{\|G\|_F}$$
2. **Execute $K$ Newton-Schulz Steps ($K \approx 5$):**
   $$X_{k+1} = \frac{1}{2} X_k \left( 3\mathbf{I} - X_k^T X_k \right)$$

Because this iteration requires only matrix multiplications and additions, modern GPU Tensor Cores evaluate 5 Newton-Schulz iterations in sub-millisecond runtimes, operating at peak hardware FLOP saturation.

<svg viewBox="0 0 560 270" role="img" aria-label="Muon&#x27;s update in five steps. Start from the raw momentum matrix G of a 2D hidden weight. Divide by its Frobenius norm: X0 equals G over the norm of G. Run five Newton-Schulz iterations: X k plus 1 equals one half X k times 3 I minus X k transpose X k. The result is an orthogonal update matrix: X transpose X is approximately the identity, all singular values equal 1. Update the weights: W becomes W minus eta X, a uniform spectral step." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="mu-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="8" width="528" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="33" text-anchor="start" style="fill:var(--c-text);font-size:13px">Raw momentum G (2D weight)</text>
<text x="530" y="33" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">G</text>
<line x1="60" y1="48" x2="60" y2="58" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="60" width="528" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="30" y="85" text-anchor="start" style="fill:var(--c-text);font-size:13px">Normalize energy</text>
<text x="530" y="85" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">X₀ = G / ‖G‖_F</text>
<line x1="60" y1="100" x2="60" y2="110" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="112" width="528" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="30" y="137" text-anchor="start" style="fill:var(--c-text);font-size:13px">5× Newton–Schulz</text>
<text x="530" y="137" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">X_k+1 = ½·X_k(3I − X_kᵀX_k)</text>
<line x1="60" y1="152" x2="60" y2="162" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="164" width="528" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="30" y="189" text-anchor="start" style="fill:var(--c-text);font-size:13px">Orthogonal update</text>
<text x="530" y="189" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">XᵀX ≈ I (all σᵢ = 1)</text>
<line x1="60" y1="204" x2="60" y2="214" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="216" width="528" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="241" text-anchor="start" style="fill:var(--c-text);font-size:13px">Uniform spectral step</text>
<text x="530" y="241" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">W ← W − η·X</text>
</svg>

### The Production Hybrid Architecture

In production large language models (such as Moonshot AI's Kimi series and modern open pre-training recipes), Muon is deployed in a **hybrid configuration**:

1. **Muon Engine:** Governs all 2D internal weight matrices (Attention Q, K, V, Output projections, and SwiGLU Gate, Up, Down matrices).
2. **AdamW Engine:** Governs all 1D vector parameters and non-matrix layers (Token Embeddings, RMSNorm gain vectors, biases, and final Unembedding heads).

### Concrete Engineering Benefits

* **25%–35% Faster Convergence:** Models trained with Muon reach benchmark loss thresholds in approximately $30\%$ fewer training tokens compared to optimally tuned AdamW.
* **VRAM Savings:** AdamW stores two state buffers per parameter in FP32 (first moment $m$ and second moment $v$). Muon tracks **only the first moment (momentum buffer $G$)** for all internal 2D layers, eliminating the second moment buffer entirely and saving gigabytes of VRAM per GPU.

| Optimizer | Parameter Representation | Update Normalization | Weight Decay Handling | State Memory Footprint | Primary Vulnerability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SGD + Momentum** | Flattened 1D Scalars | None (raw gradient) | Coupled with learning rate | $1\times$ FP32 buffer ($v$) | Severe ravine oscillations ($\kappa \gg 10^4$) |
| **Adam** | Flattened 1D Scalars | Coordinate-wise $\sqrt{v_t}$ | Distorted by L2 division | $2\times$ FP32 buffers ($m, v$) | Degraded generalization under L2 |
| **AdamW** | Flattened 1D Scalars | Coordinate-wise $\sqrt{v_t}$ | Decoupled ($1 - \eta \lambda$) | $2\times$ FP32 buffers ($m, v$) | Treats 2D matrix geometry as 1D scalars |
| **Muon** | **2D Linear Operators** | **Spectral Orthogonalization (Newton-Schulz)** | Decoupled spectral decay | **$1\times$ FP32 buffer** for 2D matrices | Applicable only to 2D matrices (requires AdamW hybrid) |

---

## 5. Learning rate dynamics: Warmup, Cosine Annealing, and WSD schedules

An optimizer update rule is only half the equation; the trajectory of the learning rate scalar $\eta(t)$ over billions of training steps dictates model stability.

### The Necessity of Warmup

At step $t = 0$, model weights are random, loss gradients are turbulent, and the optimizer's moment estimates ($m_0, v_0$) have gathered zero statistical history. If a large learning rate $\eta_{\max}$ is applied at step 0, high-variance gradient spikes knock weight matrices into distorted configurations, causing irreversible activation instability.

**Learning Rate Warmup** linearly ramps $\eta$ from 0 to $\eta_{\max}$ across the first $1\%$ to $5\%$ of training steps:

$$\eta(t) = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} \quad (t \le T_{\text{warmup}})$$

Warmup allows AdamW's second moment $v_t$ and Muon's momentum buffers to stabilize before large parameter updates occur.

### Cosine Annealing

Introduced by Loshchilov and Hutter (2016), **Cosine Decay** gradually reduces the learning rate following a half-period cosine curve:

$$\eta(t) = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \pi \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \right) \right)$$

Cosine decay smooths optimization transitions, helping weights settle into flat, wide local minima that generalize superiorly.

### The Modern Frontier: Warmup-Stable-Decay (WSD)

While cosine decay is effective, it possesses a severe production flaw: **the decay curve is hardcoded to a fixed horizon $T_{\text{total}}$**. If an organization trains a model for 1 trillion tokens under cosine decay, they cannot easily extend training to 2 trillion tokens without restarting or distorting the schedule.

Modern architectures (MiniCPM, DeepSeek) deploy the **Warmup-Stable-Decay (WSD)** schedule:
1. **Warmup Phase:** Linear ramp to $\eta_{\max}$ (first $1\%-2\%$).
2. **Stable Phase:** Constant learning rate $\eta = \eta_{\max}$ for the vast majority of training ($80\%-90\%$). The model explores representations continuously. Training can be extended indefinitely.
3. **Decay Phase:** When pre-training budget limits approach, an aggressive 10% decay (linear, cosine, or 1-sqrt) anneals the learning rate to zero, extracting final benchmark performance.

---

## 6. Step-by-step manual tensor walkthrough

To make these optimization dynamics concrete, let us trace an exact manual calculation across an ill-conditioned 2D quadratic valley and execute a complete 5-step Newton-Schulz matrix orthogonalization.

### Setup: The Ill-Conditioned Valley

Let parameters $w = \begin{bmatrix} w_1 & w_2 \end{bmatrix}$ minimize:

$$f(w_1, w_2) = \frac{1}{2} w_1^2 + 10.0 w_2^2 \implies \nabla f(w) = \begin{bmatrix} w_1 \\ 20.0 w_2 \end{bmatrix}$$

Condition number $\kappa = \frac{20}{1} = \mathbf{20}$.
Starting point:

$$w^{(0)} = \begin{bmatrix} 2.0 & 1.0 \end{bmatrix}, \quad g^{(0)} = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}$$

---

### Step 1: Vanilla SGD Update ($\eta = 0.05$)

$$w^{(1)} = w^{(0)} - \eta g^{(0)} = \begin{bmatrix} 2.0 - 0.05 \times 2.0 \\ 1.0 - 0.05 \times 20.0 \end{bmatrix} = \begin{bmatrix} 2.0 - 0.10 \\ 1.0 - 1.00 \end{bmatrix} = \begin{bmatrix} \mathbf{1.900000} & \mathbf{0.000000} \end{bmatrix}$$

* Notice: along $w_2$, the step reached the bottom in one jump.
* But along $w_1$, the position moved only from $2.0$ to $1.90$ ($5\%$ progress).
* If $\eta$ was set to $0.11$, $w_2^{(1)} = 1.0 - 0.11(20) = -1.20$, oscillating and diverging.

---

### Step 2: Momentum Update ($\beta = 0.9, \eta = 0.05$)

Initial velocity $v_1 = g^{(0)} = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}$.

$$w^{(1)} = w^{(0)} - \eta v_1 = \begin{bmatrix} 2.0 - 0.05(2.0) \\ 1.0 - 0.05(20.0) \end{bmatrix} = \begin{bmatrix} \mathbf{1.900000} & \mathbf{0.000000} \end{bmatrix}$$

At step 2, gradient is $g^{(1)} = \begin{bmatrix} 1.90 & 20.0(0.0) \end{bmatrix} = \begin{bmatrix} 1.90 & 0.0 \end{bmatrix}$.

$$v_2 = 0.9 v_1 + g^{(1)} = 0.9 \begin{bmatrix} 2.0 \\ 20.0 \end{bmatrix} + \begin{bmatrix} 1.90 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 1.80 + 1.90 \\ 18.00 + 0.00 \end{bmatrix} = \begin{bmatrix} \mathbf{3.700000} \\ \mathbf{18.000000} \end{bmatrix}$$

$$w^{(2)} = w^{(1)} - \eta v_2 = \begin{bmatrix} 1.90 - 0.05(3.70) \\ 0.00 - 0.05(18.00) \end{bmatrix} = \begin{bmatrix} \mathbf{1.715000} & \mathbf{-0.900000} \end{bmatrix}$$

Velocity along the flat axis accelerated from $2.0 \to 3.70$, doubling step speed.

---

### Step 3: Adam vs. AdamW Walkthrough ($t=1, \eta=0.1, \lambda=0.01, \beta_1=0.9, \beta_2=0.999$)

Gradients: $g = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}$.

1. First moment: $m_1 = (1 - 0.9) g = \begin{bmatrix} 0.2 & 2.0 \end{bmatrix}$.
2. Second moment: $v_1 = (1 - 0.999) g^2 = 0.001 \begin{bmatrix} 4.0 & 400.0 \end{bmatrix} = \begin{bmatrix} 0.004 & 0.400 \end{bmatrix}$.
3. Bias correction at $t=1$:
   $$\hat{m}_1 = \frac{m_1}{1 - 0.9} = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}, \quad \hat{v}_1 = \frac{v_1}{1 - 0.999} = \begin{bmatrix} 4.0 & 400.0 \end{bmatrix}$$
4. Coordinate update step:
   $$\text{step}_1 = \frac{\hat{m}_{1, 1}}{\sqrt{\hat{v}_{1, 1}} + \epsilon} = \frac{2.0}{\sqrt{4.0}} = \mathbf{1.000000}$$
   $$\text{step}_2 = \frac{\hat{m}_{1, 2}}{\sqrt{\hat{v}_{1, 2}} + \epsilon} = \frac{20.0}{\sqrt{400.0}} = \mathbf{1.000000}$$

Notice how Adam completely neutralizes the $20\times$ gradient discrepancy, assigning identical unit step magnitudes to both coordinates.

#### Parameter Updates:
* **Adam (No Weight Decay):**
  $$w^{(1)} = \begin{bmatrix} 2.0 - 0.1(1.0) & 1.0 - 0.1(1.0) \end{bmatrix} = \begin{bmatrix} \mathbf{1.900000} & \mathbf{0.900000} \end{bmatrix}$$
* **AdamW (Decoupled Weight Decay $\lambda = 0.01$):**
  $$w_1^{(1)} = 2.0 - \eta \lambda (2.0) - \eta(1.0) = 2.0 - 0.002 - 0.1 = \mathbf{1.898000}$$
  $$w_2^{(1)} = 1.0 - \eta \lambda (1.0) - \eta(1.0) = 1.0 - 0.001 - 0.1 = \mathbf{0.899000}$$

---

### Step 4: Muon Newton-Schulz Matrix Orthogonalization

Now, trace Muon's 2D matrix orthogonalization on a $2 \times 2$ momentum matrix:

$$G = \begin{bmatrix} 2.0 & 1.0 \\ 0.5 & 3.0 \end{bmatrix}$$

#### Step 4.1: Compute Frobenius Norm and Normalize
$$\|G\|_F = \sqrt{2.0^2 + 1.0^2 + 0.5^2 + 3.0^2} = \sqrt{4.0 + 1.0 + 0.25 + 9.0} = \sqrt{14.25} \approx \mathbf{3.774917}$$

$$X_0 = \frac{G}{\|G\|_F} = \begin{bmatrix} \frac{2.0}{3.774917} & \frac{1.0}{3.774917} \\ \frac{0.5}{3.774917} & \frac{3.0}{3.774917} \end{bmatrix} = \begin{bmatrix} \mathbf{0.529813} & \mathbf{0.264906} \\ \mathbf{0.132453} & \mathbf{0.794719} \end{bmatrix}$$

#### Step 4.2: Newton-Schulz Iterations ($X_{k+1} = \frac{1}{2} X_k (3\mathbf{I} - X_k^T X_k)$)
Evaluating across 5 successive matrix multiplications:
* **Iteration 1 ($X_1$):**
  $$X_1 = \begin{bmatrix} 0.683180 & 0.239345 \\ 0.081331 & 0.896964 \end{bmatrix}$$
* **Iteration 2 ($X_2$):**
  $$X_2 = \begin{bmatrix} 0.834780 & 0.175106 \\ -0.003304 & 0.949314 \end{bmatrix}$$
* **Iteration 3 ($X_3$):**
  $$X_3 = \begin{bmatrix} 0.948780 & 0.121369 \\ -0.071699 & 0.981894 \end{bmatrix}$$
* **Iteration 4 ($X_4$):**
  $$X_4 = \begin{bmatrix} 0.990978 & 0.101423 \\ -0.097063 & 0.993884 \end{bmatrix}$$
* **Iteration 5 ($X_5$):**
  $$X_5 = \begin{bmatrix} \mathbf{0.995005} & \mathbf{0.099519} \\ \mathbf{-0.099485} & \mathbf{0.995028} \end{bmatrix}$$

#### Step 4.3: Orthogonality Verification ($X_5^T X_5$)
$$X_5^T X_5 = \begin{bmatrix} 0.999933 & 0.000032 \\ 0.000032 & 0.999985 \end{bmatrix} \approx \begin{bmatrix} \mathbf{1.0} & \mathbf{0.0} \\ \mathbf{0.0} & \mathbf{1.0} \end{bmatrix} = \mathbf{I}$$

In only five matrix multiplications, Newton-Schulz has equalized all singular values of the update matrix to unity.

| Optimization Step | Mathematical Operation | Coordinate 1 ($w_1$) | Coordinate 2 ($w_2$) | Key Geometric Consequence |
| :--- | :--- | :--- | :--- | :--- |
| **Initial State** | None | $2.000000$ | $1.000000$ | Loss $= 12.000000$ |
| **Vanilla SGD** | $w - \eta g$ | $1.900000$ | $0.000000$ | $w_2$ converged, $w_1$ made only 5% progress |
| **Momentum ($v_2$)** | $w - \eta (\beta v_1 + g)$ | $1.715000$ | $-0.900000$ | Floor direction accelerated by $2\times$ |
| **AdamW ($t=1$)** | Decoupled Weight Decay | $1.898000$ | $0.899000$ | Equalized coordinate steps; clean decay |
| **Muon ($X_5^T X_5$)** | Newton-Schulz ($k=5$) | $\text{diag}_1 \approx 0.9999$ | $\text{diag}_2 \approx 0.9999$ | Full 2D matrix update orthogonalized |

```text
Summary of Manual Walkthrough:
  SGD Step:               [ 1.900000, 0.000000 ]
  Momentum Step 2:        [ 1.715000, -0.900000 ]
  AdamW Step 1:           [ 1.898000, 0.899000 ]
  Muon X_5 Frobenius Norm: 1.414167 (Ideal sqrt(2) for 2x2 Orthogonal Matrix)
  Final Orthogonality:    X^T X = [[ 0.999933, 0.000032 ], [ 0.000032, 0.999985 ]]
```

---

## 7. Two engineering lenses: Training vs inference dynamics

The optimizer is the undisputed memory elephant of model training, but undergoes complete extinction in production serving.

| Engineering Dimension | Training Regime (Pre-training / SFT) | Inference Regime (Serving / Generation) |
| :--- | :--- | :--- |
| **Optimizer Footprint** | **Dominates VRAM:** Up to $12\times$ parameter size in FP32 states | **Zero:** Optimizer is completely non-existent |
| **State Buffers** | AdamW: FP32 Master Weights, First Moments, Second Moments | None; only static inference weights loaded |
| **Learning Rate Schedule** | Dynamically modulated by step counters (Warmup, WSD) | Irrelevant; weights are static constants |
| **Distributed Sharding** | ZeRO-1 / ZeRO-2 / FSDP partitions optimizer states across nodes | Model parallelism (TP, PP) shards weight matrices only |
| **Hardware Bottleneck** | HBM capacity for optimizer states; all-gather parameter broadcast | HBM memory bandwidth during weight reading |

### Training Lens: The Optimizer Memory Monster

Consider an 8-billion parameter model trained in mixed precision (BF16 weights and gradients):
* **Model Weights (BF16):** $8\text{B} \times 2\text{ bytes} = \mathbf{16\text{ GB}}$
* **Gradients (BF16):** $8\text{B} \times 2\text{ bytes} = \mathbf{16\text{ GB}}$

Now examine the memory consumed by **AdamW**:
1. **FP32 Master Weights:** To prevent underflow when small updates $\eta \Delta w$ are added to weights, frameworks maintain an FP32 copy: $8\text{B} \times 4\text{ bytes} = \mathbf{32\text{ GB}}$.
2. **First Moment Buffer ($m$ in FP32):** $8\text{B} \times 4\text{ bytes} = \mathbf{32\text{ GB}}$.
3. **Second Moment Buffer ($v$ in FP32):** $8\text{B} \times 4\text{ bytes} = \mathbf{32\text{ GB}}$.

**Total AdamW Memory: $96\text{ GB}$!**
The optimizer consumes **six times more memory than the model itself**. Across model, gradients, and optimizer, training requires $128\text{ GB}$—exceeding the capacity of an 80 GB NVIDIA H100 GPU before a single activation tensor is allocated.

### The Muon and ZeRO Relievers
To break this memory wall:
1. **Muon:** Replaces both FP32 moments with a single momentum buffer for all 2D layers, saving $32\text{ GB}$ of state memory.
2. **ZeRO-1 / FSDP:** Shards the $96\text{ GB}$ optimizer state across all $N$ GPUs in a cluster, reducing per-GPU optimizer memory to $96 / N\text{ GB}$.

### Inference Lens: Zero Memory Overhead
During inference, the optimizer, its moment buffers, master weights, learning rate schedules, and gradient clipping thresholds are entirely stripped away. The model is saved as a static parameter checkpoint, allowing 80 GB GPUs to dedicate memory entirely to KV caches.

---

## 8. Python and PyTorch verification

The following standalone script implements vanilla SGD, Momentum, Adam, AdamW, and the 5-step Muon Newton-Schulz orthogonalization from first principles using Python's standard library, verifying analytical results against PyTorch:

```python
import math
import torch

# 1. Quadratic Valley Problem setup
# f(w1, w2) = 0.5 * w1^2 + 10.0 * w2^2
w0 = [2.0, 1.0]
g0 = [w0[0], 20.0 * w0[1]]

print("=== 1. LOSS LANDSCAPE AND GRADIENTS ===")
print(f"w0:                {w0}")
print(f"g0:                {g0}")

# 2. Vanilla SGD
eta_sgd = 0.05
w1_sgd = [w0[0] - eta_sgd * g0[0], w0[1] - eta_sgd * g0[1]]
print("\n=== 2. VANILLA SGD ===")
print(f"w1 (SGD eta=0.05): {[round(v, 6) for v in w1_sgd]}")

# 3. Momentum
beta_m = 0.9
v1 = [g0[0], g0[1]]
w1_mom = [w0[0] - eta_sgd * v1[0], w0[1] - eta_sgd * v1[1]]
print("\n=== 3. MOMENTUM ===")
print(f"v1 (Momentum):     {v1}")
print(f"w1 (Momentum):     {[round(v, 6) for v in w1_mom]}")

# 4. Adam vs AdamW
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
eta = 0.1
wd = 0.01

m1 = [(1 - beta1) * g for g in g0]
v1_adam = [(1 - beta2) * (g**2) for g in g0]
m1_hat = [m / (1 - beta1**1) for m in m1]
v1_hat = [v / (1 - beta2**1) for v in v1_adam]
step_adam = [m_h / (math.sqrt(v_h) + eps) for m_h, v_h in zip(m1_hat, v1_hat)]

w1_adam_nowd = [w - eta * s for w, s in zip(w0, step_adam)]
w1_adamw = [w - eta * wd * w - eta * s for w, s in zip(w0, step_adam)]

print("\n=== 4. ADAM VS ADAMW ===")
print(f"m1_hat:            {[round(v, 6) for v in m1_hat]}")
print(f"v1_hat:            {[round(v, 6) for v in v1_hat]}")
print(f"Normalized Step:   {[round(v, 6) for v in step_adam]}")
print(f"Adam (No WD):      {[round(v, 6) for v in w1_adam_nowd]}")
print(f"AdamW (Decoupled): {[round(v, 6) for v in w1_adamw]}")

# 5. Muon Newton-Schulz Matrix Orthogonalization
G = [[2.0, 1.0], [0.5, 3.0]]

def mat_frob_norm(M):
    return math.sqrt(sum(M[i][j]**2 for i in range(len(M)) for j in range(len(M[0]))))

def mat_mul(A, B):
    n, k, m = len(A), len(A[0]), len(B[0])
    res = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            res[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return res

def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]

frob = mat_frob_norm(G)
X = mat_scale(G, 1.0 / frob)
I = [[1.0, 0.0], [0.0, 1.0]]

print("\n=== 5. MUON NEWTON-SCHULZ ORTHOGONALIZATION ===")
print(f"Input Matrix G:    {G}")
print(f"Frobenius Norm:    {frob:.6f}")

for step in range(1, 6):
    Xt = mat_transpose(X)
    XtX = mat_mul(Xt, X)
    three_I = mat_scale(I, 3.0)
    bracket = mat_sub(three_I, XtX)
    X = mat_scale(mat_mul(X, bracket), 0.5)
    print(f"X_{step}:              {[[round(c, 6) for c in row] for row in X]}")

XtX_final = mat_mul(mat_transpose(X), X)
print(f"Final X^T @ X:     {[[round(c, 6) for c in row] for row in XtX_final]}")

# PyTorch comparison for AdamW
t_w = torch.tensor([2.0, 1.0], dtype=torch.float32, requires_grad=True)
opt = torch.optim.AdamW([t_w], lr=0.1, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
t_loss = 0.5 * t_w[0]**2 + 10.0 * t_w[1]**2
t_loss.backward()
opt.step()
print(f"\nPyTorch AdamW Update: {t_w.detach().numpy().round(6).tolist()}")
```

Executing the script verifies exact analytical and framework parity across all optimization steps:

```text
=== 1. LOSS LANDSCAPE AND GRADIENTS ===
w0:                [2.0, 1.0]
g0:                [2.0, 20.0]

=== 2. VANILLA SGD ===
w1 (SGD eta=0.05): [1.9, 0.0]

=== 3. MOMENTUM ===
v1 (Momentum):     [2.0, 20.0]
w1 (Momentum):     [1.9, 0.0]

=== 4. ADAM VS ADAMW ===
m1_hat:            [2.0, 20.0]
v1_hat:            [4.0, 400.0]
Normalized Step:   [1.0, 1.0]
Adam (No WD):      [1.9, 0.9]
AdamW (Decoupled): [1.898, 0.899]

=== 5. MUON NEWTON-SCHULZ ORTHOGONALIZATION ===
Input Matrix G:    [[2.0, 1.0], [0.5, 3.0]]
Frobenius Norm:    3.774917
X_1:              [[0.68318, 0.239345], [0.081331, 0.896964]]
X_2:              [[0.83478, 0.175106], [-0.003304, 0.949314]]
X_3:              [[0.94878, 0.121369], [-0.071699, 0.981894]]
X_4:              [[0.990978, 0.101423], [-0.097063, 0.993884]]
X_5:              [[0.995005, 0.099519], [-0.099485, 0.995028]]
Final X^T @ X:     [[0.999933, 3.2e-05], [3.2e-05, 0.999985]]

PyTorch AdamW Update: [1.898, 0.899]
```

---

## The whole story in six lines

- Deep loss landscapes form ill-conditioned ravines ($\kappa \gg 10^4$), causing vanilla SGD to oscillate wildly across canyon walls while stalling along the floor.
- Polyak momentum accumulates velocity along consistent directions, accelerating floor progress by $\frac{1}{1-\beta}$ ($10\times$).
- L2 regularization breaks in Adam because dividing the penalty by $\sqrt{\hat{v}_t}$ suppresses decay on frequent features and over-penalizes rare features.
- AdamW decouples weight decay from gradient moments, restoring uniform parameter shrinkage and fixing deep network generalization.
- The Muon optimizer treats 2D hidden layer weights as linear operators, using Newton-Schulz iterations to equalize singular values to unity.
- In LLM pre-training, AdamW states consume $6\times$ more memory than the model itself; Muon saves one state buffer while accelerating training by $30\%$.

---

## Glossary

- **Condition Number ($\kappa$)** — The ratio of maximum to minimum curvature eigenvalues ($\lambda_{\max}/\lambda_{\min}$), measuring landscape anisotropy.
- **Momentum** — An optimization velocity buffer accumulating past gradients with an exponential decay factor $\beta$, damping oscillations.
- **Decoupled Weight Decay** — An update subtracting $\eta \lambda \theta$ directly from parameters, completely isolated from adaptive gradient scaling.
- **AdamW** — The standard transformer optimizer combining first moments, second moments, and decoupled weight decay.
- **Muon (MomentUm Orthogonalized by Newton-Schulz)** — A 2D matrix optimizer mapping momentum buffers to their nearest orthogonal polar matrix.
- **Newton-Schulz Iteration** — A matrix multiplication algorithm computing matrix polar orthogonalization without expensive SVD decompositions.
- **Warmup-Stable-Decay (WSD)** — A learning rate schedule maintaining peak learning rates indefinitely before a rapid terminal decay phase.
- **ZeRO (Zero Redundancy Optimizer)** — A distributed memory framework sharding optimizer states, gradients, and parameters across training nodes.

---

## To dive deeper

- [Polyak (USSR Computational Mathematics and Mathematical Physics 1964): Some methods of speeding up the convergence of iteration methods](https://www.sciencedirect.com/science/article/pii/0041555364901375) — The foundational origin of momentum in optimization.
- [Kingma & Ba (ICLR 2015): Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — The seminal paper introducing the Adam optimizer.
- [Loshchilov & Hutter (ICLR 2019): Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — The breakthrough paper introducing AdamW and analyzing the L2 failure.
- [Keller Jordan (2024): Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — The foundational introduction of the Muon optimizer.
- On this blog: [Building Blocks of Neural Networks (3): Inside Backpropagation and Autodiff](post.html?slug=inside-backpropagation-and-autodiff) — How gradients are computed and accumulated before passing to the optimizer engine.
