A customer requests that a company erase all of their personal data. The CRM team triggers the deletion routine, and the record disappears in under a second. Three weeks later, a support representative asks the AI assistant to summarize an old email thread, and the summary helpfully outputs that exact customer's phone number. The CRM had indeed deleted the original entry. But the phone number still lived inside a chunk in the vector index, a cached semantic response, an exported trace log, an evaluation dataset curated to "look realistic," and a nightly backup image.

This is the most common data security failure in AI systems. It is not an algorithmic defect; it is a systemic inventory and lifecycle failure. Data entering an AI pipeline multiplies. A single customer complaint splinters into a retrieval passage, a dense embedding vector, a context prompt, a generated completion, an observability span, a semantic cache entry, and potentially a fine-tuning dataset. Each copy resides in a distinct storage tier with its own access control lists and retention schedules. If the deletion pipeline only accounts for the source database, erasure remains incomplete regardless of execution speed.

> **The Foundational Axiom of AI Data Security (Embedding Invertibility and Model Permanence):**
> Dense embedding vectors ($\mathbf{v} \in \mathbb{R}^d$) are not anonymized representations; embedding inversion techniques such as Vec2Text can reconstruct much of the original text, so vectors belong in the same data classification tier as the text they were computed from. Furthermore, once personal data is baked into model weights through fine-tuning, no operational query removes it, and today's unlearning methods cannot guarantee that they did. Personal data must be stripped prior to training through differential privacy and deterministic sanitization.

This article traces enterprise data through our support assistant from the moment of ingestion to its eventual erasure. We will classify the five data tiers processed by an AI system, derive the gradient mechanics of clean-label data poisoning via Koh-Liang influence functions ($I_{\text{up,loss}}$), formulate the mathematical guarantees of $(\epsilon, \delta)$-Differential Privacy, construct a multi-layered PII detector pairing modular checksums with context-aware NER, and delineate the technical and legal boundaries separating masking, pseudonymization, and true anonymization. By the end, you will have a comprehensive inventory across nine storage domains, a production PII transformation pipeline, and an auditable lifecycle checklist.

> **Carried over from Part 2:** In [Part 2](post.html?slug=ai-threat-modeling-and-trust-boundaries), our Data Flow Diagram marked two data boundaries: observability log exports (TB6) and external untrusted complaint ingestion (TB4). Now, we examine where that data multiplies across system storage and how to prevent it from permanently corrupting model weights.

> **By the end of this article**
>
> - You will be able to categorize the five data tiers of an AI system and map the distinct architectural controls required by each.
> - You will be able to explain how clean-label poisoning manipulates gradient vectors via influence functions ($I_{\text{up,loss}}$) and how split-view poisoning compromises web-scale corpora.
> - You will be able to explain how $(\epsilon, \delta)$-Differential Privacy bounds weight memorization and why dense vectors must be treated as plaintext PII.
> - You will be able to implement a modular checksum validator for structured ID numbers and identify when Turkish NER is required for contextual entities.
> - You will be able to calculate complete erasure probability across $N$ redundant stores ($(1-q)^N$) and implement crypto-shredding to reduce deletion to a single key revocation.

**In this article**

- [1. One record, nine homes: the data multiplication map](#1-one-record-nine-homes-the-data-multiplication-map)
- [2. Five data tiers, five security requirements](#2-five-data-tiers-five-security-requirements)
- [3. Provenance and poisoning: influence functions and web-scale sabotage](#3-provenance-and-poisoning-influence-functions-and-web-scale-sabotage)
- [4. Differential privacy and embedding inversion attacks](#4-differential-privacy-and-embedding-inversion-attacks)
- [5. Finding personal data: patterns, checksums and Turkish NER](#5-finding-personal-data-patterns-checksums-and-turkish-ner)
- [6. Masking, pseudonymization, anonymization](#6-masking-pseudonymization-anonymization)
- [7. Retention and deletion: the math of forgotten copies and crypto-shredding](#7-retention-and-deletion-the-math-of-forgotten-copies-and-crypto-shredding)
- [8. Two engineering lenses: build time and run time](#8-two-engineering-lenses-build-time-and-run-time)
- [9. Practical output: inventory, pipeline and checklist](#9-practical-output-inventory-pipeline-and-checklist)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. One record, nine homes: the data multiplication map

Imagine handing a confidential paper letter to a corporate office. One clerk files the original in a locked archive. Another photocopies it for a colleague. A third scribbles an executive summary on a sticky note, while the network scanner caches a digital image in volatile memory. If the sender later invokes their legal right to deletion, burning the original archival copy accomplishes very little. You must locate the photocopy, scrape off the sticky note, and purge the scanner buffer.

Our support assistant operates precisely like that office. Let us trace a synthetic customer complaint containing a name, a phone number, and a national identity number through the system:

<!-- fig:p3_homes -->
<svg viewBox="0 0 720 366" role="img" aria-label="A single complaint in the source system, home 1, is deleted on request. Eight derived copies persist elsewhere, each with distinct ownership and retention schedules: 2 retrieval chunks, 3 embedding vectors, 4 prompt and completion logs at model provider, 5 semantic cache, 6 log and trace sinks, 7 analytical CSV exports, 8 fine-tuning/eval sets, 9 backups. Homes 2, 5, 6, 8, and 9 still held PII weeks after CRM deletion." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p3h-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">A Single Synthetic Complaint, Nine Storage Homes</text>
<rect x="16" y="112" width="170" height="84" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="26" y="122" width="22" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="37.0" y="133.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="56" y="135" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Source System</text>
<text x="26" y="160" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">complaint archive / CRM</text>
<text x="26" y="180" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">Deleted on request ✓</text>
<rect x="222" y="44" width="482" height="216" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="234" y="64" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">derived copies: each maintaining separate owners, access controls, and retention rules</text>
<line x1="186" y1="154" x2="220" y2="154" marker-end="url(#p3h-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="234" y="76" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="242" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="252.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<text x="268" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Passages</text>
<text x="242" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">retrieval index</text>
<text x="242" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI team</text>
<rect x="351" y="76" width="108" height="74" rx="6" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="359" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="369.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<text x="385" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Vectors</text>
<text x="359" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">invertible reps</text>
<text x="359" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI team</text>
<rect x="468" y="76" width="108" height="74" rx="6" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="476" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="486.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="502" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Prompts</text>
<text x="476" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sent to provider</text>
<text x="476" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TB2, vendor</text>
<rect x="585" y="76" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="593" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="603.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="619" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Cache</text>
<text x="593" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">semantic output</text>
<text x="593" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">platform, TTL?</text>
<rect x="234" y="164" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="242" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="252.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<text x="268" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Logs</text>
<text x="242" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">traces, exports</text>
<text x="242" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">platform, vendor</text>
<rect x="351" y="164" width="108" height="74" rx="6" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="359" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="369.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<text x="385" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Exports</text>
<text x="359" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">CSV for reporting</text>
<text x="359" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">analyst laptop</text>
<rect x="468" y="164" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="476" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="486.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="502" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Evals/Train</text>
<text x="476" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">realistic set</text>
<text x="476" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI team, indefinite</text>
<rect x="585" y="164" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="593" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="603.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">9</text>
<text x="619" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Backups</text>
<text x="593" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">nightly snapshot</text>
<text x="593" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ops, 35 days</text>
<rect x="16" y="272" width="22" height="14" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:4 3"/>
<text x="46" y="283" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">red, dashed: storage homes where PII quietly persisted weeks after source CRM deletion</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="294" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="313" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 DATA MULTIPLICATION AXIOM: ONE RECORD ENTERS, NINE PERSISTENCE TIERS EMERGE</text>
<text x="28" y="331" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Deleting a record from primary CRM does not purge it from the AI pipeline; chunks, embeddings, and logs become orphaned replicas.</text>
<text x="28" y="347" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">End-to-end erasure completeness requires centralized pseudonymization and cryptographic key revocation (crypto-shredding).</text>
</svg>

1. **Source System (Home 1):** The raw email inside the customer complaint database. In naive architectures, this is the only tier scrubbed upon user erasure.
2. **Retrieval Passages (Home 2):** Text chunks segmented for vector index ingestion.
3. **Embedding Vectors (Home 3):** High-dimensional floating-point projections ($\mathbb{R}^d$) generated by embedding models.
4. **Prompts and Completions (Home 4):** Context windows dispatched over external network links to the model serving endpoint (TB2).
5. **Semantic Cache (Home 5):** Prompt/response pairs stored in a cache such as Redis, so that similar questions are not regenerated. (The serving engine's prefix cache of KV tensors is a separate, in-memory copy with its own lifetime.)
6. **Log and Trace Sinks (Home 6):** OpenTelemetry spans persisted to cloud observability platforms for debugging.
7. **Analytical Exports (Home 7):** CSV or Parquet files downloaded to data science workstations for quarterly reporting.
8. **Fine-Tuning and Evaluation Datasets (Home 8):** Curated datasets used to train LoRA adapters or benchmark task accuracy.
9. **Backups (Home 9):** Immutable snapshot archives scheduled across all relational and vector databases.

Each storage domain maintains distinct access controls, administrators, and retention lifespans. The goal of an AI security architect is binding these nine domains to a single cryptographic customer key and unified deletion lifecycle.

---

## 2. Five data tiers, five security requirements

Data in enterprise AI architectures is not monolithic; each tier carries distinct governance imperatives:

| Data Tier | Support Assistant Context | Authored By | Primary Threat Vector | Mandatory Security Control |
| --- | --- | --- | --- | --- |
| **Pre-Training Data** | Foundational web-scale corpus | Third-party / Open Web | Unaudited backdoors, copyright violations | Model cards, rigorous red teaming (Part 4) |
| **Fine-Tuning Data** | 5,000 complaint-summary pairs | Internal ML engineers | Memorization of PII; data poisoning | **Pre-training PII cleansing**, $(\epsilon, \delta)$-DP-SGD, provenance hashes |
| **Evaluation Data** | 400 golden benchmark complaints | Security & QA teams | Training data contamination; PII leakage in CI | Synthetic data, strict CI/CD vault isolation |
| **Runtime User Data** | Employee prompts, session turns | Authenticated user | Cross-tenant data leakage; indefinite retention | User-scoped isolation, short TTL (24h), zero-data-retention (ZDR) |
| **Retrieval Data (RAG)** | Complaint archive, CRM, knowledge base | Heterogeneous third parties | Indirect prompt injection; unauthorized access | ACL-tagged passages, provenance trust labeling |

Two engineering realities dictate architecture:

**Fine-tuning data is the most persistent and hazardous.** Once personal data is baked into neural network weights ($W$) through gradient descent, it cannot be purged via an operational query. Academic *Machine Unlearning* techniques cannot yet provide legally sound, auditable guarantees. The engineering rule is categorical: **Unless you are prepared to discard and retrain the model from scratch upon an erasure request, never introduce unencrypted personal data into fine-tuning corpora.**

**Retrieval data carries the highest adversary exposure.** Complaint archives are authored by unverified external parties. This data simultaneously represents sensitive PII (requiring confidentiality) and an active injection vector (requiring integrity filtering).

---

## 3. Provenance and poisoning: influence functions and web-scale sabotage

**Data Provenance** is the auditable cryptographic ledger tracing data from its origin through ingestion, transformation, tokenization, and model training. Like cold-chain logistics in food safety, provenance guarantees that if a poisoned batch is discovered, the affected training runs can be identified and purged without discarding the entire enterprise corpus.

**Data Poisoning** occurs when an adversary injects targeted samples into training or retrieval corpora to manipulate downstream model behavior (`AML.T0020`, `AML.T0043`).

### Koh-Liang Influence Functions

In clean-label poisoning attacks, the adversary does not alter class labels; instead, they craft perturbations that manipulate empirical gradient updates. Koh and Liang (2017) formalized this through **Influence Functions**, which compute the derivative of a model's test loss with respect to upweighting an individual training sample $z = (x, y)$:

$$
I_{\text{up,loss}}(z, z_{\text{test}}) = -\nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta})
$$

Where $\nabla_\theta L(z, \hat{\theta})$ is the parameter gradient computed on training sample $z$, $H_{\hat{\theta}}^{-1}$ is the inverse Hessian of the empirical loss across the entire training dataset (capturing weight-space curvature), and $\nabla_\theta L(z_{\text{test}}, \hat{\theta})$ is the model gradient evaluated on the target test input.

This relationship explains the physics of clean-label poisoning:
- A negative $I_{\text{up,loss}}$ indicates that introducing $z$ into training **decreases** test loss on $z_{\text{test}}$ (the model learns that behavior).
- An adversary computes perturbations that maximize alignment with target test failures, allowing an imperceptibly small set of training samples to hijack model behavior on specific target prompts.

### How Many Samples Are Required to Poison a Model?

Empirical research confirms that poisoning requires remarkably few samples:
1. **Split-View Poisoning on Web Corpora:** Carlini et al. (2023) demonstrated that in public datasets distributed via URL indices, expired domains can be purchased for **\$60**, poisoning 0.01% of datasets like LAION-400M or COYO-700M.
2. **Backdoors in LLMs:** Anthropic and the UK AI Security Institute (October 2025) proved that across models from 600M to 13B parameters, just **250 poisoned documents** reliably insert persistent backdoors. For a 13B model, this accounted for a minuscule **0.00016%** of training tokens.
3. **PoisonedRAG:** Zou et al. (2024) demonstrated that injecting just **5 adversarial passages** per target query into a million-document vector store achieved a 90% attack success rate in manipulating retrieval responses.

Proven defensive controls:
- **Collection-Time SHA-256 Hashing:** Every ingested passage is cryptographically fingerprinted; if remote content mutates, the hash mismatch immediately quarantines the file.
- **Immutable Versioned Corpora:** Fine-tuning pipelines consume versioned artifacts locked via Git LFS or DVC rather than unpinned database snapshots.

---

## 4. Differential privacy and embedding inversion attacks

Securing AI data requires dismantling two pervasive engineering fallacies: "Embeddings are just numbers and therefore anonymous," and "Removing names prevents weight memorization."

### 4.1. Embedding Inversion Attacks (Vec2Text)

Many engineering teams assume that projecting customer complaints into 1536- or 3072-dimensional embedding vectors ($\mathbf{v} \in \mathbb{R}^d$) provides inherent anonymization.

This assumption is false. Morris et al. (2023) introduced **Vec2Text**, demonstrating that dense text embeddings are highly invertible. The method trains a model that proposes a text, re-embeds it, compares the result with the target vector and corrects its guess over several rounds:
- **92%** of 32-token inputs were reconstructed **exactly**.
- From a dataset of clinical notes, the same decoder recovered personal information such as patients' full names.

$$\text{Optimization Objective:} \quad \hat{x}^* = \arg\min_{\hat{x}} \|\text{Embed}(\hat{x}) - \mathbf{v}\|_2^2$$

> [!TIP]
> **Learning Check: Embeddings Are Not Anonymous**
> Classify and protect the vectors in your vector database **exactly like the text they were computed from**. An attacker who obtains the index and can query the same embedding model can reconstruct a large share of the original customer text.

### 4.2. Differential Privacy (DP-SGD)

The only mathematically proven guarantee against training data extraction and canary token memorization (Carlini et al.) is **Differential Privacy** (Dwork, 2006).

A randomized training mechanism $\mathcal{M}$ satisfies **$(\epsilon, \delta)$-Differential Privacy** if for any two adjacent datasets $D$ and $D'$ differing by at most one customer record ($\|D - D'\|_1 = 1$), and for all query outcomes $S$:

$$
\mathcal{P}(\mathcal{M}(D) \in S) \le e^\epsilon \cdot \mathcal{P}(\mathcal{M}(D') \in S) + \delta
$$

Where $\epsilon$ represents the privacy loss budget ($\epsilon \to 0$ guarantees that model weights are statistically independent of whether a specific customer's data was included), and $\delta$ represents the failure probability, bounded such that $\delta \ll 1/|D|$.

In deep learning, this is implemented via **DP-SGD** (Abadi et al., 2016):
1. **Per-Sample Gradient Clipping:** The $L_2$ norm of each sample's gradient is bounded by clipping threshold $C$:
   $$\bar{g}_t(x_i) = \frac{g_t(x_i)}{\max\left(1, \frac{\|g_t(x_i)\|_2}{C}\right)}$$
2. **Calibrated Gaussian Noise Addition:** Noise scaled to the clipping threshold is added to the *sum* of clipped gradients, and only then is the result averaged:
   $$\tilde{g}_t = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}_t(x_i) + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right) \right)$$

This noise masks the individual contribution of any single training example, enabling the network to learn generalized linguistic structures without memorizing private customer records.

### Step-by-step numerical walk: clipping and noise

Take a toy mini-batch of $B = 3$ per-sample gradients in two dimensions, a clipping threshold $C = 1$ and a noise multiplier $\sigma = 1$. The numbers are made up to show the mechanics; real gradients have millions of dimensions.

```text
Per-sample gradients and their L2 norms:
  g1 = (0.6, 0.8)    ‖g1‖ = sqrt(0.36 + 0.64) = 1.0
  g2 = (3.0, 4.0)    ‖g2‖ = sqrt(9 + 16)      = 5.0   ← one unusual record
  g3 = (0.0, 0.5)    ‖g3‖ = 0.5

Step 1: Clip each gradient (divide by max(1, ‖g‖ / C))
  ḡ1 = (0.6, 0.8) / max(1, 1.0) = (0.6, 0.8)
  ḡ2 = (3.0, 4.0) / max(1, 5.0) = (0.6, 0.8)   ← shrunk 5 times; it can no longer dominate
  ḡ3 = (0.0, 0.5) / max(1, 0.5) = (0.0, 0.5)   ← small gradients are left alone

Step 2: Sum, add noise with standard deviation σC = 1 per coordinate, then average
  sum   = (0.6 + 0.6 + 0.0, 0.8 + 0.8 + 0.5) = (1.2, 2.1)
  noise = (0.4, −0.7)                          ← one random draw from N(0, 1)
  g̃    = (1/3) × ((1.2, 2.1) + (0.4, −0.7)) = (1/3) × (1.6, 1.4) ≈ (0.53, 0.47)

Without DP: plain average = (1/3) × (3.6, 5.3) = (1.20, 1.77)   ← driven by g2 alone
```

After clipping, no single record can move the sum by more than $C = 1$ in norm, and the noise has a standard deviation of the same size, $\sigma C = 1$. That is what hides whether a given record was in the batch. The price is visible too: with three samples the noise is as large as the signal. Real DP-SGD uses batches of thousands, because the noise on the *average* shrinks as $\sigma C / B$ while the signal does not.

---

## 5. Finding personal data: patterns, checksums and Turkish NER

Detecting personal data (PII) before it contaminates vectors or weights requires a hybrid pipeline:

1. **Structured Identifiers:** Fixed-format strings (national IDs, IBANs, credit cards). Regular expressions surface candidate strings; **checksums** eliminate false positives.
2. **Contextual Entities:** Freeform text (names, home addresses, medical details). These demand domain-adapted **Named Entity Recognition (NER)** models.

### 5.1. National Identity Checksum Mathematics

Turkey's 11-digit national identity number (TCKN) enforces rigorous modular arithmetic across its final two digits:

$$
d_{10} = \Big( 7\,(d_1 + d_3 + d_5 + d_7 + d_9) - (d_2 + d_4 + d_6 + d_8) \Big) \bmod 10
$$

$$
d_{11} = \Big( \sum_{i=1}^{10} d_i \Big) \bmod 10
$$

Where $d_i$ denotes the digit at position $i$ from the left. Digit 10 is derived by multiplying the sum of odd-positioned digits by 7, subtracting the sum of even-positioned digits, and computing modulo 10. Digit 11 is the least significant digit of the sum of the first 10 digits.

### Step-by-Step Pedagogical Numerical Walk

Evaluating a synthetic 11-digit candidate with initial digits `1 2 3 4 5 6 7 8 9`:

```text
Digits:            d1 d2 d3 d4 d5 d6 d7 d8 d9
                   1  2  3  4  5  6  7  8  9

Step 1: Compute positional sums
  Odd positions:   d1 + d3 + d5 + d7 + d9 = 1 + 3 + 5 + 7 + 9 = 25
  Even positions:  d2 + d4 + d6 + d8      = 2 + 4 + 6 + 8     = 20

Step 2: Calculate 10th digit
  Raw calculation = 7 × 25 - 20 = 175 - 20 = 155
  d10 = 155 mod 10 = 5

Step 3: Calculate 11th digit
  Full sum = (1+2+3+4+5+6+7+8+9) + d10 = 45 + 5 = 50
  d11 = 50 mod 10 = 0

Result: Valid synthetic candidate = 12345678950 (Valid checksum)
Altering the last digit:           12345678951 (d11=1 != 0 -> Checksum failed!)
```

### Checksums Eliminate 99% of False Positives

A naive regex like `[1-9]\d{10}` flags every 11-digit order number, barcode, or invoice ID as sensitive PII. Because random 11-digit sequences have a $1/10$ chance of matching $d_{10}$ and an independent $1/10$ chance of matching $d_{11}$:

$$
P(\text{random string passes}) = \frac{1}{10} \times \frac{1}{10} = 0.01
$$

Checksum validation eliminates **99% of false alarms** at negligible CPU cost.

```python
import random
import re

def tckn_valid(s: str) -> bool:
    """Validates structural checksum of national ID candidates."""
    if not re.fullmatch(r"[1-9]\d{10}", s):
        return False
    d = [int(c) for c in s]
    d10 = (7 * (d[0] + d[2] + d[4] + d[6] + d[8]) - (d[1] + d[3] + d[5] + d[7])) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11

# Verification
print("12345678950 valid?:", tckn_valid("12345678950"))
print("12345678951 valid?:", tckn_valid("12345678951"))

# 1,000,000 random string benchmark
random.seed(42)
N = 1_000_000
hits = sum(tckn_valid(str(random.randint(10**10, 10**11 - 1))) for _ in range(N))
print(f"Pass rate of random 11-digit strings: {hits / N * 100:.2f}%")
```

Console output:

```text
12345678950 valid?: True
12345678951 valid?: False
Pass rate of random 11-digit strings: 0.99%
```

The simulation lands at 0.99% rather than exactly 1.00%: sampling noise around the expected $1/100$.

### 5.2. Linguistic Complexities in Turkish NER

Checksums cannot detect customer names or home addresses. NER models tasked with processing Turkish face distinct morphological hurdles:
- Suffix agglutination ("Ayşe'ye", "Ankara'daki") causes dictionary-based approaches to fail when apostrophes are omitted.
- Common given names double as common nouns ("Deniz", "Kaya", "Umut").
- Solution: Deploy transformer-based token classifiers (e.g., Turkish BERT NER models) chained directly downstream from regex filters.

---

## 6. Masking, pseudonymization, anonymization

The technical and legal implications of these transformation techniques are fundamentally distinct:

<!-- fig:p3_spectrum -->
<svg viewBox="0 0 720 328" role="img" aria-label="Spectrum from least protected to most protected in four columns. Raw value: 12345678950; readable by all; personal data. Masked: 123******50; raw value stored; personal data. Pseudonymized: ID generated via keyed HMAC; re-linkable only with the key; legally personal data under GDPR Recital 26. Anonymized: cannot be reversed by any reasonable means; excluded from privacy laws." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p3s-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">From Masking to Anonymization: The Technical Spectrum</text>
<line x1="112" y1="46" x2="700" y2="46" marker-end="url(#p3s-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="704" y="40" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">less identifiable →</text>
<text x="16" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">Example</text>
<text x="16" y="156" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">Reversible?</text>
<text x="16" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">Legal Status</text>
<rect x="112" y="58" width="140" height="182" rx="8" style="fill:var(--c-danger);fill-opacity:.07;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="182" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Raw Value</text>
<text x="182" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">12345678950</text>
<text x="182" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">anyone with</text>
<text x="182" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">access reads</text>
<text x="182" y="202" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:600">Personal Data</text>
<rect x="260" y="58" width="140" height="182" rx="8" style="fill:var(--c-warn);fill-opacity:.07;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="330" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Masked</text>
<text x="330" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">123******50</text>
<text x="330" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">raw value kept;</text>
<text x="330" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">display only</text>
<text x="330" y="202" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">Personal Data</text>
<rect x="408" y="58" width="140" height="182" rx="8" style="fill:var(--c-warn);fill-opacity:.07;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="478" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Pseudonymized</text>
<text x="478" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">ID_f0a155…</text>
<text x="478" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">re-linkable only</text>
<text x="478" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">with isolated key</text>
<text x="478" y="202" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">Personal Data</text>
<text x="478" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">(GDPR Rec. 26)</text>
<rect x="556" y="58" width="140" height="182" rx="8" style="fill:var(--c-success);fill-opacity:.07;stroke:var(--c-success);stroke-width:1.3"/>
<text x="626" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Anonymized</text>
<text x="626" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">a customer</text>
<text x="626" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">irreversible by</text>
<text x="626" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">reasonable means</text>
<text x="626" y="202" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:600">Outside Scope</text>
<text x="626" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">(Statutory Exemption)</text>
<text x="16" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Unless quasi-identifiers (neighborhood, timestamps, order sizes) are scrubbed, data remains legally identifiable.</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="256" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="275" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 PRIVACY SPECTRUM AXIOM: MASKED AND PSEUDONYMIZED DATA REMAIN PERSONAL DATA</text>
<text x="28" y="293" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Redaction and HMAC pseudonymization do not constitute legal anonymization; quasi-identifiers and re-identification risks persist.</text>
<text x="28" y="309" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">True anonymization demands mathematical guarantees resistant to Vec2Text inversion and verifiable differential privacy (DP-SGD).</text>
</svg>

- **Masking:** Obfuscates characters for UI rendering (`123******50`). The database persists raw plaintext. Legally personal data.
- **Pseudonymization:** Replaces identifiers with keyed HMAC digests:
  $$\text{Pseudonym} = \text{HMAC}_K(\text{NationalID})$$
  HMAC itself is one-way, but anyone holding key $K$ can recompute the pseudonym of a candidate ID and link the records again; with fewer than a billion valid national ID numbers, they can simply enumerate all of them. Under GDPR Article 4(5) and Recital 26, **pseudonymized data remains legally classified as personal data**. Turkey's KVKK reaches the same result from the other side: Article 3 defines anonymization as making data impossible to associate with an identified or identifiable person *even when matched with other data*, a bar that pseudonymized data does not clear.
- **Anonymization:** Eliminates re-identification risks under all reasonable methods, including statistical linkability attacks against secondary datasets. Sweeney (2000) demonstrated that 5-digit postal code, gender, and date of birth uniquely re-identify 87% of the US population. True anonymization requires scrubbing or perturbing **quasi-identifiers**.

---

## 7. Retention and deletion: the math of forgotten copies and crypto-shredding

The opening incident failed because of mathematical attrition across storage domains. Assume an enterprise architecture maintains $N$ independent storage copies of a customer's record. Let $q$ denote the probability that an erasure pipeline fails to purge an individual storage home (due to missing metadata, network timeouts, or uncataloged caches). The probability of **complete, faultless erasure** across all stores is:

$$
P_{\text{complete}} = (1 - q)^{N}
$$

Where $1 - q$ is the per-store erasure success rate, evaluated independently across all $N$ stores.

### Step-by-Step Pedagogical Numerical Walk

With a modest per-store failure rate of 5% ($q = 0.05$):

```text
Parameters:
  q = 0.05   (5% probability of omission per store)

Step 1: CRM database only (N=1)
  P_complete = (1 - 0.05)^1 = 0.9500   → 95.0% success

Step 2: CRM + Vector Store + Log Sinks (N=3)
  P_complete = (0.95)^3 = 0.95 × 0.95 × 0.95
             = 0.9025 × 0.95 = 0.8574   → 85.7% success

Step 3: All Nine Storage Domains (N=9)
  P_complete = (0.95)^9 ≈ 0.6302        → 63.0% success!
```

Across nine storage tiers, more than one in three deletion requests (37%) leaves behind an orphaned, unpurged customer record!

### Architectural Solution: Crypto-Shredding

Instead of issuing procedural delete sweeps across $N$ disparate APIs, we deploy **Crypto-Shredding**:
1. Customer records entering any storage tier are encrypted with a customer-specific Data Encryption Key ($K_{\text{user}}$).
2. When an erasure request arrives, the Key Management Service (KMS) permanently destroys $K_{\text{user}}$.
3. From that moment every ciphertext copy is unreadable: chunk text at rest, cache entries, trace payloads and, crucially, the immutable backups that no delete sweep can reach.

**What crypto-shredding does not reach.** It covers only copies stored as ciphertext under *your* key. Four homes on our map need something else:

- **Vectors inside the live ANN index (Home 3).** Similarity search needs the vectors in plaintext in memory, so the running index holds a searchable copy even when its disk snapshot is encrypted. Delete those rows by the customer's pseudonym and let the index compact or rebuild.
- **Prompts already sent to a model provider (Home 4).** They sit outside your KMS. Only a zero-data-retention agreement, checked in the contract and in the provider's settings, covers them.
- **Analyst exports (Home 7).** A CSV on a laptop is plaintext. Lower $N$ instead: export pseudonymized data only.
- **Fine-tuned weights (Home 8).** No key deletes what gradient descent absorbed. The only control is keeping personal data out of the training set (Section 2).

In the terms of the formula: crypto-shredding lowers $q$ sharply for the copies it covers, pseudonym-keyed deletes handle the live index, and the rest is solved by lowering $N$, that is, by never creating the copy. The weekly erasure test in the checklist below is what tells you whether all three still work.

---

## 8. Two engineering lenses: build time and run time

**Build Time:**
- Ingestion of poisoning payloads and weight memorization occurs here.
- Mitigations: Immutable SHA-256 provenance registries, frozen corpus versions, automated regex/NER cleaning prior to tokenization, and DP-SGD training constraints.

**Run Time:**
- Dynamic replication into semantic caches, trace sinks, and prompt buffers occurs here.
- Mitigations: Minimizing context PII, disabling raw payload persistence in OTel spans, enforcing 24-hour cache TTLs, and HMAC-based pseudonymization.

---

## 9. Practical output: inventory, pipeline and checklist

### 9.1. PII Transformation Pipeline

<!-- fig:p3_flow -->
<svg viewBox="0 0 720 426" role="img" aria-label="PII transformation pipeline. Incoming complaints are routed to regex with checksum for structured data, and Turkish NER for contextual data. Both feed the policy engine. Policy routes data: pseudonymized with ACLs to vector index, minimal PII to prompts, zero content to logs, and synthetic/DP data to training. Pseudonymization uses an isolated KMS key. Deletion operates globally by customer pseudonym." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p3f-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p3f-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="p3f-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">PII Ingestion Pipeline: Detect Once, Transform by Destination</text>
<rect x="16" y="120" width="112" height="64" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="72.0" y="140.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Complaint</text>
<text x="72.0" y="156.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">raw text</text>
<text x="72.0" y="172.5" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px">TB4: external</text>
<rect x="158" y="76" width="150" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="233.0" y="92.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Regex + Checksum</text>
<text x="233.0" y="108.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">National ID, IBAN (mod 97)</text>
<text x="233.0" y="122.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Credit Card (Luhn)</text>
<rect x="158" y="172" width="150" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="233.0" y="188.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Turkish NER Model</text>
<text x="233.0" y="204.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">names, addresses, orgs</text>
<text x="233.0" y="220.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">morphology-aware</text>
<path d="M128 152 H142 V104 H156" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M128 152 H142 V200 H156" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="338" y="112" width="132" height="80" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="404.0" y="136.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Policy Engine</text>
<text x="404.0" y="154.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">rules by destination:</text>
<text x="404.0" y="170.5" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10.5px">mask / pseudonymize</text>
<path d="M308 104 H322 V140 H336" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M308 200 H322 V164 H336" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="510" y="52" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="522" y="73" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Retrieval Store</text>
<text x="522" y="90" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pseudonymized + ACLs</text>
<path d="M470 152 C 490 152, 490 77, 508 77" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="510" y="116" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="522" y="137" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Inference Prompt</text>
<text x="522" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">minimal task-specific PII</text>
<path d="M470 152 C 490 152, 490 141, 508 141" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="510" y="180" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="522" y="201" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Logs &amp; Telemetry</text>
<text x="522" y="218" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">zero payloads; salted token hashes</text>
<path d="M470 152 C 490 152, 490 205, 508 205" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="510" y="244" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="522" y="265" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Training Corpus</text>
<text x="522" y="282" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">synthetic only / DP-SGD</text>
<path d="M470 152 C 490 152, 490 269, 508 269" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<path d="M148 277 V321 A85.0 7 0 0 0 318 321 V277" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<ellipse cx="233.0" cy="277" rx="85.0" ry="7" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="233.0" y="300.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Key Management</text>
<text x="233.0" y="315.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">KMS / HMAC Key Vault</text>
<path d="M404 192 V299 H320" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="410" y="250" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pseudonyms</text>
<rect x="510" y="312" width="194" height="30" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<text x="607" y="331" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Crypto-shred: erase key, ciphertext dies</text>
<path d="M318 312 H470 V327 H508" marker-end="url(#p3f-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5;stroke-dasharray:5 4"/>
<!-- Bottom Educational Banner -->
<rect x="16" y="354" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="373" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 ARCHITECTURAL SOLUTION: INSTANT PURGING VIA CRYPTO-SHREDDING</text>
<text x="28" y="391" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Instead of executing complex cascading delete queries across nine decoupled systems, each identity is assigned a dedicated key.</text>
<text x="28" y="407" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Destroying the key makes every encrypted copy unreadable, backups included; live index rows are deleted by pseudonym.</text>
</svg>

### 9.2. Data Lifecycle Verification Checklist

1. **Nine Homes Inventory:** Every storage sink harboring customer copies (vectors, caches, traces, backups) is cataloged with assigned operational owners.
2. **Cryptographic Provenance Fingerprinting:** Ingested corpora are hashed with SHA-256 upon ingestion and tracked via immutable version manifests.
3. **Pre-Training PII Elimination:** Raw customer PII is strictly excluded from fine-tuning corpora; models are trained under $(\epsilon, \delta)$-DP-SGD constraints.
4. **Dual-Layer PII Identification:** Structured IDs are validated via algorithmic checksums; contextual entities are detected via domain-adapted NER.
5. **Vector Classification:** Vector databases and semantic caches are governed under identical encryption, access control, and classification rules as plaintext PII.
6. **Crypto-Shredding Architecture:** Customer records are encrypted with per-user keys; deletion requests destroy the key, which neutralizes every ciphertext copy including backups; live index rows are deleted by pseudonym, and provider retention is covered by a zero-data-retention agreement.
7. **Automated Erasure Verification:** Weekly synthetic customer records are scheduled for automated end-to-end erasure, verifying that all nine tiers return zero hits.

### What questions can the reader now answer independently?

- Why does scrubbing an entry from an operational CRM fail to guarantee complete erasure in an AI assistant?
- Why do Vec2Text inversion attacks compel security teams to treat dense vector embeddings with the same rigor as raw PII?
- How does the Koh-Liang influence function ($I_{\text{up,loss}}$) explain clean-label poisoning without altering class labels?
- By what mathematical mechanism does modular checksum validation eliminate 99% of regex false positives?
- Why does the probability of complete deletion ($(1-q)^N$) collapse across multi-tiered architectures, and how does crypto-shredding solve it?

---

## The whole story in six lines

- AI data multiplies: a single complaint splinters across nine distinct storage domains with disparate lifespans.
- Dense embeddings can be inverted (Vec2Text recovered 92% of 32-token inputs exactly), so vectors are protected like the text they came from.
- Personal data cannot be excised from fine-tuned weights; training corpora must be sanitized prior to gradient updates.
- Clean-label data poisoning operates via influence functions; 250 poisoned samples reliably compromised a 13B LLM.
- Checksums eliminate 99% of regex false alarms on structured IDs, while contextual entities require domain-adapted NER.
- Erasure completeness degrades as $(1-q)^N$; lower $N$, crypto-shred what is encrypted under your key, delete live index rows by pseudonym, and test deletion weekly.

---

## Glossary

- **Data Provenance:** The auditable cryptographic chain of custody documenting a dataset's origin, mutations, and approval signatures.
- **Influence Function ($I_{\text{up,loss}}$):** A measure of how upweighting an individual training sample shifts the empirical model parameters and test loss.
- **Vec2Text (Embedding Inversion):** An iterative optimization technique that reconstructs original plaintext from dense embedding vectors.
- **Differential Privacy (DP-SGD):** A training algorithm that injects calibrated Gaussian noise into clipped gradients, bounding private data extraction.
- **Checksum:** Algorithmic verification digits derived from an identifier's payload that deterministically eliminate random false alarms.
- **Quasi-Identifier:** Attributes that do not identify an individual alone, but uniquely distinguish them when linked with external data.
- **Pseudonymization:** Replacing direct identifiers with keyed cryptographic tokens; legally classified as personal data while the key persists.
- **Crypto-Shredding:** Irreversibly purging customer data across all distributed stores and backups by deleting the unique encryption key.
- **Semantic Cache:** A storage layer preserving prompt/completion pairs to accelerate inference and suppress redundant token expenditures.
- **Model Card:** An architectural specification documenting model training corpora, intended use cases, and evaluated safety boundaries.

---

## Going deeper

- [K1] N. Carlini et al., [*Poisoning Web-Scale Training Datasets is Practical*](https://arxiv.org/abs/2302.10149) (2023): Split-view web poisoning.
- [K2] Anthropic, UK AI Security Institute, and Alan Turing Institute, [*A small number of samples can poison LLMs of any size*](https://www.anthropic.com/research/small-samples-poison) (October 2025).
- [K3] P. W. Koh and P. Liang, [*Understanding Black-box Predictions via Influence Functions*](https://arxiv.org/abs/1703.04730) (ICML 2017): Mathematical foundations of training sample attribution.
- [K4] J. X. Morris et al., [*Text Embeddings Reveal (Almost) As Much As Text*](https://arxiv.org/abs/2310.06816) (EMNLP 2023): The Vec2Text inversion framework.
- [K5] M. Abadi et al., [*Deep Learning with Differential Privacy*](https://arxiv.org/abs/1607.00133) (ACM CCS 2016): Formulation of DP-SGD and gradient clipping.
- [K6] W. Zou et al., [*PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation*](https://arxiv.org/abs/2402.07867) (2024): Targeted knowledge corruption.
- [K7] L. Sweeney, *Simple Demographics Often Identify People Uniquely*, Carnegie Mellon University Data Privacy Working Paper 3 (2000).
- [K8] ISO/IEC, [*27701:2025 Privacy Information Management System*](https://www.iso.org/standard/27701) and NIST, [*AI 600-1*](https://doi.org/10.6028/NIST.AI.600-1) (Data Privacy Profile).
- [K9] In this series: [Secure AI by Design (4): From Model to Dependency — Securing the AI Supply Chain](post.html?slug=securing-the-ai-supply-chain).
