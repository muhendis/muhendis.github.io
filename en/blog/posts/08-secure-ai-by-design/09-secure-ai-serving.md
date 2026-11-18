When the penetration test report from an independent cybersecurity firm arrived on the executive desk, leadership was stunned. The audit uncovered three critical-severity vulnerabilities in the corporate AI assistant, none of which had anything to do with model weights or training datasets:

1. **Exposed Model Server Ports:** The vLLM inference server listening on port 8000 was directly accessible from the entire internal corporate network. The engineering team had started the daemon with the `--api-key` flag; consequently, standard requests to `/v1/chat/completions` returned a 401 Unauthorized as expected. However, in the vLLM architecture, the API key protects only specific URL route prefixes. Administrative and auxiliary endpoints—including `/invocations`, `/tokenize`, `/metrics`, `/pause` and `/update_weights`—remained completely unauthenticated.
2. **Unshielded Fallback Routing:** On the AI Gateway, engineers had configured an automatic fallback route designed to redirect traffic to a commercial hosted cloud model whenever the on-premise GPU cluster experienced heavy load. Input and output guardrails, however, were attached exclusively to the primary on-premise pipeline. Whenever internal GPUs became saturated—or when an adversary deliberately flooded the servers with high-concurrency requests—traffic shifted to the cloud provider, bypassing all security inspections entirely.
3. **Streaming Data Leakage:** The output guardrail was configured in `post_call` mode across a streaming Server-Sent Events (SSE) endpoint. When the model emitted a response disclosing another customer's national ID and private phone number, the guardrail detected the violation; yet by the time the inspection completed, the sensitive tokens had already been rendered onto the employee's browser terminal.

Every control developed across the preceding eight articles was in place; yet adversaries did not need to defeat any of them. All they had to do was **find a path that bypassed them**.

The serving layer exists precisely to enforce this guarantee: Ensuring that every inference request, across every routing path, under every failure mode, traverses the identical security controls, and that no backend infrastructure can be reached by circumventing the gateway.

In this guide, we engineer enterprise AI serving infrastructure from the perspective of a Principal AI Security Architect. We isolate model servers (vLLM) and the AI Gateway (LiteLLM) from hardware silicon to network ingress. We examine **NVIDIA H100 Confidential Computing**: SPDM (Security Protocol and Data Model) attestation, encrypted transfers between the CPU TEE and the GPU, and where the **PCIe IDE (Integrity and Data Encryption)** standard fits. We close the timing side channel that vLLM's prefix cache opens between tenants with per-tenant cache salts. We resolve the tension between Time-to-First-Token (TTFT) latency and leakage windows with a holdback window and **speculative rollback**, and cover the documented inference-time side channels.

> **What we carry over from the previous part:** [Part 6](post.html?slug=layered-guardrails) structured layered detection pipelines and defined failure behavior on a per-operation basis. [Part 8](post.html?slug=least-privilege-for-ai-agents) positioned the executor as the sole gateway to external mutations. Here we establish the AI Gateway as the sole legitimate path to model inference.

> **By the end of this article**
>
> - You will be able to construct clear security boundaries and responsibility matrices between the AI Gateway, security services, and model servers.
> - You will be able to explain what GPU confidential computing protects (attestation, SPDM, encrypted transfers), what it does not, and where PCIe IDE fits.
> - You will be able to isolate unauthenticated vLLM endpoints using network policies and reverse proxies.
> - You will be able to close the prefix-cache timing channel between tenants with a gateway-set, per-tenant `cache_salt`.
> - You will be able to implement speculative rollback buffers on streaming endpoints, catching structured PII before display at a known, small cost in TTFT.

**In this article**

- [1. Three components, three responsibilities: Gateway, security service, and model server](#1-three-components-three-responsibilities-gateway-security-service-and-model-server)
- [2. Hardware security and Confidential AI: H100 TEE, SPDM, and PCIe IDE](#2-hardware-security-and-confidential-ai-h100-tee-spdm-and-pcie-ide)
  - [2.1. Hardware root of trust and SPDM firmware attestation](#21-hardware-root-of-trust-and-spdm-firmware-attestation)
  - [2.2. Encrypted transfers on the PCIe bus and what stays unprotected](#22-encrypted-transfers-on-the-pcie-bus-and-what-stays-unprotected)
- [3. Provider-agnostic reference architecture: LiteLLM in front of vLLM](#3-provider-agnostic-reference-architecture-litellm-in-front-of-vllm)
- [4. Network, identity, secrets, and model server endpoint inventory](#4-network-identity-secrets-and-model-server-endpoint-inventory)
  - [4.1. What an API key fails to cover: The vLLM attack surface](#41-what-an-api-key-fails-to-cover-the-vllm-attack-surface)
  - [4.2. Multi-tenant prefix caching: timing leaks and cache salts](#42-multi-tenant-prefix-caching-timing-leaks-and-cache-salts)
- [5. Streaming guardrails: Latency budgets and speculative rollback](#5-streaming-guardrails-latency-budgets-and-speculative-rollback)
  - [5.1. The streaming latency vs leakage window dilemma](#51-the-streaming-latency-vs-leakage-window-dilemma)
  - [5.2. Sliding window buffering and speculative cancellation algorithm](#52-sliding-window-buffering-and-speculative-cancellation-algorithm)
- [6. Inference-time side-channel attacks](#6-inference-time-side-channel-attacks)
- [7. Caching, retries, fallback routing, and guard calls](#7-caching-retries-fallback-routing-and-guard-calls)
- [8. Rate limits, quotas, and budgets as security controls](#8-rate-limits-quotas-and-budgets-as-security-controls)
- [9. Practical output: Deployment architecture, acceptance test suite, and Python security engine](#9-practical-output-deployment-architecture-acceptance-test-suite-and-python-security-engine)
  - [9.1. Deployment acceptance test matrix](#91-deployment-acceptance-test-matrix)
  - [9.2. Production-grade Python serving security engine](#92-production-grade-python-serving-security-engine)
- [Pedagogical checkpoints](#pedagogical-checkpoints)
  - [Self-check](#self-check)
  - [Thought experiment](#thought-experiment)
  - [Disambiguation](#disambiguation)
  - [What questions can the reader now answer independently?](#what-questions-can-the-reader-now-answer-independently)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. Three components, three responsibilities: Gateway, security service, and model server

Consider a fortified treasury: At the entrance stands a security reception desk, behind it an automated inspection checkpoint, and in the secure core sits a reinforced vault. The reception desk verifies identity credentials. The inspection checkpoint screens individuals for contraband. The vault performs one task: opening exclusively for individuals cleared by both reception and inspection. No direct doorway connects the public street to the vault.

An enterprise AI serving architecture enforces the identical tri-partite separation:

| Component | Primary Responsibility | Fatal Antipattern |
| :--- | :--- | :--- |
| **AI Gateway** | Authenticate clients; enforce quotas and rate limits; route requests; trigger guardrails across all paths; manage mTLS | Possessing an unshielded fallback or bypass route that circumvents security |
| **Security Service** | Execute input/output detectors; evaluate policy engines; return deterministic allow/deny verdicts | Operating as an optional hook; failing open on high-risk operations upon timeout |
| **Model Server** | Execute high-performance GPU tensor inference exclusively for requests dispatched by the gateway | Exposing management, metric, or inference ports directly to external or LAN networks |

<!-- fig:p9_deploy -->
<svg viewBox="0 0 720 460" role="img" aria-label="Deployment diagram. In the client zone, the chat UI communicates with the orchestrator and executor. The orchestrator calls the gateway over mutual TLS. The gateway authenticates, enforces quotas, routes, calls the security service on every path including fallbacks, and logs versions; it retrieves secrets from a secret manager. The model zone is a private subnet where vLLM nodes are accessible exclusively from the gateway over mTLS; internal ports for KV transfer and data parallelism are open only between nodes; endpoints outside /v1 are not routed. Direct paths from application to vLLM are blocked by network policy. Hosted fallbacks are reached solely through the gateway with identical guardrails." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p9d-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p9d-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
<marker id="p9d-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
<marker id="p9d-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Deployment: single path to model, shielded across all routes</text>
<rect x="16" y="44" width="160" height="290" rx="8" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="26" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">client zone</text>
<rect x="192" y="44" width="196" height="290" rx="8" style="fill:var(--c-warn);fill-opacity:.05;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="202" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">gateway zone</text>
<rect x="404" y="44" width="300" height="186" rx="8" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.3;stroke-dasharray:5 4"/>
<text x="414" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">model zone (private subnet)</text>
<rect x="28" y="76" width="136" height="50" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="97.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Chat UI</text>
<text x="96.0" y="113.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">no images/links</text>
<rect x="28" y="150" width="136" height="64" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="170.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Orchestrator</text>
<text x="96.0" y="186.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">executor, provenance</text>
<text x="96.0" y="202.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">tags (Part 8)</text>
<line x1="96" y1="126" x2="96" y2="148" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="204" y="76" width="172" height="138" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="290" y="98" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Gateway (LiteLLM)</text>
<text x="216" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">authenticate client</text>
<text x="216" y="136" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">quota: tokens, cost</text>
<text x="216" y="152" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">route + fallbacks</text>
<text x="216" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">guards on every path</text>
<text x="216" y="184" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">log route, versions</text>
<text x="216" y="200" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">mTLS on ingress/egress</text>
<line x1="164" y1="182" x2="202" y2="182" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="204" y="240" width="172" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="290.0" y="256.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Security Service</text>
<text x="290.0" y="272.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">detectors, policy,</text>
<text x="290.0" y="288.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">guard SLM (Part 6)</text>
<line x1="276" y1="214" x2="276" y2="238" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="304" y1="238" x2="304" y2="216" marker-end="url(#p9d-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M28 253 V291 A68.0 7 0 0 0 164 291 V253" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<ellipse cx="96.0" cy="253" rx="68.0" ry="7" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="96.0" y="273.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Secrets Manager</text>
<text x="96.0" y="288.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">provider keys</text>
<line x1="164" y1="270" x2="202" y2="202" marker-end="url(#p9d-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="420" y="84" width="128" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="484.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">vLLM node</text>
<text x="484.0" y="116.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">/v1 only</text>
<text x="484.0" y="132.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">from gateway</text>
<rect x="562" y="84" width="128" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="626.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">vLLM node</text>
<text x="626.0" y="116.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">/v1 only</text>
<text x="626.0" y="132.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">from gateway</text>
<line x1="548" y1="112" x2="562" y2="112" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="555" y="156" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">KV :14579, DP :29500 inter-node only</text>
<line x1="376" y1="112" x2="418" y2="112" marker-end="url(#p9d-o)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="420" y="176" width="270" height="40" rx="6" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="430" y="192" text-anchor="start" style="fill:var(--c-text);font-size:11px">/invocations, /score, /pooling, /pause,</text>
<text x="430" y="206" text-anchor="start" style="fill:var(--c-text);font-size:11px">/update_weights, gRPC, dev mode: disabled</text>
<path d="M164 204 H184 V338 H400 V196 H416" marker-end="url(#p9d-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="420" y="322" text-anchor="start" style="fill:var(--c-danger);font-size:11px">app → vLLM direct: blocked by network policy</text>
<rect x="420" y="250" width="156" height="50" rx="2" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:2"/>
<text x="498.0" y="272.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Hosted fallback</text>
<text x="498.0" y="287.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">external provider</text>
<path d="M376 180 C 398 180, 398 275, 418 275" marker-end="url(#p9d-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="590" y="270" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">same guards</text>
<text x="590" y="285" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">as primary</text>
<rect x="16" y="346" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.3"/>
<text x="28" y="365" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Every model call—including guard SLMs and fallbacks—traverses the gateway: authenticated, checked, counted.</text>
<text x="28" y="381" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Acceptance tests S-1 through S-10 verify every boundary in this architecture.</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="388" width="688" height="60" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="407" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 SERVING SECURITY: NEVER EXPOSE MODEL SERVERS DIRECTLY</text>
<text x="28" y="425" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Model server endpoints (/invocations, /metrics) lack authentication and must reside behind an allowlisting Gateway via mTLS on a private network.</text>
<text x="28" y="440" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Confidential AI (NVIDIA H100 TEE / AMD SEV-SNP) cryptographically shields model weights and KV memory even from the host hypervisor operator.</text>
</svg>

---

## 2. Hardware security and Confidential AI: H100 TEE, SPDM, and PCIe IDE

### 2.1. Hardware root of trust and SPDM firmware attestation

In a conventional cloud deployment, whoever controls the hypervisor or the host OS can read what a guest sends to its GPU: model weights, incoming prompts and KV cache contents all pass through memory the host can inspect.

**Confidential computing** removes the host from the trusted base. The workload runs in a confidential VM (a CPU TEE: AMD SEV-SNP or Intel TDX), and since the H100 generation NVIDIA GPUs can be switched into a confidential mode that extends that TEE to the accelerator. Two mechanisms carry the design:

1. **Attestation.** The GPU has a hardware root of trust and a device identity certificate that chains to NVIDIA's certificate authority. On request it produces a signed set of measurements of its firmware and configuration, the *attestation report*. The report is checked against reference values, through the NVIDIA Remote Attestation Service (NRAS) or locally in air-gapped environments.
2. **An SPDM session.** The GPU driver *inside* the confidential VM opens a secure session with the GPU using SPDM (Security Protocol and Data Model, the DMTF standard for device authentication and key exchange) and derives the keys for everything that will cross the bus.

The control that matters for a deployment is **attestation-gated key release**: the key that decrypts the model weights is handed out by your key service only after both attestations, CPU TEE and GPU, have verified. A node with tampered firmware never receives the key.

### 2.2. Encrypted transfers on the PCIe bus and what stays unprotected

The bus between host memory and the GPU is the obvious tap: memory shared with the device is visible to the hypervisor, and someone with physical access can place an interposer in the PCIe slot.

On the H100 the protection is implemented by the driver and the GPU, with the SPDM session keys. Data moves through an **encrypted bounce buffer** in shared memory, and command buffers and CUDA kernels are encrypted and signed before they cross the bus. The host and the interposer see ciphertext.

Three limits are part of the design and belong in your threat model:

- **GPU memory itself is not encrypted.** NVIDIA's own description is that the on-package HBM is considered secure against everyday physical attack tools and is not encrypted. The boundary is the GPU package, not a cipher over VRAM.
- **PCIe IDE is the standard, not the H100 mechanism.** Link-level encryption of the bus is standardized as PCIe IDE (Integrity and Data Encryption, AES-GCM), and TDISP defines how a TEE takes ownership of a device interface. Platforms built on them (Intel TDX Connect, AMD SEV-TIO) move the protection from the driver into the link. If a vendor says "PCIe IDE", check which generation of CPU, GPU and firmware you actually run.
- **Confidential mode is not a tenancy boundary.** It protects one workload from the infrastructure underneath it. It does nothing about two tenants sharing one model server process; that is Section 4.2.

---

## 3. Provider-agnostic reference architecture: LiteLLM in front of vLLM

Our production reference architecture pairs the open-source **vLLM** inference engine with **LiteLLM Proxy** acting as the unified AI Gateway.

LiteLLM supports custom guardrails across three operational execution modes:

| Guardrail Mode | Execution Phase | Can Mutate Payload? | Can Terminate Call? |
| :--- | :--- | :--- | :--- |
| `pre_call` | Prior to model dispatch on request payload | Yes (PII masking, context truncation) | Yes (returns 403 / 400 per PDP policy) |
| `during_call` | Concurrently with model inference | No (observability / auditing only) | Yes (drops response upon completion) |
| `post_call` | Upon model response generation | Non-streaming: Yes; **Streaming: No** | Non-streaming: Yes; **Streaming: No** |

The inability of `post_call` hooks to block streaming tokens is an architectural constraint: Once the initial token is dispatched to the client socket, disclosure has occurred. Section 5 resolves this through speculative buffering.

Sample secure gateway configuration:

```yaml
model_list:
  - model_name: enterprise-assistant
    litellm_params:
      model: hosted_vllm/mistral-7b-instruct-v0.3
      api_base: "https://vllm-node.internal:8443/v1"
      ssl_verify: "/etc/ssl/certs/internal-ca.crt"
  - model_name: enterprise-assistant-fallback
    litellm_params:
      model: openai/gpt-4o
      api_key: "os.environ/SECURE_FALLBACK_KEY"

router_settings:
  fallbacks: [{"enterprise-assistant": ["enterprise-assistant-fallback"]}]

guardrails:
  - guardrail_name: input-pep-policy
    litellm_params:
      guardrail: enterprise_security.InputPolicy
      mode: pre_call
      default_on: true
  - guardrail_name: streaming-speculative-guard
    litellm_params:
      guardrail: enterprise_security.StreamingSpeculativeGuard
      mode: post_call          # on streams this hook buffers chunks itself (Section 5)
      default_on: true
```

---

## 4. Network, identity, secrets, and model server endpoint inventory

### 4.1. What an API key fails to cover: The vLLM attack surface

vLLM's official documentation highlights an important operational boundary: The `--api-key` flag protects only `/v1`, `/v2`, `/inference`, and `/cohere` URL route prefixes. Endpoints residing **outside** these prefixes remain unauthenticated:

- **Unauthenticated Inference Paths:** `/invocations`, `/pooling`, `/classify`, `/score`, `/rerank`. An attacker can invoke model forward passes without supplying an API token.
- **Operational Routes:** `/pause`, `/resume`, `/abort_requests`, `/update_weights` and others. An attacker can stall inference, abort other users' requests or replace model weights.
- **Utility Routes:** `/metrics`, `/tokenize`, `/detokenize`, `/health`, `/version`. They leak load, model identity and tokenizer behavior.
- **Internal Distributed Ports:** The data-parallel master port (default `:29500`) and the KV cache transfer port (`--kv-port`, default `:14579`). vLLM's documentation describes this inter-node communication as insecure by default: unencrypted and unauthenticated, to be isolated on its own network. The optional gRPC port has no authentication or encryption either.

Defense: vLLM is isolated inside a private subnet (`subnet-private-inference`). Kubernetes `NetworkPolicy` rules restrict port `:8443` access strictly to the Gateway pod via mTLS, while blocking external access to auxiliary ports.

```text
Enterprise Inference Endpoint Inventory and Policy Matrix:
--------------------------------------------------------------------------------
Endpoint / Port         Access Authorization  Enforced Security Control
--------------------------------------------------------------------------------
vllm :8443 /v1/*        Gateway ONLY          mTLS + API Key + NetworkPolicy
vllm :8443 /invocations BLOCKED               Reverse proxy drops with 404
vllm :8443 /pause, ...  BLOCKED               Reverse proxy drops with 404
vllm :8443 /metrics     Prometheus ONLY       mTLS + ServiceAccount RBAC
vllm :14579 (KV Trans)  vLLM Nodes ONLY       Isolated subnet, node-to-node only
gateway :443 /v1/*      Clients / Apps        OAuth2 JWT + Distributed Rate Limits
--------------------------------------------------------------------------------
```

### 4.2. Multi-tenant prefix caching: timing leaks and cache salts

To avoid recomputing shared prompt prefixes, vLLM's automatic prefix caching keeps KV cache blocks and reuses them across requests. Each block is identified by a hash of its own tokens chained with the hash of the block before it, so a request reuses a block only if its prompt is **identical up to and including that block**.

That rules out one fear and leaves a subtler one. A request cannot be handed another tenant's *different* context: a hit requires the same tokens, and the same tokens produce the same KV tensors, so nothing new is revealed by the content. What leaks is the **hit itself**. A cached prefix skips prefill, so the response starts measurably sooner, and Time to First Token becomes an oracle:

1. The attacker sends a candidate prefix and measures TTFT.
2. A fast response means some other user sent exactly that prefix recently.
3. Extending the guess block by block reconstructs a system prompt or the beginning of someone else's document.

vLLM documents this as a timing side channel (CVE-2025-46570) and cites research in which a prefix of just 8 tokens was distinguishable with a ROC AUC of 0.99.

The mitigation is built in. A request may carry a `cache_salt`; vLLM mixes it into the hash of the first block, so only requests with the same salt can share cached prefixes:

$$
h_1 = H(\text{cache\_salt} \parallel \text{tokens}_{\text{block }1}), \qquad h_i = H(h_{i-1} \parallel \text{tokens}_{\text{block }i})
$$

Three rules make it a control rather than a hint:

- **The gateway sets the salt, never the client.** It is derived per tenant from the authenticated identity, for example $\text{cache\_salt} = \text{HMAC}_{K}(\text{tenant\_id})$, and any client-supplied value is overwritten.
- **Salts are secret and random-looking** (256 bits), not guessable tenant IDs.
- **A salt is not an isolation boundary.** vLLM's own guidance is that it does not isolate tenants that share a server process; for tenants who must not influence each other at all, run one instance per tenant.

The price is cache efficiency: an identical system prompt is now computed once per tenant instead of once overall.

---

## 5. Streaming guardrails: Latency budgets and speculative rollback

### 5.1. The streaming latency vs leakage window dilemma

In consumer-facing AI applications, Time-to-First-Token (TTFT) is paramount: Users expect visible output within 500 ms. This generates a fundamental architectural tension:
- **Zero Buffering (Latency = 0, Leakage Window = 100%):** Tokens are written to the socket as they generate. Malicious or sensitive outputs appear instantly on the user terminal; post-hoc filters cannot retract them.
- **Full Response Buffering (Latency = 100%, Leakage Window = 0):** The entire completion is buffered and inspected prior to transmission. TTFT degrades by 3–5 seconds, destroying conversational streaming.

When generation speed is $v$ chars/sec and the holdback buffer is $W$ chars, the added initial latency ($\Delta t$) and the condition for complete containment are given by:

$$
\Delta t = \frac{W}{v}, \qquad \text{contained without disclosure} \iff s \le W
$$

where $s$ denotes the number of characters the detector needs to see to identify an infraction ($s_{\text{TCKN}} = 11$, $s_{\text{IBAN}} \approx 26$, $s_{\text{API\_KEY}} \approx 40$).

```text
Performance analysis for a model generating 40 tokens/sec (v ≈ 160 chars/sec):
--------------------------------------------------------------------------------
W = 64 characters (Sufficient margin for TCKNs, IBANs, and API tokens):
    Added Latency: Δt = 64 / 160 = 0.40 seconds (Imperceptible to users)
    Containment: Structured PII is intercepted BEFORE reaching the client screen.

W = 200 characters (Sentence-level prompt injection and profanity detection):
    Added Latency: Δt = 200 / 160 = 1.25 seconds.
--------------------------------------------------------------------------------
```

<!-- fig:p9_stream -->
<svg viewBox="0 0 720 356" role="img" aria-label="Timeline comparing three streaming strategies: post-call audit, full buffering, and 200-character holdback window. PII generated at character 300 is caught before display only by the holdback window." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Streaming Output: Three Defense Strategies for a 500-Character Response</text>
<!-- Time Axis -->
<line x1="16" y1="52" x2="704" y2="52" style="stroke:var(--c-border);stroke-width:1"/>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">t = 0.0s (start)</text>
<text x="430" y="44" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-weight:700;font-family:var(--font-mono)">t = 1.9s (violation generated)</text>
<text x="704" y="44" text-anchor="end" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">t = 3.1s (500 characters)</text>

<!-- Strategy 1 -->
<rect x="16" y="64" width="688" height="60" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="28" y="84" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">1. Post-Call Audit (Zero Delay)</text>
<rect x="28" y="94" width="664" height="14" rx="3" style="fill:var(--c-danger);fill-opacity:.18;stroke:var(--c-danger);stroke-width:1"/>
<circle cx="430" cy="101" r="5" style="fill:var(--c-danger)"/>
<text x="28" y="118" text-anchor="start" style="fill:var(--c-danger);font-size:10px;font-weight:600">Delay: 0s · Checked at end · PII already rendered to user! ❌</text>

<!-- Strategy 2 -->
<rect x="16" y="134" width="688" height="60" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="28" y="154" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2. Full Buffering (Max Security)</text>
<rect x="28" y="164" width="664" height="14" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="650" y="164" width="42" height="14" rx="3" style="fill:var(--c-warn);fill-opacity:.3"/>
<text x="28" y="188" text-anchor="start" style="fill:var(--c-warn);font-size:10px;font-weight:600">Delay: 3.1s · Nothing shown until stream finishes · TTFT ruined! ❌</text>

<!-- Strategy 3 -->
<rect x="16" y="204" width="688" height="64" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.5"/>
<text x="28" y="224" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">3. Sliding Holdback Window (Optimal)</text>
<rect x="28" y="234" width="402" height="14" rx="3" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1"/>
<rect x="430" y="234" width="60" height="14" rx="3" style="fill:var(--c-warn);fill-opacity:.4;stroke:var(--c-warn);stroke-width:1"/>
<line x1="490" y1="228" x2="490" y2="254" style="stroke:var(--c-danger);stroke-width:2.5;stroke-dasharray:3 2"/>
<text x="28" y="260" text-anchor="start" style="fill:var(--c-success);font-size:10px;font-weight:700">Delay: 1.25s (200 chars) · PII intercepted before client display! ✅</text>
<rect x="430" y="208" width="262" height="18" rx="4" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1"/>
<text x="561" y="221" text-anchor="middle" style="fill:var(--c-danger);font-size:9.5px;font-weight:700">🛡️ SPECULATIVE_REVOKE: Stream aborted</text>

<!-- Bottom Educational Banner -->
<rect x="16" y="278" width="688" height="66" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="297" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 STREAMING TRADEOFF: THE OPTIMAL BALANCE BETWEEN LATENCY AND SAFETY</text>
<text x="28" y="316" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Post-call guardrails are audit tools, not controls; once tokens stream to the client screen, data cannot be recalled.</text>
<text x="28" y="332" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">A sliding holdback window incurs minimal latency overhead while intercepting sensitive PII before rendering, enabling clean speculative rollbacks.</text>
</svg>

The containment condition cuts both ways: a violation longer than the window, or one that only becomes recognizable after more context (a sentence-level policy breach), is partly on screen before it is caught. Size $W$ for the longest pattern the detector must match, and treat semantic checks on a stream as best effort.

### 5.2. Sliding window buffering and speculative cancellation algorithm

The production architectural solution combines sliding window buffers with SSE-level **Speculative Rollback**:
1. The gateway retains incoming tokens in a sliding window buffer ($W = 64$ characters).
2. As new tokens enter, verified text falling outside the trailing window is released to the client socket as confirmed `TOKEN` events.
3. If an infraction (PII, prompt injection) is detected within the active buffer, the gateway aborts generation, **drops the held-back text, which never left the gateway**, and emits a `SPECULATIVE_REVOKE` event.
4. The client handles the event by replacing the partial answer already on screen with a policy notice. That text had passed inspection, but a sentence cut off before a withheld identifier is itself a hint, so it is retracted too. The "rollback" is of the display; nothing sensitive was sent.

---

## 6. Inference-time side-channel attacks

TLS hides the content of a response. It does not hide its *shape*: how large each packet is and when it arrives. Two documented channels matter for a serving layer:

- **Token-length channel.** When each token is streamed in its own event, the size of every encrypted packet reveals the length of that token. Weiss et al. (USENIX Security 2024) used exactly this sequence of lengths to reconstruct a large share of assistant responses from encrypted traffic. Later work showed that packet sizes and timings alone are enough to classify the *topic* of a conversation.
- **Cache-timing channel.** The prefix-cache oracle of Section 4.2: a fast first token tells the attacker that someone else sent the same prefix.

Mitigations are cheap and sit at the gateway: pad every streamed event to a fixed or randomized size, and batch several tokens per event instead of one. Both blur the length sequence. Against cache timing, the per-tenant `cache_salt` is the control; random delays only add noise that an attacker can average away.

---

## 7. Caching, retries, fallback routing, and guard calls

Reliability mechanisms introduce alternate execution paths that must be secured with identical rigor:

**Fallbacks Must Enforce Identical Guards:** If primary inference failure probability is $u_{\text{primary}}$, an unshielded fallback route exposes an uncontrolled traffic volume of:

$$
N_{\text{uncontrolled}} = N_{\text{total}} \cdot u_{\text{primary}}
$$

```text
In a system handling 100,000 requests/day with 99.5% primary availability (u = 0.005):
    Unshielded daily requests = 100,000 × 0.005 = 500 requests!
```

These 500 requests cluster during operational incidents. An adversary can trigger a denial-of-service attack against the primary cluster specifically to divert traffic into the unshielded fallback. Rule: Guardrails attach to the Gateway ingress, not individual model backends.

**Guard Calls Are Inference Calls:** Security SLMs and LLM judges are themselves inference workloads. If a guard model times out during high-impact operations, systems must strictly **fail-closed**, denying the transaction.

---

## 8. Rate limits, quotas, and budgets as security controls

Protecting serving infrastructure against denial-of-service and cost exhaustion requires measuring **tokens and currency**, not merely raw HTTP requests:

| Quota Dimension | Scope of Enforcement | Example Value |
| :--- | :--- | :--- |
| **Request Rate** | Per Client / API Token | 60 requests / minute |
| **Token Throughput** | Per User Session | 40,000 tokens (input+output) / min |
| **Completion Ceiling** | Per Routing Endpoint | 1,024 tokens for summaries, 4,096 for code |
| **Financial Budget** | Per Project / Department | Hard ceiling; 80% alert, 100% hard block |
| **Active Streams** | Per User Identity | Maximum 3 concurrent streaming sockets |

These quotas are the deterministic control on the exponent $n$ in Part 5's $1 - (1-p)^n$: they cap how many attempts an attacker gets per minute.

---

## 9. Practical output: Deployment architecture, acceptance test suite, and Python security engine

### 9.1. Deployment acceptance test matrix

The acceptance test suite verifies that no path bypasses security controls within the CI/CD pipeline:

| ID | Test Scenario | Verification Method | Pass Criterion |
| :--- | :--- | :--- | :--- |
| **S-1** | Direct Server Isolation | Attempt direct TCP connection from an app pod to vLLM | Network-level `Connection Refused` / Drop |
| **S-2** | mTLS Client Attestation | Connect from gateway to vLLM omitting client certificate | TLS Handshake Failure (Drop) |
| **S-3** | Open Port Reconnaissance | Scan model server subnet using Nmap from external VLAN | Port `:8443` visible only from gateway IP |
| **S-4** | Shielded Fallback Invariant| Terminate primary vLLM node; submit prompt injection suite | Fallback provider engaged; guardrail active on 100% of calls |
| **S-5** | Speculative Stream Revocation | Stream completion containing synthetic TCKN at character 300 | TCKN intercepted before display; `REVOKE` dispatched |
| **S-6** | Client Markdown Hardening | Stream completion containing `![leak](https://attacker.com)` | UI renders plain text; image loading suppressed |
| **S-7** | Guardrail Timeout Fail-Closed| Induce simulated timeout in security service | High-risk requests return 503 Service Unavailable |
| **S-8** | Prefix Cache Isolation | Send an identical prompt prefix under two tenant identities; send a client-chosen `cache_salt` | No prefix-cache hit across tenants (hit counter unchanged, TTFT not shortened); the gateway overwrites the client's salt |
| **S-9** | Token Quota Enforcement | Submit completion request exceeding per-minute token quota | 429 Too Many Requests returned with JSON error |
| **S-10**| Static Credential Scan | Inspect container images and deployment manifests | Zero plaintext API tokens outside secrets manager |

### 9.2. Production-grade Python serving security engine

The following standalone Python engine simulates per-tenant cache salting of a prefix cache and a holdback-window streaming guardrail:

```python
import hmac
import hashlib
import json
from typing import List, Dict, Any, Generator, Tuple

class TenantPrefixCacheManager:
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.block_cache: Dict[str, str] = {}

    def cache_salt(self, tenant_id: str) -> str:
        # The gateway derives the salt from the authenticated tenant; clients never choose it
        return hmac.new(self.secret_key, tenant_id.encode(), hashlib.sha256).hexdigest()

    def get_or_set_prefix(self, tenant_id: str, model_id: str, tokens: List[int], kv_pointer: str) -> Tuple[bool, str]:
        # Like vLLM's first-block hash: the salt is mixed into the key of the cached prefix
        payload = f"{self.cache_salt(tenant_id)}:{model_id}:{tokens}".encode()
        cache_key = hashlib.sha256(payload).hexdigest()
        if cache_key in self.block_cache:
            return True, self.block_cache[cache_key]
        self.block_cache[cache_key] = kv_pointer
        return False, kv_pointer

class SpeculativeStreamingGuardrail:
    def __init__(self, window_chars: int = 20):
        self.window_chars = window_chars
        self.blocked_patterns = ["TCKN:", "IBAN:", "SECRET_KEY"]

    def evaluate_stream(self, token_chunks: List[str]) -> Generator[Dict[str, str], None, None]:
        speculative_buffer = ""
        for token in token_chunks:
            speculative_buffer += token
            # Evaluate pattern violations within the sliding window
            if any(p in speculative_buffer for p in self.blocked_patterns):
                yield {"event": "SPECULATIVE_REVOKE", "action": "ERASE_CLIENT_BUFFER", "detail": "Sensitive data pattern intercepted"}
                return
            # Emit verified tokens that trail behind the holdback window
            if len(speculative_buffer) > self.window_chars:
                safe_len = len(speculative_buffer) - self.window_chars
                emitted_text = speculative_buffer[:safe_len]
                speculative_buffer = speculative_buffer[safe_len:]
                yield {"event": "TOKEN_EMIT", "text": emitted_text}
        # Flush remaining clean tokens at completion
        if speculative_buffer:
            yield {"event": "TOKEN_EMIT", "text": speculative_buffer}

# Demonstration
cache_mgr = TenantPrefixCacheManager(b"kv-master-salt-key-32b")
tokens = [1, 512, 1024, 2048]
MODEL = "mistral-7b-instruct-v0.3"
hit_a, ptr_a = cache_mgr.get_or_set_prefix("tenant-alpha", MODEL, tokens, "kv-block-0x1A")
hit_b, ptr_b = cache_mgr.get_or_set_prefix("tenant-beta", MODEL, tokens, "kv-block-0x9F")
hit_a2, ptr_a2 = cache_mgr.get_or_set_prefix("tenant-alpha", MODEL, tokens, "kv-block-0x2C")

print("--- 1. Multi-Tenant Prefix Cache Isolation ---")
print(f"Tenant-A first request, cache hit: {hit_a} (Block: {ptr_a})")
print(f"Tenant-B same prefix tokens, cache hit: {hit_b} (Block: {ptr_b})")
print(f"Tenant-A same prefix again, cache hit: {hit_a2} (Block: {ptr_a2})")

guard = SpeculativeStreamingGuardrail(window_chars=16)
incoming_tokens = ["Sayın çalışan, ", "ilgili müşterinin ", "TCKN: 10987654321", " numaralı hesabına iade yapılmıştır."]
print("\n--- 2. Speculative Streaming Guardrail ---")
for event in guard.evaluate_stream(incoming_tokens):
    print(f"Event: {event['event']} -> {event.get('text', event.get('detail'))}")
```

Execution terminal output:

```text
--- 1. Multi-Tenant Prefix Cache Isolation ---
Tenant-A first request, cache hit: False (Block: kv-block-0x1A)
Tenant-B same prefix tokens, cache hit: False (Block: kv-block-0x9F)
Tenant-A same prefix again, cache hit: True (Block: kv-block-0x1A)

--- 2. Speculative Streaming Guardrail ---
Event: TOKEN_EMIT -> Sayın çalışan, il
Event: SPECULATIVE_REVOKE -> Sensitive data pattern intercepted
```

---

## Pedagogical checkpoints

### Self-check

**Question 1:** Why is starting a vLLM server with `--api-key` insufficient to protect its attack surface from internal network reconnaissance?
- A) API keys in vLLM expire after 60 seconds.
- B) The `--api-key` flag protects only the `/v1`, `/v2`, `/inference` and `/cohere` prefixes; routes like `/invocations`, `/tokenize`, `/pause` and the inter-node ports remain unauthenticated.
- C) vLLM lacks an internal cryptographic engine.
- D) vLLM transmits traffic solely over unencrypted UDP.
*(Correct Answer: B. Administrative and auxiliary endpoints bypass the API key; reverse proxy routing and network policies are required.)*

**Question 2:** In NVIDIA H100 confidential computing, how is data protected while it crosses the PCIe bus between the confidential VM and the GPU?
- A) It is not protected; the bus is trusted.
- B) The driver and GPU derive keys in an SPDM session and move data through an encrypted bounce buffer.
- C) GPU memory (HBM) is encrypted, so the bus needs no protection.
- D) The hypervisor encrypts it on the guest's behalf.
*(Correct Answer: B. Link-level PCIe IDE is the standard that newer platforms build on; the on-package HBM itself is not encrypted.)*

**Question 3:** What is the critical failure mode of running output guardrails in `post_call` mode on streaming endpoints?
- A) Doubling GPU memory utilization.
- B) Inspection completes only after all tokens reach the client; sensitive data renders on screen before interception can occur.
- C) Reducing generation throughput to 1 token/sec.
- D) Causing dropped TCP packets.
*(Correct Answer: B. Post-call streaming operates in audit-only mode; speculative windowed buffers are required to prevent disclosure.)*

**Question 4:** What risk arises if a multi-tenant vLLM server shares its prefix cache across tenants without a per-tenant `cache_salt`?
- A) Overheating GPU silicon.
- B) A cache hit shortens Time to First Token, so one tenant can test guesses about another tenant's prompt prefix by timing responses.
- C) Model parameter counts degrade spontaneously.
- D) The model reverts exclusively to English outputs.
*(Correct Answer: B. The cache is content-addressed, so no foreign content is served; the leak is the timing of the hit, CVE-2025-46570.)*

### Thought experiment

**A PCIe interposer against a confidential GPU:**
A state-sponsored actor gains unauthorized physical access to a co-location data facility. The intruder installs an inline PCIe x16 interposer protocol analyzer between the motherboard and an NVIDIA H100 GPU hosting proprietary customer models.

*Question:* What does the intruder capture under standard serving, and what with the GPU in confidential mode?
*Answer:* Under standard serving, all DMA tensor blocks—including model weights, customer prompt embeddings, and unencrypted KV cache pages—traverse the bus as plaintext tensors; the adversary reconstructs all proprietary models and customer data. In confidential mode the payloads crossing the bus were encrypted by the driver inside the confidential VM with keys from the SPDM session, so the capture is ciphertext. What the interposer does not change is the residual risk: the HBM on the GPU package is not encrypted, so an attacker able to probe the package itself is outside what this design defends against.

### Disambiguation

- **mTLS vs bus encryption:** mTLS encrypts network traffic between two software endpoints (gateway and vLLM pod). Bus encryption protects data between the CPU TEE and the GPU: on the H100 through SPDM-keyed encrypted bounce buffers, on newer platforms at the link itself with PCIe IDE. One answers network eavesdropping; the other answers a hostile host or a bus probe.
- **Audit-Only vs Speculative Rollback:** Audit-only records log events after sensitive data has rendered on client displays. Speculative rollback holds tokens in a short sliding window, dropping the unsent text on a violation and telling the client to retract the partial answer.

### What questions can the reader now answer independently?

1. Why must enterprise application clients be prevented from communicating directly with model inference servers?
2. How does NVIDIA H100 Confidential Computing with SPDM attestation prevent model weight theft by malicious cloud host administrators?
3. Which vLLM endpoints bypass `--api-key` authentication, and how are they locked down?
4. How do sliding window buffers prevent structured PII leakage without degrading Time-to-First-Token (TTFT)?
5. Why must fallback routing paths enforce identical guardrails as primary inference clusters?

---

## The whole story in six lines

- If attackers can bypass security controls through alternative paths, they do not need to defeat them; the serving layer enforces universal compliance.
- The AI Gateway authenticates, meters, and routes; model servers communicate exclusively through isolated mTLS connections.
- GPU confidential computing attests the device, keys an SPDM session and encrypts what crosses the bus; on-package HBM is not encrypted, and PCIe IDE is the link-level standard newer platforms adopt.
- vLLM's API key covers four prefixes and leaves operational routes open; a shared prefix cache leaks through timing, so the gateway sets a per-tenant `cache_salt`.
- Post-call streaming guards cannot retract displayed tokens; a holdback window catches structured PII before it leaves the gateway.
- Unshielded fallback routes leave platforms vulnerable during outages; multi-dimensional token and cost quotas preserve service continuity.

---

## Glossary

- **AI Gateway:** The unified ingress proxy that authenticates clients, enforces rate limits, manages fallbacks, and executes guardrails.
- **Model Server:** The high-throughput inference engine (e.g., vLLM) hosting models on GPU hardware.
- **Confidential Computing:** The security paradigm of protecting data in-use via hardware-isolated Trusted Execution Environments (TEEs).
- **PCIe IDE (Integrity and Data Encryption):** The PCIe standard for link-level encryption (AES-GCM). Platforms with TDISP use it to protect CPU TEE to device traffic; the H100 instead encrypts transfers in the driver with SPDM session keys.
- **SPDM (Security Protocol and Data Model):** A standard protocol for hardware device attestation and firmware integrity verification.
- **Prefix Cache and `cache_salt`:** vLLM reuses KV blocks for identical prompt prefixes, identified by chained block hashes; a per-tenant `cache_salt` mixed into the first hash keeps tenants from sharing, and timing, each other's prefixes.
- **Speculative Rollback:** On a violation inside the holdback window, the gateway drops the unsent text and signals the client to retract the partial answer already shown.
- **TTFT (Time-to-First-Token):** Elapsed duration between request dispatch and the rendering of the initial output token.
- **Side-Channel Attack:** An attack technique inferring sensitive data from timing variances and packet size distributions rather than breaking cryptography.
- **Fail-Closed:** The principle of terminating a request securely when a downstream guardrail or security service experiences a timeout.

---

## Going deeper

- Previous part: [Secure AI by Design (8): Least Privilege and Safe Tool Use for AI Agents](post.html?slug=least-privilege-for-ai-agents)
- Next part: [Secure AI by Design (10): Measuring AI Security — Red Teaming and Turkish Evaluation](post.html?slug=measuring-ai-security)
- NVIDIA, [*Confidential Computing on NVIDIA H100 GPUs for Secure and Trustworthy AI*](https://developer.nvidia.com/blog/confidential-computing-on-h100-gpus-for-secure-and-trustworthy-ai/) and [*Hardware-Rooted Trust with SPDM*](https://docs.nvidia.com/confidential-computing/).
- vLLM, [*Security Best Practices*](https://docs.vllm.ai/en/latest/usage/security.html) and [*Automatic Prefix Caching Architecture*](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html).
- LiteLLM, [*Custom Guardrail Implementation and Streaming Hooks*](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail).
- OWASP, [*API Security Top 10 (2023)*](https://owasp.github.io/www-project-api-security/) (API4, API9) and [*Top 10 for LLM Applications 2026*](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10) (LLM06, LLM10).
- R. Weiss et al., [*What Was Your Prompt? A Remote Keylogging Attack on AI Assistants*](https://arxiv.org/abs/2403.09751) (USENIX Security 2024).
- OWASP, [*AISVS 1.0*](https://owasp.github.io/www-project-artificial-intelligence-security-verification-standard-aisvs-docs/), Chapter C4 (Infrastructure and Deployment Security).
- From this blog: [Inside the vLLM Engine](post.html?slug=vllm-from-paper-to-production) and [LLM Cost and Latency Optimization](post.html?slug=optimizing-llm-cost-and-latency).
