The demo went perfectly. The support assistant pulled up a customer record, summarized a three-page complaint in four sentences, and drafted a polite reply. Then someone from the security team asked two questions: *"What exactly are we protecting in this system, and who owns each of those things?"* The room went quiet. The AI team pointed at the model provider's safety filter, the platform team pointed at the API gateway, and the business owner assumed "IT handles security."

That silence is the most common AI security vulnerability, and it is not a bug in any one component. Our assistant has one tool, `send_email`, and that tool fundamentally changes what kind of system it is. Without it, the worst outcome of a manipulated model is a bad paragraph on a screen. With it, a hidden sentence inside a customer's complaint can make the assistant exfiltrate a customer record to an external address. Nothing about the model changed. What changed is the **blast radius**: the set of real-world consequences a single wrong token can trigger.

> **The Foundational Architectural Axiom of AI Security (Propose vs. Commit):**
> A generative AI model's output is never an executable command to the rest of the system; it is merely an unverified **proposal**. Large language models cannot possess native execution authority. Any action that generates a side effect cannot be committed unless verified by a deterministic Policy Enforcement Point (**PEP**) within the user's permission boundaries and parameter constraints.

This article is the founding piece of the *Secure AI by Design* series. We will not start with prompt tweaks or generic filters. First, we establish what is worth protecting: data, model and configuration, action authority, and service continuity. Next, we draw sharp boundaries between four concepts teams constantly confuse: security, safety, privacy, and reliability. Then we lay down four design principles and prove—using reliability engineering's $\beta$-factor model—why "defense in depth" geometrically slashes risk under independent assumptions, yet quietly collapses under Common Cause Failures and systemic correlation. By the end, you will have a reference architecture partitioned into six trust zones, a one-page principles sheet, and an enterprise RACI matrix that later parts build upon.

> **The running example of this series:** A corporate *support assistant* that (1) fetches a customer record, (2) summarizes a complaint, and (3) can send an email **only after user approval and to an authorized recipient**. All data in the series is synthetic; no real enterprise systems or secrets are used. Each part adds the next layer to this same system.

> **By the end of this article**
>
> - You will be able to list the four asset classes of an AI system and pinpoint where each lives across six trust zones (Z0–Z5) on an architectural diagram.
> - You will be able to tell security, safety, privacy, and reliability apart, articulating exactly which operational failure each prevents.
> - You will be able to calculate how common systemic blind spots between layers multiply residual risk using reliability engineering's $\beta$-factor formula.
> - You will have a one-page principles sheet enforcing the "Propose vs. Commit" invariant and a RACI matrix ready to copy into your own project.

**In this article**

- [1. The running example: anatomy of a support assistant and six trust zones](#1-the-running-example-anatomy-of-a-support-assistant-and-six-trust-zones)
- [2. What are we protecting? Four asset classes](#2-what-are-we-protecting-four-asset-classes)
- [3. CIA for AI: confidentiality, integrity, availability and safe behavior](#3-cia-for-ai-confidentiality-integrity-availability-and-safe-behavior)
- [4. Four words teams confuse: security, safety, privacy, reliability](#4-four-words-teams-confuse-security-safety-privacy-reliability)
- [5. Four design principles](#5-four-design-principles)
- [6. The math of defense in depth: the beta-factor model and common-cause collapse](#6-the-math-of-defense-in-depth-the-beta-factor-model-and-common-cause-collapse)
- [7. Two engineering lenses: build time and run time](#7-two-engineering-lenses-build-time-and-run-time)
- [8. Who owns what? The responsibility matrix](#8-who-owns-what-the-responsibility-matrix)
- [9. Practical output: the one-page principles sheet](#9-practical-output-the-one-page-principles-sheet)
- [10. Where this series goes next](#10-where-this-series-goes-next)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. The running example: anatomy of a support assistant and six trust zones

A security conversation becomes concrete only when trust boundaries are formally drawn on the whiteboard. Therefore, before stating any principle, let us map our support assistant across six distinct **trust zones** aligned with enterprise security standards (NIST AI 600-1 and Google SAIF).

A support agent opens a chat window and types: *"Summarize customer 4711's last complaint and draft a reply."* The request traverses six distinct trust zones:

1. **Zone 0 (External Untrusted Plane):** Support emails, external webhooks, and public web forms written by outside parties. Every single byte originating here is potentially hostile.
2. **Zone 1 (Ingress & Network Boundary):** Authenticated internal employee sessions, mTLS transport, API Gateway, WAF, rate limiting, and ingress PII tokenization/masking.
3. **Zone 2 (Quarantined Reasoning Plane):** The orchestrator's context assembly engine and the large language model (LLM). Untrusted complaint text and system prompts merge here into a single token sequence. The model operates strictly as an **isolated proposer**; it holds no native credentials to call external APIs.
4. **Zone 3 (Policy Enforcement Point & Policy Decision Point / PEP & PDP):** The deterministic security gate that intercepts the model's text completions and proposed tool calls. It validates user identity scopes, tool schemas, parameter constraints, and enforces human-in-the-loop (HITL) approval for high-impact actions.
5. **Zone 4 (Execution & Tool Plane):** Sandboxed tool runners (`get_customer_record`, `summarize_complaint`, `send_email`). They execute solely upon receiving an authenticated cryptographic ticket or signature from Zone 3.
6. **Zone 5 (Immutable Audit & Telemetry Plane):** OpenTelemetry GenAI semantic conventions, scrubbed of PII, stored in WORM (*Write Once Read Many*) storage for tamper-proof forensics.

The critical insight is the **direction of trust and execution decoupling**. The employee is authenticated. The orchestrator is trusted. However, an external entity wrote the complaint text. When that complaint enters the model, attention weights can be hijacked. Even if the model is fully subverted, no external side effect can occur unless the deterministic policy gate in Zone 3 is breached. The following diagram illustrates these six trust zones and their critical enforcement chokepoint:

<!-- fig:p1_arch -->
<svg viewBox="0 0 720 516" role="img" aria-label="Architecture of the support assistant across six trust zones. Zone 0 contains external untrusted senders. Zone 1 is the employee and chat gateway. Zone 2 is the quarantined reasoning plane with orchestrator and LLM. Zone 3 is the deterministic Policy Enforcement Point PEP and PDP gate validating model proposals. Zone 4 is the tool execution plane. Zone 5 is the immutable audit plane." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p1a-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p1a-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="p1a-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
<marker id="p1a-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="p1a-p" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Support Assistant: Six Trust Zones and the Propose vs. Commit Decoupling</text>
<rect x="16" y="56" width="688" height="376" rx="10" style="fill:none;stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 4"/>
<!-- Zone 0 -->
<rect x="28" y="320" width="130" height="92" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="93" y="338" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">Z0: UNTRUSTED EXT</text>
<text x="93" y="362" text-anchor="middle" style="fill:var(--c-text);font-size:12px">External Sender</text>
<text x="93" y="378" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">complaint email</text>
<text x="93" y="394" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-family:var(--font-mono)">injection payload</text>
<!-- Zone 1 -->
<rect x="28" y="76" width="130" height="210" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="93" y="94" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:700">Z1: INGRESS BOUNDARY</text>
<rect x="38" y="110" width="110" height="60" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="93" y="132" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Employee</text>
<text x="93" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">OAuth2 · mTLS</text>
<rect x="38" y="186" width="110" height="84" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="93" y="206" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Chat UI &amp; Gateway</text>
<text x="93" y="224" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">session context</text>
<text x="93" y="240" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">rate limiter</text>
<text x="93" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">PII masking</text>
<!-- Zone 2 -->
<rect x="180" y="76" width="180" height="336" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.3"/>
<text x="270" y="94" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:700">Z2: ISOLATED REASONING</text>
<rect x="190" y="110" width="160" height="110" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="270" y="130" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Orchestrator</text>
<text x="270" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">prompt assembly</text>
<text x="270" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">data/instruction split</text>
<text x="270" y="180" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px">untrusted text</text>
<text x="270" y="196" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px">meets logic here</text>
<line x1="270" y1="220" x2="270" y2="246" marker-end="url(#p1a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="190" y="248" width="160" height="88" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="270" y="268" text-anchor="middle" style="fill:var(--c-accent-2);font-size:13px;font-weight:600">LLM Inference</text>
<text x="270" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">pure text / proposal engine</text>
<text x="270" y="302" text-anchor="middle" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">PROPOSES ONLY</text>
<text x="270" y="318" text-anchor="middle" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">(CANNOT EXECUTE)</text>
<!-- Zone 3 -->
<rect x="380" y="76" width="140" height="336" rx="6" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.5"/>
<text x="450" y="94" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:700">Z3: PEP / PDP GATE</text>
<rect x="390" y="110" width="120" height="286" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="450" y="132" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Policy Enforcement</text>
<text x="450" y="148" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:600">PEP / PDP</text>
<line x1="400" y1="160" x2="500" y2="160" style="stroke:var(--c-border);stroke-width:1"/>
<text x="450" y="178" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">1. Schema Validation</text>
<text x="450" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">2. User Token Scope</text>
<text x="450" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">3. Recipient Allowlist</text>
<text x="450" y="232" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">4. Parameter Signing</text>
<text x="450" y="250" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">5. Human Approval (HITL)</text>
<rect x="398" y="270" width="104" height="42" rx="4" style="fill:var(--c-warn);fill-opacity:.15;stroke:var(--c-warn);stroke-width:1"/>
<text x="450" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-weight:600">Approval Ticket:</text>
<text x="450" y="300" text-anchor="middle" style="fill:var(--c-warn);font-size:9.5px;font-family:var(--font-mono)">HMAC(id,to,body)</text>
<text x="450" y="334" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px;font-weight:700">COMMIT / DROP</text>
<text x="450" y="352" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">verdict issued</text>
<!-- Zone 4 -->
<rect x="540" y="76" width="150" height="210" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="615" y="94" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">Z4: EXECUTION &amp; TOOLS</text>
<rect x="548" y="110" width="134" height="42" rx="4" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="615" y="127" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">get_customer_record</text>
<text x="615" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">CRM (read-only)</text>
<rect x="548" y="160" width="134" height="42" rx="4" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="615" y="177" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">summarize_complaint</text>
<text x="615" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Archive (read-only)</text>
<rect x="548" y="210" width="134" height="58" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="615" y="228" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">send_email</text>
<text x="615" y="244" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-weight:600">HIGH BLAST RADIUS</text>
<text x="615" y="258" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px">approved recipients only</text>
<!-- Zone 5 -->
<rect x="540" y="302" width="150" height="110" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="615" y="320" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Z5: IMMUTABLE AUDIT</text>
<text x="615" y="344" text-anchor="middle" style="fill:var(--c-text);font-size:11px">OTel GenAI Spans</text>
<text x="615" y="362" text-anchor="middle" style="fill:var(--c-text);font-size:11px">WORM Security Log</text>
<text x="615" y="380" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Salted Token Hashes</text>
<text x="615" y="396" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Signed Decision Proof</text>
<!-- Connections -->
<line x1="148" y1="190" x2="190" y2="190" marker-end="url(#p1a-p)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M158 366 H190 V220" marker-end="url(#p1a-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5;stroke-dasharray:4 3"/>
<line x1="350" y1="290" x2="390" y2="290" marker-end="url(#p1a-w)" style="stroke:var(--c-warn);stroke-width:1.6"/>
<text x="370" y="280" text-anchor="middle" style="fill:var(--c-warn);font-size:9px;font-family:var(--font-mono)">proposal</text>
<line x1="510" y1="210" x2="548" y2="210" marker-end="url(#p1a-g)" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<text x="528" y="200" text-anchor="middle" style="fill:var(--c-accent-2);font-size:9px;font-family:var(--font-mono)">commit</text>
<path d="M450 396 V416 H540 V360" marker-end="url(#p1a-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<!-- Bottom Educational Banner -->
<rect x="16" y="444" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="463" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 FOUNDATIONAL ARCHITECTURAL LAW: PROPOSAL IS STRICTLY DECOUPLED FROM COMMIT</text>
<text x="28" y="481" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">The stochastic AI reasoning plane (Zone 2) possesses no external agency; it can only propose structured actions.</text>
<text x="28" y="497" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">State transitions (commit) execute exclusively through the deterministic Policy Enforcement Point (PEP - Zone 3).</text>
</svg>

> [!TIP]
> **Learning Check: The Chasm Between Model Safety and System Security**
> Training a model via RLHF or DPO to "refuse malicious prompts" is an exercise in **model safety**. To a security architect, this cannot serve as a trust perimeter. Historically, any language model can be subverted by clever jailbreak prefixes, out-of-distribution tokens, or semantic redirection. **System security** guarantees that even when the model is fully compromised and happily proposes a malicious action, the deterministic Policy Enforcement Point in Zone 3 drops the call silently.

---

## 2. What are we protecting? Four asset classes

Think of a bank branch. The cash inside the vault is an asset. But the vault's time-delayed combination, the teller's transfer authorization key, and the branch remaining open during business hours are also assets. An adversary who learns the combination has stolen zero dollars yet acquired operational control over everything.

An AI system harbors the exact same four asset classes across its trust zones:

| Asset Class | In Our Support Assistant | Trust Zone | What Does "Losing It" Look Like? |
| --- | --- | --- | --- |
| **Data** | Customer records, complaint archive, chat history, vector index, audit logs | Z1, Z4, Z5 | One customer's PII surfaces in another user's summary; complaint text leaks into an unprivileged log export. |
| **Model & configuration** | Model weights, system prompt, tool JSON schemas, guardrail thresholds, eval sets | Z2 | Confidential system prompt is extracted; tool descriptions are poisoned; security classifier threshold is quietly loosened. |
| **Action authority** | `send_email` execution right, CRM API credentials, OAuth scopes, approval tokens | Z3, Z4 | The assistant dispatches an unapproved email; leaked service keys permit unauthorized CRM exfiltration. |
| **Service continuity** | Endpoint availability, inference latency budget, token and financial quota | Z1, Z2, Z4 | A recursive injection attack consumes the monthly GPU budget in two hours; guardrail service crashes and leaves the system open. |

Two rows demand special focus from an AI security architect:

**Action authority** is absent from traditional ML threat models. A standard churn prediction model simply outputs a probability score ($[0, 1]$); it cannot delete databases or execute wire transfers. An agent armed with tools, however, *borrows* authority: from the logged-in user, from a service account, or both. OWASP's Top 10 for LLM Applications ranks *Excessive Agency* in its top tier. OWASP's Top 10 for Agentic Applications opens with *Agent Goal Hijack* (ASI01): The moment an AI system gains tool execution rights, a manipulated token sequence is no longer just a text quality glitch—it becomes an immediate, real-world exploit.

**Service continuity** includes *financial exhaustion*. A cyclical prompt injection inducing 64,000 generated tokens per call may not crash an operating system, but blasting parallel requests exhausts API credit limits within hours, rendering services unavailable to legitimate users (OWASP *Unbounded Consumption*).

> [!NOTE]
> **Thought Experiment: The Cost of a Single Token**
> Imagine this single string appearing in the model's generated token stream:
> `{"name": "send_email", "arguments": {"to": "audit@competitor.com", "body": "ALL_CRM_DATA..."}}`
> Has any enterprise breach occurred at the exact instant those tokens are decoded into GPU VRAM? No. A breach occurs only when the orchestrator parses that string into a JSON payload and dispatches it over an SMTP socket. The security failure is not that the model hallucinated or complied with an adversary; the failure is that the architecture treated that unverified string as executable code.

---

## 3. CIA for AI: confidentiality, integrity, availability and safe behavior

Classic cybersecurity rests upon three pillars: **Confidentiality**, **Integrity**, and **Availability** (the CIA triad). While applicable to AI systems, their failure modes manifest in novel ways:

- **Confidentiality** is breached not just via database exfiltration, but when a model regurgitates memorized training data, leaks confidential system prompts, or when dense embeddings are inverted back into plaintext via Vec2Text attacks.
- **Integrity** traditionally implies unauthorized modification of database records. In AI, the dominant vulnerability is the loss of **behavioral integrity**. A prompt injection alters zero weight bytes in the model, yet diverts the system's runtime decision logic from the authorized user's intent to the attacker's directive.
- **Availability** is disrupted by denial-of-wallet attacks, runaway multi-agent loops, and guardrail service timeouts.

However, the classic CIA triad fails to capture AI's most pervasive hazard: **Safe behavior**.

A system can preserve perfect confidentiality, maintain unblemished database integrity, and sustain 100% uptime, yet output legally binding defamatory claims, generate toxic rhetoric, or provide step-by-step chemical weapon synthesis. This is not a classic security outage; it is a **behavioral alignment failure**.

> **Keep in mind:** In an AI system, an *integrity attack* can succeed without altering a single stored byte. The adversary corrupts the *input*; the corrupt input warps the model's *reasoning*; warped reasoning executes an unauthorized *action*. Every architectural control in this series exists to break that execution chain.

---

## 4. Four words teams confuse: security, safety, privacy, reliability

In architectural reviews, these four terms are routinely conflated, creating confusion and diffusion of responsibility. Each addresses a fundamentally different question:

| Concept | Core Question | Adversary Required? | Example System Failure | Primary Owner |
| --- | --- | --- | --- | --- |
| **Security** | Can a malicious *adversary* deliberately compel the system into unauthorized actions? | **Yes** (Hostile intent) | Injection hidden in a complaint exfiltrates customer records to an attacker | Security Architect + Platform Team |
| **Safety** | Can the system inflict *harm* upon humans or operations under benign usage? | **No** (Normal user) | Assistant confidently advises a lethal chemical combination to a customer | AI Safety Lead + Business Owner |
| **Privacy** | Is *personal data* (PII/GDPR) processed lawfully, minimally, and strictly for its stated purpose? | **No** (Systemic leak) | Customer government ID numbers are stored unencrypted in debug logs for two years | Data Protection Officer + Legal |
| **Reliability** | Does the system function *correctly and consistently* under standard operational conditions? | **No** (Statistical variance) | The complaint summarizer silently drops the primary customer grievance in 8% of cases | ML Engineering + QA Team |

> [!NOTE]
> **Concept Untangling: Safety vs. Security — Why They Must Not Be Conflated**
> Erecting bulletproof glass and installing biometric locks at a bank counter is a **security** control; it repels intentional robbers. Training a teller not to hand out the wrong statement or applying non-slip floor coating is a **safety and reliability** control.
> In the LLM domain, "safe model" must be disambiguated: Ensuring a model does not emit profanity is safety. Ensuring it does not leak its hidden system prompt or yield email execution rights to an adversary is security.

The NIST AI Risk Management Framework (NIST AI RMF 1.0) defines seven characteristics of trustworthy AI: valid and reliable, safe, secure and resilient, accountable and transparent, explainable, privacy-enhanced, and fair. The goal is not rote memorization, but recognizing that **security is only one of seven imperatives**. An AI system can be impeccably polite and safe, yet catastrophically vulnerable to security exploits.

---

## 5. Four design principles

Principles are actionable only when they can be evaluated via definitive yes/no architectural tests during code reviews.

### 5.1. Least privilege and least agency
Grant each component strictly the minimal permissions necessary for the immediate task. In agentic systems, OWASP broadens this to **least agency**: Do not confer greater operational autonomy than the task strictly demands.

- *Implementation:* The `get_customer_record` tool must not connect via an all-powerful `db_admin` role; it must execute under a scoped service identity restricted to the specific customer ID authorized for the active user session. The `send_email` tool must reject arbitrary recipient addresses, restricting delivery strictly to verified customer records.
- *Architectural Test:* Select any tool and ask: *"If the model were fully compromised by an adversary, what is the absolute worst action this tool could execute with its credentials?"* If the answer is "read the entire CRM table," least privilege has failed.

### 5.2. Defense in depth
Never stake system integrity upon a single prompt instruction, classifier, or guardrail filter. Chain complementary controls with distinct failure modes across boundaries: input validation, context tagging, execution gating, allowlists, and human verification.

- *Architectural Test:* For any high-impact operation, enumerate the independent control points that must fail simultaneously for an unauthorized transaction to occur. If that number is 1, defense in depth does not exist.

### 5.3. Secure defaults and fail-closed behavior
The default configuration must always be the most restrictive state. When a security filter or guardrail service experiences a network timeout or internal crash, the system must not bypass controls (*fail-open*); it must halt the operation (*fail-closed*).

- *Implementation:* If an external guardrail classifier fails to respond within 500 ms, side-effecting operations like `send_email` must be blocked immediately. In contrast, low-risk read operations (e.g., complaint summarization) may degrade gracefully by returning a safe error fallback.
- *Architectural Test:* Terminate the guardrail service in a staging environment. Does the system dispatch emails (*fail-open*) or block execution (*fail-closed*)? The majority of enterprise breaches occur through quietly failing-open systems.

### 5.4. Meaningful human oversight (Meaningful HITL)
Human approval constitutes a valid control only if the operator possesses the visibility, context, and time required to evaluate the operation. An unadorned dialog box asking "Send email? [Yes/No]" without displaying the recipient and body is not oversight; it is liability shifting (OWASP *Human-Agent Trust Exploitation*, ASI09).

- *Implementation:* The approval screen must explicitly display the resolved recipient, subject line, full message body, and source trigger. The approval token must be cryptographically bound to an HMAC digest of these exact parameters (`HMAC(id, to, body)`). If the model mutates a single character, the ticket becomes invalid.
- *Architectural Test:* Does the human approval ticket match the exact JSON payload submitted to the downstream API? If operator approval rates exceed 99.9%, real review has degraded into click-through fatigue.

---

## 6. The math of defense in depth: the beta-factor model and common-cause collapse

The intuitive model behind defense in depth is James Reason's "Swiss Cheese" model. Each slice contains holes, but stacking enough slices renders alignment of the holes improbable. However, this intuition fails catastrophically when confronted with **Common Cause Failures (CCF)**, a well-understood phenomenon in nuclear and aerospace reliability engineering.

Consider an adversarial payload traversing $n$ sequential security layers. Let $p_i$ denote the independent probability of the $i$-th layer failing to intercept the attack. If all layers are **strictly independent**, the joint probability of a complete system breach is given by the multiplication rule:

$$
P_{\text{breach, independent}} = \prod_{i=1}^{n} p_i
$$

Where $p_i$ is the miss rate of layer $i$, and $\prod$ denotes the product across all $n$ layers.

Now consider reality: Reliability engineering (IEC 61508 / ISO 26262) utilizes the **Beta-Factor ($\beta$) Model** to account for the reality that a fraction of failures stem from a *shared systemic cause*. In AI systems, this common cause arises when all layers rely on identical base models, share identical BPE tokenizers, lack coverage for identical language families, or succumb to identical prompt obfuscation techniques.

If a fraction $\beta$ of adversarial inputs exploit a shared blind spot where the joint failure rate is $p_{\text{common}}$, the true breach probability becomes:

$$
P_{\text{breach}} = \beta \cdot p_{\text{common}} + (1 - \beta) \prod_{i=1}^{n} p_i
$$

Where $\beta$ represents the proportion of attacks falling into the shared architectural blind spot, $p_{\text{common}}$ is the failure probability under that blind spot (for zero-day injection patterns, $p_{\text{common}} = 1$), and $1-\beta$ represents standard attacks subjected to independent filtering.

When an attacker crafts an exploit targeting the shared blind spot directly ($p_{\text{common}} = 1$), the equation simplifies to:

$$
P_{\text{breach}} = \beta + (1 - \beta) \prod_{i=1}^{n} p_i
$$

### Step-by-Step Pedagogical Numerical Walk

Suppose our support assistant protects `send_email` with three defense layers:
1. Layer 1 ($p_1 = 0.10$): Input prompt injection detector (misses 10% of attacks).
2. Layer 2 ($p_2 = 0.20$): LLM-as-a-judge output policy filter (misses 20% of attacks).
3. Layer 3 ($p_3 = 0.05$): Deterministic schema and allowlist validator (fails on 5% of configuration oversights).

Assume that across these layers exists a shared tokenizer vulnerability with $\beta = 0.04$ (4% common cause fraction) where $p_{\text{common}} = 1.0$:

```text
Layer Parameters:
  p1 = 0.10   (Input injection detector)
  p2 = 0.20   (Output policy judge)
  p3 = 0.05   (Schema & allowlist validator)
  β  = 0.04   (Shared systemic blind spot)
  p_common = 1.0 (Complete breach within blind spot)

Step 1: Independent probability calculation
  P_indep = p1 × p2 × p3
          = 0.10 × 0.20 × 0.05
          = 0.02 × 0.05
          = 0.0010              → On paper: 1 breach per 1,000 attacks (0.10%)

Step 2: Beta-factor common cause calculation
  P_common_term = β × p_common
                = 0.04 × 1.0 = 0.0400
  P_routine_term = (1 - β) × P_indep
                 = (1 - 0.04) × 0.0010
                 = 0.96 × 0.0010 = 0.00096
  P_total       = P_common_term + P_routine_term
                = 0.0400 + 0.00096
                = 0.04096       → In reality: ~41 breaches per 1,000 attacks (4.10%)

Step 3: Vulnerability multiplier
  Ratio = 0.04096 / 0.0010 ≈ 41.0x worse than the independent assumption!
```

This reveals an essential architectural axiom: **The systemic floor set by $\beta$ is entirely independent of how many AI guardrails you stack.** Adding a 4th and 5th LLM-based filter cannot push residual risk below 4%, because when an adversary discovers a representation bypass, all LLM-based layers fail in tandem.

The only mechanism to break this floor is **Architectural Diversity**: introducing deterministic, non-model controls. Recipient allowlists and HMAC parameter verifications do not evaluate semantic nuance; they enforce strict relational logic.

The following Python script verifies this analytical derivation against a 1,000,000-run Monte Carlo simulation:

```python
import random

p = [0.10, 0.20, 0.05]   # Layer miss rates
beta = 0.04              # Common cause blind spot fraction
p_common = 1.0           # Breach rate under common cause

# 1. Theoretical analytical calculation
independent = 1.0
for x in p:
    independent *= x
correlated = beta * p_common + (1.0 - beta) * independent

print(f"Theoretical Independent Miss: {independent:.6f} ({independent*100:.3f}%)")
print(f"Theoretical Beta-Factor Miss: {correlated:.6f} ({correlated*100:.3f}%)")
print(f"Vulnerability Multiplier    : {correlated / independent:.1f}x")

# 2. Monte Carlo empirical validation (1,000,000 iterations)
random.seed(42)
N, missed = 1_000_000, 0

for _ in range(N):
    if random.random() < beta:
        # Falls into systemic common blind spot
        if random.random() < p_common:
            missed += 1
    else:
        # Passes through independent layers
        if all(random.random() < x for x in p):
            missed += 1

simulated_rate = missed / N
print(f"Monte Carlo Simulated Rate  : {simulated_rate:.6f} ({simulated_rate*100:.3f}%)")
```

Console output:

```text
Theoretical Independent Miss: 0.001000 (0.100%)
Theoretical Beta-Factor Miss: 0.040960 (4.096%)
Vulnerability Multiplier    : 41.0x
Monte Carlo Simulated Rate  : 0.040845 (4.085%)
```

---

## 7. Two engineering lenses: build time and run time

Just as AI infrastructure separates prefill from decode or training from inference, AI security architecture operates across two distinct operational regimes:

**Build Time** encompasses everything prior to serving a live user request: curating datasets, fine-tuning, downloading open weights, authoring system prompts, tuning guardrails, and red-teaming. Threats here are **slow, stealthy, and persistent**. A poisoned adapter or backdoored weight artifact compromises all future sessions; the attacker needs to succeed only once. Key controls are **provenance, cryptographic signatures (Sigstore/Cosign), SBOMs, and change management**.

**Run Time** spans the sub-second lifespan of an inference call: authentication, prompt construction, RAG retrieval, LLM forward passes, PEP verification, tool dispatch, and logging. Threats here are **fast, dynamic, and automated**. An adversary can probe hundreds of prompt permutations per minute. Key controls are **trust boundaries, deterministic validation, least privilege, and resource ceilings**.

ETSI EN 304 223 (Europe's Standard for Securing AI) structures security requirements across five lifecycle phases: Design, Development, Deployment, Maintenance, and Decommissioning. Decommissioning is frequently overlooked: When an assistant is retired, orphaned vector databases, fine-tuned weights, and debug traces remain active leak vectors unless systematically wiped.

<!-- fig:p1_lifecycle -->
<svg viewBox="0 0 720 346" role="img" aria-label="AI lifecycle across five phases: design, development, deployment, maintenance, and decommissioning. Design and development constitute build time with slow, persistent threats. Deployment and maintenance constitute run time with fast, repeatable threats. Decommissioning purges indices, weights, and revokes keys." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p1l-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p1l-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Single Lifecycle, Five Security Phases (ETSI EN 304 223)</text>
<rect x="16" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="76" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Design</text>
<text x="26" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">assets, owners</text>
<text x="26" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">threat model</text>
<text x="26" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">trust zones Z0-Z5</text>
<line x1="136" y1="140" x2="156" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="158" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="218" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Development</text>
<text x="168" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">data, provenance</text>
<text x="168" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">model, adapters</text>
<text x="168" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">policy schemas</text>
<line x1="278" y1="140" x2="298" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="300" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Deployment</text>
<text x="310" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PEP/PDP gateway</text>
<text x="310" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tool sandboxing</text>
<text x="310" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HMAC approvals</text>
<line x1="420" y1="140" x2="440" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="442" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="502" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Maintenance</text>
<text x="452" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">red teaming (ART)</text>
<text x="452" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">OTel telemetry</text>
<text x="452" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">incident response</text>
<line x1="562" y1="140" x2="582" y2="140" marker-end="url(#p1l-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="584" y="92" width="120" height="96" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="644" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Retirement</text>
<text x="594" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">purge vector index</text>
<text x="594" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">crypto-erase weights</text>
<text x="594" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">revoke API tokens</text>
<rect x="16" y="46" width="262" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="147" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px">build time: provenance, signing, audits</text>
<rect x="300" y="46" width="262" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="431" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px">run time: boundaries, PEP, quotas</text>
<rect x="584" y="46" width="120" height="32" rx="6" style="fill:var(--c-danger);fill-opacity:.10;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="644" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px">crypto erasure</text>
<path d="M502 188 V218 H218 V190" marker-end="url(#p1l-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="360" y="236" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">every prompt, model, or tool schema change triggers regression evals</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="274" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="293" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 LIFECYCLE GOVERNANCE AXIOM: AI SECURITY IS A CONTINUOUS ASSURANCE DISCIPLINE</text>
<text x="28" y="311" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">From initial design to decommissioning, every prompt, weight, or policy update triggers automated regression verification.</text>
<text x="28" y="327" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Decommissioning enforces permanent cryptographic erasure across all fine-tuned adapters, vector stores, and cached embeddings.</text>
</svg>

---

## 8. Who owns what? The responsibility matrix

Returning to our opening boardroom scene: Confusion is eliminated not by introducing a newer filter, but by establishing unambiguous organizational ownership. A **RACI** matrix provides this contract: specifying who is **R**esponsible, **A**ccountable (strictly one entity per row), **C**onsulted, and **I**nformed.

| Activity | Business Owner | AI / ML Team | Software Team | Platform / SRE | Security Architect | DPO / Legal |
| --- | --- | --- | --- | --- | --- | --- |
| Use case definition & risk tolerance thresholds | **A** | C | C | I | C | C |
| Asset inventory & trust boundary mapping (Z0–Z5) | C | R | R | R | **A** | C |
| Data provenance, PII sanitization & retention policy | C | R | R | I | C | **A** |
| Model, adapter, and supply chain acceptance | I | R | C | R | **A** | I |
| PEP/PDP gate, tool scopes, and HITL workflow | C | C | R | C | **A** | I |
| Gateway, mTLS, and network isolation infrastructure | I | I | C | R | **A** | I |
| Automated Red Teaming (ART) & security evaluations | I | R | C | I | **A** | C |
| OTel telemetry auditing & incident response runbook | I | C | R | R | **A** | C |
| Production release sign-off & residual risk acceptance | **A** | C | C | C | R | C |

Two governance rules are non-negotiable:
1. **Residual risk is formally accepted by the Business Owner, never the Security Architect.** Security measures, models, and mitigates risk. However, the commercial decision to deploy with known residual risk belongs solely to the business stakeholder.
2. **Exactly one "A" exists per row.** When multiple teams are deemed accountable, no one is accountable. This is the primary artifact audited under ISO/IEC 42001 (AI Management System).

---

## 9. Practical output: the one-page principles sheet

A checklist directly portable into your project's architectural RFC or ADR:

1. **Six trust zones (Z0–Z5) are explicitly mapped.** External inputs, gateway, inference engine, PEP chokepoint, tool execution, and audit planes are physically separated.
2. **The Propose vs. Commit invariant is strictly enforced.** Model outputs are treated as unverified proposals; actions require deterministic verification.
3. **Model output is untrusted input.** No generated token sequence passes to databases, shells, or SMTP sockets without validation.
4. **Tools operate under user-scoped delegation.** Tools execute with downscoped credentials bounded by the active user's permissions.
5. **High-impact actions are protected by deterministic allowlists.** Execution gates do not rely on natural language classification.
6. **Human approvals are bound to an HMAC parameter digest.** Any alteration between displayed details and executed payloads triggers rejection.
7. **High-risk controls fail closed.** Critical operations halt if guardrail or validation services time out.
8. **Resource ceilings are enforced.** Strict ceilings govern token consumption, execution steps, latency, and financial costs.
9. **Build-time artifacts carry cryptographic provenance.** Weights, adapters, and datasets are signed and verified against an SBOM.
10. **Audit logs follow OpenTelemetry standards using salted token hashes.** Forensic traceability is maintained without persisting raw PII.
11. **Security claims are backed by statistical confidence intervals.** Vague claims of "complete protection" are prohibited.
12. **Decommissioning procedures are established.** Retiring systems triggers cryptographic erasure of weights, vector indices, and keys.

---

## 10. Where this series goes next

Our roadmap to transforming the support assistant into a battle-tested enterprise fortress:

| Phase | Parts | Core Engineering Challenge Addressed |
| --- | --- | --- |
| **Establish Foundation** | 1–2 | Asset taxonomy, six trust zones, data flows, and STRIDE/ATLAS threat models |
| **Secure Data & Models** | 3–4 | Differential privacy, data poisoning, Safetensors, SBOM, and MCP supply chains |
| **Runtime Defenses** | 5–9 | GCG jailbreaks, layered guardrails, RAG access isolation, agent sandboxing, and confidential serving |
| **Measure & Operate** | 10–11 | Automated red teaming, Clopper-Pearson intervals, OTel logging, and incident runbooks |
| **Prove & Release** | 12 | ISO 42001 & NIST AI 600-1 mapping, signed deployment manifests, and release gates |

### What questions can the reader now answer independently?

- Exactly where in the support assistant's architecture does untrusted input intersect with action authority?
- Why does the "Propose vs. Commit" invariant elevate AI security from a prompt-level battle to an architectural guarantee?
- Why does a 4% systemic common cause fraction ($\beta = 0.04$) multiply nominal breach probability by over 40 times?
- What distinguishes safety from security, and why can a system fail catastrophically in one while excelling in the other?
- How should the architectural trade-off between *fail-closed* and *fail-open* be governed across high-impact and low-impact operations?

---

## The whole story in six lines

- An AI system protects four asset classes: data, model and configuration, action authority, and (including cost) service continuity.
- Model outputs are unverified proposals; no action executes without passing a deterministic Policy Enforcement Point (PEP).
- Security, safety, privacy, and reliability answer four distinct questions; no single filter covers all four.
- Layers multiply risk reduction only when independent; a 4% common systemic blind spot worsened the miss rate by 41 times.
- Build-time threats are stealthy and permanent; run-time threats are rapid and repeatable.
- Every security activity must have exactly one accountable owner; residual risk is accepted by the business owner, not security.

---

## Glossary

- **Trust Zone:** A bounded domain separating components with homogeneous trust assumptions; our architecture defines six zones (Z0 through Z5).
- **Propose vs. Commit:** The architectural axiom establishing that LLMs generate unverified proposals, while deterministic policy engines commit state changes.
- **Policy Enforcement Point (PEP):** The deterministic gateway that intercepts proposed tool calls, verifying credentials, schemas, and parameter bounds.
- **Blast Radius:** The maximum real-world damage resulting from the compromise of a specific system component or execution path.
- **Action Authority:** The privilege granted to alter external state (sending emails, modifying databases, initiating transactions).
- **Behavioral Integrity:** The assurance that system execution strictly reflects the authorized user's intent, uncorrupted by adversarial prompts.
- **Common Cause Failure (CCF):** The simultaneous failure of multiple defense layers resulting from a shared root cause, assumption, or representation.
- **Beta-Factor ($\beta$):** A reliability engineering metric denoting the fraction of failures attributable to common cause events across redundant channels.
- **Fail-Closed / Fail-Open:** The architectural policy determining whether a security failure halts execution (closed) or permits bypass (open).
- **Residual Risk:** The unmitigated risk remaining after all defensive controls are deployed, formally accepted by the accountable business owner.

---

## Going deeper

- [K1] NIST, [*AI 600-1: Generative AI Profile to the AI Risk Management Framework*](https://doi.org/10.6028/NIST.AI.600-1) (July 2024): Twelve generative AI risk profiles and architectural trust boundaries.
- [K2] NIST, [*SP 800-218A: Secure Software Development Practices for Generative AI*](https://csrc.nist.gov/pubs/sp/800/218/a/ipd) (2024): Build-time and run-time security controls for AI lifecycles.
- [K3] ISO/IEC, [*42001:2023 Artificial Intelligence Management System*](https://www.iso.org/standard/42001): Governance, organizational roles, and accountability standards.
- [K4] ETSI, [*EN 304 223: Securing Artificial Intelligence, Cyber Security Requirements*](https://www.etsi.org/newsroom/press-releases/2627-etsi-releases-world-leading-standard-for-securing-ai) (January 2026): 13 baseline cybersecurity requirements across five lifecycle phases.
- [K5] IEC, [*61508: Functional Safety of Electrical/Electronic/Programmable Electronic Safety-related Systems*](https://www.iec.ch/functional-safety): Common Cause Failures (CCF) and Beta-Factor reliability modeling.
- [K6] OWASP GenAI Security Project, [*Top 10 for LLM Applications 2025/2026*](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10) and [*Top 10 for Agentic Applications*](https://genai.owasp.org/): Guidance on Excessive Agency and Goal Hijacking.
- [K7] Google, [*Secure AI Framework (SAIF)*](https://safety.google/cybersecurity-advancements/saif/): Foundational security framework for enterprise AI systems.
- [K8] In this series: [Secure AI by Design (2): Threat Modeling and Trust Boundaries in AI Systems](post.html?slug=ai-threat-modeling-and-trust-boundaries).
