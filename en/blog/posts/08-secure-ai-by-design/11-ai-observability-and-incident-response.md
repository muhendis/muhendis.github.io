On a Tuesday morning a customer called the support line: they had received an email about a refund, addressed to them, that described *another* customer's order. The incident team opened the logs. They found a line saying `send_email called` for that case. They could not find which model version had drafted the email, which policy version had approved it, which retrieved chunks had been in the context, or who had clicked "approve". Traces were sampled at 1%, and this request was not in the sample. The prompt and response content that *was* captured had been exported to an observability vendor, so the investigation itself revealed that the other customer's data now also sat in a third-party system. And during the previous week's traffic peak, the audit pipeline had silently dropped events, so nobody could say whether other emails had gone to the wrong people.

Containing the problem took nine days, mostly spent reconstructing what had happened. The controls from Parts 5 to 9 might all have worked except for one edge case, but without records the team could not prove it, could not find the edge case, and could not tell the affected customers who they were.

This article designs the records and the response for the support assistant. We separate traces, security audit events, service logs and metrics, decide what must be recorded so that every decision can be reconstructed with its versions, keep the records from becoming a second copy of sensitive data, make audit delivery durable enough to rely on, detect incidents with signals that do not depend on luck, and turn NIST's incident response guidance into AI-specific containment levers. The practical output is an event schema, a Python forensic replay engine, and an incident response runbook.

> **What we carry over from the previous part:** [Part 10](post.html?slug=measuring-ai-security) measured the system before release. This part measures it in production and prepares for the day a measurement turns into an incident. It relies on the version discipline of [Part 4](post.html?slug=securing-the-ai-supply-chain), the source labels of [Part 5](post.html?slug=prompt-injection-and-the-trust-boundary) and the executor of [Part 8](post.html?slug=least-privilege-for-ai-agents).

> **By the end of this article**
>
> - You will be able to explain the different jobs of traces, audit events, service logs and metrics, and why only some may be sampled.
> - You will be able to define an event schema that ties each AI decision to its model, prompt, policy, sources and approval.
> - You will be able to minimize and mask what is logged with salted HMAC digests so that observability does not create a new leak.
> - You will be able to design audit delivery that fails safe and handle actions whose outcome is unknown.
> - You will be able to operate AI-specific 4-tier containment levers and execute a deterministic forensic replay engine.

**In this article**

- [1. Four kinds of records](#1-four-kinds-of-records)
- [2. Reconstructing a decision: what to record](#2-reconstructing-a-decision-what-to-record)
- [3. Observability without a second leak](#3-observability-without-a-second-leak)
- [4. Durable delivery and the math of sampling](#4-durable-delivery-and-the-math-of-sampling)
  - [4.1. Why audit events must not be sampled](#41-why-audit-events-must-not-be-sampled)
  - [4.2. Delivery that fails safe](#42-delivery-that-fails-safe)
  - [4.3. Actions with unknown outcomes](#43-actions-with-unknown-outcomes)
- [5. Detecting incidents](#5-detecting-incidents)
- [6. Responding: contain, roll back, recover, learn](#6-responding-contain-roll-back-recover-learn)
- [7. Practical output: event schema, audit engine and runbook](#7-practical-output-event-schema-audit-engine-and-runbook)
  - [7.1. Security audit event schema (example)](#71-security-audit-event-schema-example)
  - [7.2. Forensic deterministic replay and outbox audit engine](#72-forensic-deterministic-replay-and-outbox-audit-engine)
  - [7.3. AI security incident runbook (excerpt)](#73-ai-security-incident-runbook-excerpt)
  - [7.4. Two engineering lenses: Build time and run time](#74-two-engineering-lenses-build-time-and-run-time)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. Four kinds of records

An aircraft carries two different recorders. The flight data recorder captures hundreds of parameters continuously and must survive a crash; the maintenance log captures what the crew and engineers chose to write down. Nobody would investigate an accident from the maintenance log alone, and nobody would fill the flight recorder with coffee-machine faults.

An AI system needs the same separation. Four kinds of records are often lumped together as "logs", and treating them alike is how the opening incident happened:

| Record | Purpose | Completeness | Content | Retention | Readers |
| --- | --- | --- | --- | --- | --- |
| **Traces** | Debugging performance and behavior across services | Sampled is acceptable | Timing, attributes; content only by opt-in | Days to weeks | Engineers |
| **Security audit events** | Reconstructing decisions; evidence | **Complete, never sampled** | Identifiers, versions, hashes, decisions; no raw content | Per policy, often a year or more | Security, audit, incident team |
| **Service logs** | Operating the service (errors, restarts, configuration) | Best effort | Technical messages | Weeks | On-call engineers |
| **Metrics** | Trends and alerting | Aggregated | Counts, rates, latencies | Months | Dashboards, alerting |

The key distinction is between traces and audit events. A trace answers *"why was this request slow?"* and losing 99% of traces is fine. An audit event answers *"who approved this email, under which policy?"* and losing a single one can make an incident impossible to scope. They can share instrumentation, but they need different pipelines, storage, access control and retention.

OpenTelemetry's semantic conventions for generative AI give a common vocabulary for traces: operation names such as `chat`, `invoke_agent`, `execute_tool` and `retrieval`, and attributes such as `gen_ai.request.model`, `gen_ai.tool.name`, `gen_ai.conversation.id` and `gen_ai.data_source.id`. The conventions are still marked as *Development*, so names may change; pin the version you implement.

---

## 2. Reconstructing a decision: what to record

The test of an audit design is simple: for any action the system took, can you answer *who asked, what the model saw, what it proposed, which controls decided, who approved, and what actually happened*, together with the version of every component involved? Each question maps to fields:

| Question | Fields in the audit event |
| --- | --- |
| Who asked? | `user_id` (pseudonymous), `session_id`, `task`, client credential |
| What did the model see? | Source list: chunk IDs, collections, trust levels $\tau$ (Part 5), memory IDs; **hash** of the assembled prompt |
| Which model and configuration? | `model` and revision, adapter, chat template, system prompt version |
| What did it propose? | Tool name, **hash of canonical parameters** (Part 8), destination class |
| Which controls decided? | Policy version, detector versions and scores (findings), decision, rule ID |
| Who approved? | Approval token ID, approver, timestamp, displayed-parameter hash |
| What happened? | Execution result, downstream request ID, idempotency key, error |
| Through which path? | Gateway route, primary or fallback, guard mode (Part 9) |

All events for one user request carry the same `trace_id`, and every downstream call (model, guard, tool, MCP server) propagates it. That makes the opening question, *"which emails did this policy version approve last week?"*, a query instead of an investigation.

Two design details matter more than they look:

1. **Hashes instead of content.** The audit event stores the SHA-256 of the prompt and of the canonical tool parameters, not the text. If the content is needed later, it can be retrieved from the source systems (which have their own access control and retention) and checked against the hash. The audit trail proves *what* was seen without becoming a copy of it.
2. **Versions as identifiers, not descriptions.** "Latest policy" is useless in an incident. Every component reports an immutable version identifier, the same ones the release flow in Part 4 signs.

<!-- fig:p11_records -->
<svg viewBox="0 0 720 408" role="img" aria-label="A request traverses gateway, orchestrator, executor, model, and tools; trace_id propagates untouched. Four streams are produced: sampled traces, never-sampled audit events, best-effort service logs, and aggregated metrics." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p11-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p11-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">One Request, Four Record Streams: Purpose, Sampling, and Security Boundaries</text>
<!-- Component chain -->
<rect x="16" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="77" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">① Gateway</text>
<text x="77" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">routes, quotas</text>
<line x1="138" y1="70" x2="154" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="156" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="217" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">② Orchestrator</text>
<text x="217" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">context, τ</text>
<line x1="278" y1="70" x2="294" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="296" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="357" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">③ Executor (PEP)</text>
<text x="357" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">policy, approval</text>
<line x1="418" y1="70" x2="434" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="436" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="497" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">④ Model + Guard</text>
<text x="497" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">inference, score</text>
<line x1="558" y1="70" x2="574" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="576" y="48" width="128" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="640" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">⑤ Tool / MCP</text>
<text x="640" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">action, result</text>
<!-- Trace Context bar -->
<rect x="16" y="100" width="688" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360" y="115" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">trace_id propagates untouched across the service chain via W3C TraceContext, correlating all records</text>
<!-- 4 Stream Cards -->
<!-- 1. Traces -->
<rect x="16" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Traces (APM)</text>
<rect x="26" y="162" width="144" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="98" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:600">Sampled (~1%)</text>
<text x="26" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Purpose: Latency &amp; bottlenecks</text>
<text x="26" y="214" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Scope: Span timings, metadata</text>
<text x="26" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Raw payload: Off by default</text>
<text x="26" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Retention: 7–14 days · SRE</text>
<rect x="24" y="274" width="148" height="28" rx="4" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1"/>
<text x="98" y="292" text-anchor="middle" style="fill:var(--c-danger);font-size:9.5px;font-weight:600">⚠️ Sampling misses attacks</text>
<!-- 2. Audit Events -->
<rect x="190" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.6"/>
<text x="200" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Audit Events</text>
<rect x="200" y="162" width="144" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1"/>
<text x="272" y="175" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-weight:700">100% NEVER SAMPLED</text>
<text x="200" y="198" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Purpose: Compliance &amp; forensics</text>
<text x="200" y="214" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Scope: Identity, version, decision</text>
<text x="200" y="230" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Salted Hashes: Zero PII leak</text>
<text x="200" y="246" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Retention: 1+ year · Audit</text>
<rect x="198" y="274" width="148" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="272" y="292" text-anchor="middle" style="fill:var(--c-success);font-size:9px;font-weight:700">🛡️ Outbox + Seq No + Gap Alarm</text>
<!-- 3. Service Logs -->
<rect x="364" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="374" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Service Logs</text>
<rect x="374" y="162" width="144" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="446" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:600">Best-Effort</text>
<text x="374" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Purpose: Crashes, restarts</text>
<text x="374" y="214" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Scope: Error &amp; warn logs</text>
<text x="374" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Format: Structured JSON</text>
<text x="374" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Retention: 30 days · On-call</text>
<rect x="372" y="274" width="148" height="28" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="446" y="292" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Filtered payloads, error-focused</text>
<!-- 4. Metrics -->
<rect x="538" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="548" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Metrics</text>
<rect x="548" y="162" width="144" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="620" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:600">Aggregated</text>
<text x="548" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Purpose: SLA, quota &amp; budgets</text>
<text x="548" y="214" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Scope: Counters &amp; histograms</text>
<text x="548" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Dimensions: token/s, p99, err%</text>
<text x="548" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Retention: 90+ days · Alerting</text>
<rect x="546" y="274" width="148" height="28" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="620" y="292" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Numeric summary, zero raw text</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="326" width="688" height="68" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="346" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 CORE ENGINEERING RULE: LOGS ARE NOT EVIDENCE, AUDIT EVENTS ARE EVIDENCE</text>
<text x="28" y="365" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Traces and service logs serve performance and debugging; sampling guarantees targeted attacks go unseen.</text>
<text x="28" y="381" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Security audit events are forensic proof: 100% recorded, never sampled, carrying salted hashes instead of raw PII, and sequence-ordered via a Transactional Outbox.</text>
</svg>

---

## 3. Observability without a second leak

The opening incident's most avoidable damage was the second leak: the investigation discovered customer data in the observability vendor's platform because prompts and responses had been captured in traces and exported. Part 2 marked log export as trust boundary TB6; this is where it bites.

The OpenTelemetry GenAI conventions take the same position: instructions, user messages and model outputs are sensitive and often large, so instrumentations should not capture them by default. The setting `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` defaults to `NO_CONTENT`; for production, the conventions recommend storing content externally and recording references, rather than putting it on spans.

Rules for the support assistant:

1. **Content capture off by default** in all environments that see real data. Turn it on only in pre-production with synthetic data, or for a short, approved debugging window with masking applied.
2. **Mask before export, not after.** The masking policy from Part 6 (structured PII by checksum, names by NER) runs in the collector *before* anything leaves the trust boundary. Masking failure means drop, never pass through.
3. **References, not copies.** Spans carry chunk IDs, case IDs and pseudonyms (Part 3), not the text those IDs point to.
4. **Separate access.** Audit events live in a store with restricted access and immutability (write-once or append-only with integrity checks). Engineers who debug performance do not need access to who approved which email.
5. **Retention by type.** Traces expire in days; audit events follow the retention policy agreed with privacy and legal; content, if captured at all, expires fastest.

A rough volume estimate shows why content matters for cost as well as risk. With 100,000 requests a day, six spans per request and about 2 KB per span, traces produce about 1.2 GB a day. Capturing prompt and response content at roughly 10 KB per request adds another 1 GB a day, almost doubling volume, and every byte of it is personal data that must be protected, retained and eventually deleted (Part 3's deletion math applies to observability stores too).

---

## 4. Durable delivery and the math of sampling

### 4.1. Why audit events must not be sampled

Sampling is a fine tool for performance traces and a bad one for security evidence. Suppose an attack produces $k$ suspicious requests and the pipeline keeps each request with probability $s$, independently. The probability that **at least one** of them survives sampling is:

$$
P(\text{seen}) = 1 - (1 - s)^{k}
$$

Read it like this: $(1-s)$ is the chance that one request is dropped, $(1-s)^k$ the chance that all $k$ are dropped, and its complement the chance that at least one is kept.

**Walk** with the 1% trace sampling from the opening story:

```text
s = 0.01, one malicious request (k = 1):     1 − 0.99^1   = 0.01    →  1% chance to have any record
s = 0.01, an attack of 50 requests:          1 − 0.99^50  = 0.395   → 40%
s = 0.01, an attack of 300 requests:         1 − 0.99^300 = 0.951   → 95%
```

A careful attacker who needs only a handful of requests is almost invisible in sampled data, and even when one request is kept, the others needed to scope the incident are not. Audit events for security decisions (every detector finding above a threshold, every policy decision for a tool with risk $\ge 1$, every approval and every execution) are therefore recorded for **every** request. Their volume is small because they contain identifiers and hashes, not content.

### 4.2. Delivery that fails safe

An audit event that is lost is as bad as one never written. Three mechanisms make delivery durable:

1. **Write before acting.** For high-impact tools, the executor writes the "decision" event to durable storage *before* calling the tool, and the "result" event after. If the decision event cannot be written, the action does not happen: fail closed, as Part 1 decided for high-impact actions.
2. **Transactional outbox.** Events are written to a local, durable outbox in the same transaction as the state change, and a separate process forwards them to the audit store with retries. A network outage delays events; it does not lose them.
3. **Gap detection.** Events carry per-producer sequence numbers. The audit store alerts on gaps, so "the pipeline silently dropped events last week" becomes "the pipeline alerted at 14:02 that producer 7 skipped 312 events".

### 4.3. Actions with unknown outcomes

The hardest records are for actions whose result is unknown: the email API timed out, so the executor does not know whether the email was sent. Part 8 used idempotency keys to make retries safe. For the audit trail, the executor records the outcome as `unknown`, and a reconciliation job later queries the downstream system by idempotency key and writes the final result. During an incident, `unknown` outcomes are part of the scope until reconciled.

---

## 5. Detecting incidents

The opening incident was detected by a customer. That is the worst detector: slow, partial and public. Better signals come from the records designed above.

**Behavioral signals** from audit events and metrics, compared against baselines per task:

- A rise in approval requests or in their rejection rate (users rejecting suspicious proposals is a symptom).
- A rise in detector scores on retrieved content from one source or sender.
- Tool calls rejected by the executor's validation (wrong recipient, amount above limit): each one is a near miss.
- Budget exhaustion, unusual token usage or loops.
- Memory writes sent to quarantine (Part 7).

**Canary records.** Place a few synthetic customer records in the index and CRM that no legitimate task should ever retrieve or mention, each with a unique, unguessable marker. If a marker ever appears in a model output, an outgoing email, a log export or a network request, something is reading data it should not. Canaries turn exfiltration from something you may never notice into a high-precision alert.

**Mapping to the SOC.** Alerts carry the threat register ID and, where it applies, the MITRE ATLAS technique (for example `AML.T0051.001` for indirect injection or `AML.T0086` for exfiltration via tool invocation), so that the security operations center can handle AI incidents with the same triage process as others.

---

## 6. Responding: contain, roll back, recover, learn

NIST SP 800-61 Revision 3 (April 2025) reframes incident response as a Community Profile of the Cybersecurity Framework 2.0. Preparation maps to CSF's *Govern*, *Identify* and *Protect* functions, and the handling of an incident maps to *Detect*, *Respond* and *Recover*. The structure fits AI systems well; what AI adds is a new set of containment levers.

<!-- fig:p11_runbook -->
<svg viewBox="0 0 720 410" role="img" aria-label="Six incident response stages in a loop: detect, contain, analyze, eradicate, recover, and govern. On the right, 8 cascaded containment levers across 4 operational tiers." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p11b-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p11b-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">AI Incident Response Lifecycle and Cascaded Containment Levers</text>
<!-- Left: 6-stage lifecycle -->
<!-- 1. Detect -->
<rect x="16" y="48" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.3"/>
<text x="26" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">① DETECT</text>
<text x="178" y="68" text-anchor="end" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Detect</text>
<text x="26" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">canary, drift, near-miss</text>
<line x1="102" y1="100" x2="102" y2="114" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 2. Contain -->
<rect x="16" y="116" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.3"/>
<text x="26" y="136" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">② CONTAIN</text>
<text x="178" y="136" text-anchor="end" style="fill:var(--c-danger);font-size:10px;font-family:var(--font-mono)">Respond</text>
<text x="26" y="156" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">narrowest effective lever</text>
<line x1="102" y1="168" x2="102" y2="182" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 3. Scope -->
<rect x="16" y="184" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.3"/>
<text x="26" y="204" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">③ ANALYZE</text>
<text x="178" y="204" text-anchor="end" style="fill:var(--c-danger);font-size:10px;font-family:var(--font-mono)">Respond</text>
<text x="26" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">audit query, exfil scope</text>
<line x1="188" y1="210" x2="202" y2="210" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 4. Eradicate -->
<rect x="204" y="184" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.3"/>
<text x="214" y="204" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">④ ERADICATE</text>
<text x="366" y="204" text-anchor="end" style="fill:var(--c-warn);font-size:10px;font-family:var(--font-mono)">Respond</text>
<text x="214" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">quarantine, fix policy</text>
<line x1="290" y1="184" x2="290" y2="170" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 5. Recover -->
<rect x="204" y="116" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.3"/>
<text x="214" y="136" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">⑤ RECOVER</text>
<text x="366" y="136" text-anchor="end" style="fill:var(--c-success);font-size:10px;font-family:var(--font-mono)">Recover</text>
<text x="214" y="156" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">read-only → approve → norm</text>
<line x1="290" y1="116" x2="290" y2="102" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 6. Govern -->
<rect x="204" y="48" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="214" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">⑥ GOVERN</text>
<text x="366" y="68" text-anchor="end" style="fill:var(--c-accent-2);font-size:10px;font-family:var(--font-mono)">Govern</text>
<text x="214" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">add tests, update threat</text>
<!-- Loop back path -->
<path d="M290 48 V38 H102 V46" marker-end="url(#p11b-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.3;stroke-dasharray:4 3"/>
<text x="196" y="34" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10px;font-weight:600">readiness feedback</text>
<!-- Left summary status -->
<rect x="16" y="248" width="360" height="74" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="268" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NIST SP 800-61r3 / CSF 2.0 Alignment</text>
<text x="26" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Forensic Replay: Reconstruct attacks via deterministic seed &amp; DB snapshot</text>
<text x="26" y="304" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Phased Recovery: Read-only → Forced approval → Full operational status</text>
<!-- Right: 8 Containment Levers -->
<rect x="392" y="48" width="312" height="274" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="404" y="68" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:700;letter-spacing:.02em">CONTAINMENT LEVERS (Narrow → Broad)</text>
<!-- Tier 1 -->
<rect x="402" y="78" width="292" height="52" rx="4" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1"/>
<text x="410" y="94" text-anchor="start" style="fill:var(--c-danger);font-size:10px;font-weight:700">TIER 1: REQUEST &amp; TOOL (Micro)</text>
<text x="410" y="109" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Disable a single tool / MCP endpoint</text>
<text x="410" y="123" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">• Force human-in-the-loop approval on all tools</text>
<!-- Tier 2 -->
<rect x="402" y="136" width="292" height="52" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1"/>
<text x="410" y="152" text-anchor="start" style="fill:var(--c-warn);font-size:10px;font-weight:700">TIER 2: MEMORY &amp; DATA (Mid)</text>
<text x="410" y="167" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Quarantine suspicious data collection/source</text>
<text x="410" y="181" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">• Freeze episodic memory writes</text>
<!-- Tier 3 -->
<rect x="402" y="194" width="292" height="52" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1"/>
<text x="410" y="210" text-anchor="start" style="fill:var(--c-warn);font-size:10px;font-weight:700">TIER 3: IDENTITY &amp; AGENT (Macro)</text>
<text x="410" y="225" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Revoke delegated credentials &amp; MCP tokens</text>
<text x="410" y="239" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">• Kill switch: Freeze individual or all agents</text>
<!-- Tier 4 -->
<rect x="402" y="252" width="292" height="58" rx="4" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1"/>
<text x="410" y="268" text-anchor="start" style="fill:var(--c-danger);font-size:10px;font-weight:700">TIER 4: SYSTEM &amp; MODEL (Root)</text>
<text x="410" y="283" text-anchor="start" style="fill:var(--c-text);font-size:10px">• Rollback model weights, prompt or guardrails</text>
<text x="410" y="297" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">• Flush prefix / KV caches holding leaked data</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="332" width="688" height="66" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="351" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 OPERATIONAL AXIOM: NEVER IMPROVISE ARCHITECTURE DURING A BREACH</text>
<text x="28" y="370" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Every containment lever must be pre-built, pre-authorized, and exercised before an incident occurs.</text>
<text x="28" y="386" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Response always begins at the narrowest lever (Tier 1) and scales up only if needed. An incident is never closed without a regression test.</text>
</svg>

**Containment levers for the support assistant**, from narrowest to broadest:

| Lever | Effect | Typical use |
| --- | --- | --- |
| Disable one tool | `send_email` or `issue_refund` stops; summaries continue | Suspected misuse of one action |
| Force approval for all actions | Every side effect needs a human | Unclear scope, keep service running |
| Quarantine a source or collection | Retrieval stops using it (Part 7) | Poisoned documents or a malicious sender |
| Freeze memory writes | No new memories; existing ones reviewed | Suspected memory poisoning |
| Revoke credentials | Delegated tokens and MCP server credentials invalidated | Suspected credential misuse |
| Stop an agent or all agents (kill switch) | Executor refuses all proposals from that agent | Active, spreading incident |
| Roll back model, prompt or policy | Previous approved set is loaded (Part 4) | A change caused the incident |
| Purge caches | Cached answers that may contain leaked data are deleted | Data exposure through cache |

Each lever must exist *before* the incident, be tested, and be operable by the on-call team in minutes. A kill switch that requires a code deployment is not a kill switch.

**Scoping** uses the audit trail: *which requests ran under the affected versions, which sources were in their contexts, which actions were proposed, approved and executed, with which outcomes (including `unknown`)?* This is the query the opening team could not run.

**Recovery** includes restoring correct state downstream (reversing wrong refunds where possible, notifying affected customers as the privacy team requires), reconciling `unknown` outcomes, and bringing the system back in steps: first read-only, then with forced approval, then normal.

**Learning** closes the loop with earlier parts: the attack that worked becomes test cases in the development and next held-out sets (Part 10), the threat register gets a new or updated row (Part 2), and the control that failed gets a regression test. An incident that does not change a test or a threat register row will happen again.

---

## 7. Practical output: event schema, audit engine and runbook

### 7.1. Security audit event schema (example)

```json
{
  "event": "tool_decision",
  "event_id": "01J…",
  "producer": "executor-3", "seq": 884213,
  "time": "2026-11-03T09:14:22.418Z",
  "trace_id": "4f2c…",
  "user": "emp-7f3a", "session": "s-91c", "task": "handle_complaint",
  "versions": {"model": "support-7b@a1b2c3", "adapter": "v3", "template": "t2",
               "system_prompt": "p14", "policy": "2026-10-29.3",
               "detectors": {"injection": "inj-clf-v7", "content": "guard-slm-v3"}},
  "context": {"prompt_sha256": "9e1d…", "sources": [
      {"id": "case-8812/email-1", "collection": "complaints", "trust": 0},
      {"id": "crm/cust-4711", "collection": "crm", "trust": 1}], "context_trust": 0},
  "proposal": {"tool": "send_email", "params_sha256": "a168…", "destination_class": "record_email"},
  "findings": [{"detector": "injection", "score": 0.07, "span": "case-8812/email-1"}],
  "decision": {"result": "require_approval", "rule": "send_email.context_trust_lt_3"},
  "approval": {"token_id": "n-81", "approver": "emp-7f3a", "displayed_params_sha256": "a168…"},
  "execution": {"status": "unknown", "idempotency_key": "n-81", "downstream_id": null},
  "route": {"gateway": "g9", "path": "primary", "guards": ["input-policy", "stream-policy"]}
}
```

### 7.2. Forensic deterministic replay and outbox audit engine

The following simplified Python engine does three things: it appends audit events to an outbox with per-producer sequence numbers, lets an incident responder deterministically re-check a past policy decision, and finds gaps in what the audit store received:

```python
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

@dataclass(frozen=True)
class AuditDecisionEvent:
    event_id: str
    producer_id: str
    seq: int
    trace_id: str
    model_version: str
    policy_version: str
    prompt_hash: str
    proposed_tool: str
    params_hash: str
    decision: str

class ForensicReplayEngine:
    def __init__(self, policy_rules: Dict[str, Any]):
        self.policy_rules = policy_rules

    @staticmethod
    def hash_canonical_json(data: Dict[str, Any], salt: str = "") -> str:
        """SHA-256 over sorted-key, whitespace-free JSON (an approximation of RFC 8785 JCS)."""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((canonical + salt).encode("utf-8")).hexdigest()

    def replay_decision(
        self,
        event: AuditDecisionEvent,
        raw_params: Dict[str, Any],
        context_trust: int
    ) -> Tuple[bool, str]:
        """Deterministically recalculates whether a past policy decision was strictly valid."""
        computed_hash = self.hash_canonical_json(raw_params)
        if computed_hash != event.params_hash:
            return (False, "Parameter integrity violation: Hash mismatch!")

        rule_key = f"{event.proposed_tool}.min_trust"
        min_required_trust = self.policy_rules.get(rule_key, 2)

        expected_decision = "allow" if context_trust >= min_required_trust else "require_approval"
        is_consistent = (expected_decision == event.decision)
        return (is_consistent, f"Expected: {expected_decision}, Recorded: {event.decision}")


class AuditOutbox:
    """Per-producer outbox: in production, append() runs in the same transaction as the state change."""

    def __init__(self, producer_id: str):
        self.producer_id = producer_id
        self.seq = 0
        self.pending: List[AuditDecisionEvent] = []

    def append(self, **fields) -> AuditDecisionEvent:
        self.seq += 1
        event = AuditDecisionEvent(producer_id=self.producer_id, seq=self.seq, **fields)
        self.pending.append(event)   # a separate forwarder ships these to the audit store, with retries
        return event

def find_gaps(received_seqs: List[int]) -> List[int]:
    """Sequence numbers missing between the lowest and highest one received."""
    expected = set(range(min(received_seqs), max(received_seqs) + 1))
    return sorted(expected - set(received_seqs))


# Demonstration with the tool_decision event of Section 7.1
params = {"to": "ayse@example.com", "case_id": 8812}
outbox = AuditOutbox("executor-3")
event = outbox.append(
    event_id="01J-demo", trace_id="4f2c", model_version="support-7b@a1b2c3",
    policy_version="2026-10-29.3", prompt_hash="9e1d",
    proposed_tool="send_email",
    params_hash=ForensicReplayEngine.hash_canonical_json(params),
    decision="require_approval",
)
engine = ForensicReplayEngine({"send_email.min_trust": 3})
print("replay, context trust 0 :", engine.replay_decision(event, params, context_trust=0))
print("replay, edited params   :", engine.replay_decision(event, dict(params, to="x@outside.example"), context_trust=0))

# Gap detection on what the audit store received from executor-3
print("missing sequence numbers :", find_gaps([884210, 884211, 884213, 884216]))
```

Console output:

```text
replay, context trust 0 : (True, 'Expected: require_approval, Recorded: require_approval')
replay, edited params   : (False, 'Parameter integrity violation: Hash mismatch!')
missing sequence numbers : [884212, 884214, 884215]
```

### 7.3. AI security incident runbook (excerpt)

1. **Detect and triage (Detect).** Alert received from canary, behavioral signal or report. Assign severity by asset (Part 1) and blast radius. Open an incident record with the threat register ID and ATLAS technique if known.
2. **Contain (Respond).** Apply the narrowest effective lever from the table; escalate to broader levers if scope is unclear within 30 minutes. Record every lever used, with time and operator.
3. **Scope (Respond).** Query audit events by affected versions, sources, tools and time window. List executed actions and `unknown` outcomes. Check canaries and log exports for data exposure.
4. **Eradicate (Respond).** Remove the cause: quarantine sources, fix the policy, roll back the model or prompt, rotate credentials.
5. **Recover (Recover).** Reconcile outcomes, correct downstream state, notify as required, restore service in stages (read-only → forced approval → normal).
6. **Learn (Govern / Identify).** Post-incident review within five working days: timeline from audit events, root cause, new test cases, threat register update, owner sign-off on residual risk.

### 7.4. Two engineering lenses: Build time and run time

**Build time:** the event schema, instrumentation with trace propagation, content capture defaults and masking in the collector, the separate audit store with immutability and access control, the outbox, canary records, alert rules, and containment levers that are built and rehearsed.

**Run time:** complete audit events for every security decision, sampled traces for performance, gap detection, reconciliation of unknown outcomes, and an on-call team that can pull any lever in minutes.

---

## The whole story in six lines

- Traces, security audit events, service logs and metrics have different jobs; traces may be sampled, audit events never.
- An audit event must let you reconstruct who asked, what the model saw, what it proposed, which controls decided, who approved and what happened, with every version.
- Store hashes, IDs and versions instead of content, keep content capture off by default and mask before export, or observability becomes a second leak.
- At 1% sampling, a single malicious request leaves a record with 1% probability; write the decision before acting, use an outbox, detect gaps and reconcile unknown outcomes.
- Canary records, near misses at the executor and behavioral baselines detect incidents faster than customers do.
- Response follows NIST SP 800-61r3 and CSF 2.0 with AI levers prepared in advance: disable tools, force approval, quarantine sources, freeze memory, revoke, kill, roll back, purge, then turn the incident into tests.

---

## Glossary

- **Trace:** A record of one request's path through services, used for debugging; sampling is acceptable.
- **Security audit event:** A complete, tamper-evident record of a security decision, used as evidence; never sampled.
- **Trace ID propagation:** Passing one identifier through every call a request makes, so all records of that request can be joined.
- **Content capture:** Recording prompts and responses in telemetry; off by default because it copies sensitive data.
- **Transactional outbox:** Writing events to local durable storage in the same transaction as the state change, then forwarding them with retries.
- **Gap detection:** Using per-producer sequence numbers to notice lost events.
- **Unknown outcome:** An action whose result was not confirmed (for example after a timeout), kept in scope until reconciled.
- **Canary record:** A synthetic record that no legitimate task should touch; its appearance anywhere signals data access or exfiltration.
- **Containment lever:** A prepared, tested control that limits an incident's spread, such as disabling a tool or forcing approval.
- **Kill switch:** A lever that stops an agent's actions immediately without a code deployment.

---

## Going deeper

- Previous part: [Secure AI by Design (10): Measuring AI Security, Red Teaming and Turkish Evaluation](post.html?slug=measuring-ai-security)
- Next part: [Secure AI by Design (12): From Standards to Evidence, Making AI Security Auditable](post.html?slug=from-standards-to-evidence)
- NIST, [*SP 800-61 Rev. 3: Incident Response Recommendations and Considerations for Cybersecurity Risk Management, a CSF 2.0 Community Profile*](https://csrc.nist.gov/pubs/sp/800/61/r3/final) (April 2025).
- NIST, [*Cybersecurity Framework (CSF) 2.0*](https://www.nist.gov/cyberframework).
- OpenTelemetry, [*Semantic conventions for generative AI*](https://github.com/open-telemetry/semantic-conventions-genai) (status: Development; content capture guidance).
- OWASP, [*AISVS 1.0*](https://owasp.github.io/www-project-artificial-intelligence-security-verification-standard-aisvs-docs/), chapter C12 (monitoring, logging and anomaly detection).
- MITRE [*ATLAS*](https://atlas.mitre.org/) for tagging AI incidents with technique IDs.
- From this blog: [Why does an AI cluster slow down?](post.html?slug=from-gpus-to-ai-supercomputers) for operational telemetry at the infrastructure layer.
