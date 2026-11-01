Two weeks before go-live, a large customer sent a questionnaire. *"Is the assistant compliant with ISO/IEC 42001? Do you follow the OWASP Top 10 for LLM Applications? Is it secure?"* The team had done serious work: a threat register, signed artifacts, a trust rule at every tool call, an executor with parameter-bound approvals, a guarded gateway, an evaluation report and an incident runbook. Yet the first draft of the answer said *"Yes, we are fully compliant with ISO 42001 and OWASP, and the assistant is secure."* Every part of that sentence was wrong. The company was not certified to ISO/IEC 42001. The OWASP Top 10 is an awareness list, not something one complies with. And "secure" without a scope, a method and a version is a claim no evidence can support.

The problem was not missing controls. It was a missing **chain**: from a requirement in some document, to the control that addresses it, to the feature or policy that implements it, to the test that checks it, to the evidence that the test passed on a specific version, to the risk that remains. With that chain, the honest answer is both shorter and stronger: *"Our controls are mapped to ISO/IEC 27090, OWASP AISVS Level 2 and the OWASP LLM and Agentic Top 10 categories; here is the evidence index for version 2026-11, with 87.5% of applicable AISVS requirements verified and the remaining gaps listed with owners and dates."*

This final part of the series builds that chain. We sort the documents by what kind of thing they are and what they let you claim, place ISO/IEC 42001, 23894, 42005 and 27090 in their roles, define the evidence chain and its coverage, assemble the support assistant's unified reference architecture from all twelve parts, and finish with a mapping matrix, an evidence index and a risk-based go-live checklist.

> **What we carry over from the previous parts:** Everything. The assets and owners of [Part 1](post.html?slug=ai-security-starts-at-design), the threat register of [Part 2](post.html?slug=ai-threat-modeling-and-trust-boundaries), the controls of Parts 3 to 9, the evaluation report of [Part 10](post.html?slug=measuring-ai-security) and the audit trail of [Part 11](post.html?slug=ai-observability-and-incident-response) become the links of one evidence chain.

> **By the end of this article**
>
> - You will be able to tell a management system standard, a guidance document, a framework, a taxonomy, an awareness list and a verification standard apart, and what each lets you claim.
> - You will be able to explain the roles of ISO/IEC 42001, 23894, 42005 and 27090.
> - You will be able to build an evidence chain from requirement to residual risk and compute its coverage.
> - You will be able to run a risk-based programmatic go-live gate with explicit acceptance of what remains.

**In this article**

- [1. Not all documents are the same kind of thing](#1-not-all-documents-are-the-same-kind-of-thing)
- [2. The ISO family: management, risk, impact and security](#2-the-iso-family-management-risk-impact-and-security)
- [3. What you can honestly claim](#3-what-you-can-honestly-claim)
- [4. The evidence chain and its coverage](#4-the-evidence-chain-and-its-coverage)
- [5. The support assistant, assembled](#5-the-support-assistant-assembled)
- [6. A risk-based release gate](#6-a-risk-based-release-gate)
- [7. Practical output: mapping matrix, evidence index and go-live checklist](#7-practical-output-mapping-matrix-evidence-index-and-go-live-checklist)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

---

## 1. Not all documents are the same kind of thing

A building inspector, a fire safety guide, a list of the ten most common causes of house fires and a catalogue of arson techniques are all useful when you build a house, but they are different kinds of documents and you use them differently. Nobody says their house "complies with the list of the ten most common causes of fire".

The documents this series has cited fall into seven kinds:

| Kind | Examples | Purpose | Can you be certified? | Typical honest claim |
| --- | --- | --- | --- | --- |
| **Management system standard** | ISO/IEC 42001:2023 (AI), ISO/IEC 27001 (information security), ISO/IEC 27701:2025 (privacy) | Requirements for how an organization governs, assigns roles, manages risk and improves | Yes, by an accredited body, for a defined scope | "Certified to ISO/IEC 42001 by [body], scope [X], certificate [ID]" |
| **Guidance standard** | ISO/IEC 23894:2023 (AI risk), ISO/IEC 42005:2025 (impact assessment), ISO/IEC 27090:2026 (AI security threats) | Recommendations on how to do something | No | "Our process follows the guidance of ISO/IEC 23894" |
| **European / national standard** | ETSI EN 304 223 (January 2026) | Baseline requirements, here 13 principles across five lifecycle phases | Depends on the scheme built on it | "Assessed against EN 304 223, results in [report]" |
| **Framework** | NIST AI RMF 1.0, NIST AI 600-1, NIST CSF 2.0, NIST SP 800-61r3, SP 800-218A | A structure of functions and outcomes to organize work | No | "Organized according to NIST AI RMF functions" |
| **Taxonomy / knowledge base** | NIST AI 100-2 E2025, MITRE ATLAS | Shared names for attacks and mitigations | No | "Threats identified with ATLAS technique IDs" |
| **Awareness list** | OWASP Top 10 for LLM Applications 2026, OWASP Top 10 for Agentic Applications | The most important risk categories, for prioritization | No | "Each Top 10 category is addressed in the threat register" |
| **Verification standard / specification** | OWASP AISVS 1.0 (levels 1–3), OWASP ASVS 5.0, SLSA 1.2, MCP specification | Testable requirements or protocol rules | Not certification, but verifiable | "Verified against AISVS Level 2; evidence index attached" |

The distinction drives everything that follows. A **management system** standard asks *"does the organization have a working process?"*, a **verification standard** asks *"does this system meet these testable requirements?"*, and a **taxonomy** asks *"did you name the threats in a way others understand?"*. Mixing them up leads to claims that sound strong and mean nothing.

---

## 2. The ISO family: management, risk, impact and security

Four ISO/IEC documents divide the work around AI in a way that matches the roles from Part 1:

- **ISO/IEC 42001:2023: Artificial intelligence management system.** Specifies requirements for establishing, implementing, maintaining and continually improving an AI management system: policy, roles and responsibilities, risk and impact processes, controls, monitoring and improvement. It is certifiable. The RACI matrix from Part 1 and the sign-off below are its smallest practical expressions.
- **ISO/IEC 23894:2023: Guidance on risk management in artificial intelligence** (published February 2023). Applies the general risk management process of ISO 31000:2018 to AI, adding AI-specific considerations. It is guidance, not certifiable. The threat register and scoring from Part 2 are one way to execute its process.
- **ISO/IEC 42005:2025: AI system impact assessment** (published May 2025). Provides guidance on assessing the impacts of an AI system on individuals, groups and society. This is a different question from security risk: *who can be affected, and how?* For the support assistant, that covers the customers whose complaints are summarized and the employees whose work changes.
- **ISO/IEC 27090:2026: Cybersecurity guidance for artificial intelligence.** BSI announced it in October 2026 under the title *Addressing security threats and compromises to AI systems*. It describes how attacks such as data poisoning, prompt injection, model theft and exfiltration of training data or inputs affect AI data, models, outputs and services across the lifecycle, and how conventional and AI-specific controls respond. It is guidance designed to work with ISO/IEC 27001, 27002 and 42001 rather than creating a separate management system.

An honesty rule from the plan for this series applies here with particular force: ISO texts are paywalled documents; public summaries and press releases describe their scope, not their clauses. **A clause-level statement of compliance can only be written after the text itself has been read.** Until then, the honest claim is not "compliant with clause X of ISO/IEC 27090" but "mapped to the scope of ISO/IEC 27090 as described in its published summary".

---

## 3. What you can honestly claim

Before writing any sentence that contains the name of a standard, check it against three questions: *What kind of document is it? What exactly was evaluated, against which version, by whom? What was the scope?*

| Avoid | Write instead |
| --- | --- |
| "We are fully compliant with ISO 42001" | "Certified to ISO/IEC 42001:2023 by [body]; scope: AI customer support services, certificate [ID], valid until [date]". If uncertified: "Our AI governance practices are designed around ISO/IEC 42001; certification planned for [date]" |
| "OWASP compliant" | "Each category of the OWASP LLM Top 10 (2026) and Agentic Top 10 is mapped to threats and controls in our register" |
| "Secure", "zero leakage", "total protection" | "Harm rate 0 on evaluation set v4 (1,200 attack cases, EN/TR), 95% upper bound 0.25%; model, policy and dataset versions listed" |
| "We follow NIST" | "Our risk process is organized according to the Govern, Map, Measure and Manage functions of the NIST AI RMF" |
| "We meet AISVS" | "Verified against OWASP AISVS 1.0 Level 2: 112 of 128 applicable requirements evidenced; 16 open items with owners" |

The numbers in the third and fifth rows are examples showing the *form* of a claim. Real reports contain only measured values (Part 10).

---

## 4. The evidence chain and its coverage

The evidence chain links six things, each with a persistent identifier:

1. **Source requirement:** A clause, principle, category or requirement from a document: an AISVS requirement, an EN 304 223 principle, an OWASP category or an ATLAS technique.
2. **Control:** The design decision that addresses it, one of the controls of this series (for example "C-08 recipient is bound to the case record").
3. **Implementation:** The feature, configuration or policy that puts it into practice, with its version (the executor rule `send_email.recipient_in_record`, policy `2026-10-29.3`).
4. **Test:** The check that demonstrates the implementation works, with its pass criterion (PI-1: "200 variants, 0 deliveries to non-customer addresses").
5. **Evidence:** The result of that test on a specific system version, stored immutably (report ID, run ID, artifact hashes).
6. **Residual risk:** What remains, its score, and the owner who accepted it ($R'$ from Part 2, signed according to the RACI of Part 1).

<!-- fig:p12_chain -->
<svg viewBox="0 0 720 236" role="img" aria-label="Six interconnected links: source standard requirement, control definition, code implementation, verification test, signed run evidence, and owner-accepted residual risk." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p12c-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Cryptographic Evidence Chain: 6 Unbroken Links from Standard to Residual Risk</text>
<!-- 6 Chain Links -->
<!-- 1 -->
<rect x="16" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="22" y="54" width="92" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="68" y="67" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:700">① SOURCE</text>
<text x="68" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Requirement</text>
<text x="68" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">AISVS L2 (v1.0)</text>
<text x="68" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">OWASP LLM01</text>
<text x="68" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">ATLAS AML.T0051</text>
<rect x="24" y="136" width="88" height="14" rx="3" style="fill:var(--c-accent);fill-opacity:.12"/>
<text x="68" y="146.5" text-anchor="middle" style="fill:var(--c-accent);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">STANDARD</text>
<line x1="120" y1="101" x2="127" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 2 -->
<rect x="132" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<rect x="138" y="54" width="92" height="18" rx="4" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1"/>
<text x="184" y="67" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-weight:700">② CONTROL</text>
<text x="184" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Policy</text>
<text x="184" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">C-08: Recipient must</text>
<text x="184" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">match authenticated</text>
<text x="184" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">case record</text>
<rect x="140" y="136" width="88" height="14" rx="3" style="fill:var(--c-danger);fill-opacity:.12"/>
<text x="184" y="146.5" text-anchor="middle" style="fill:var(--c-danger);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">SPEC</text>
<line x1="236" y1="101" x2="243" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 3 -->
<rect x="248" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="254" y="54" width="92" height="18" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1"/>
<text x="300" y="67" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-weight:700">③ IMPLEMENT</text>
<text x="300" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">PEP Engine</text>
<text x="300" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Executor rule</text>
<text x="300" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">send_email allowlist</text>
<text x="300" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Policy: 10-29.3</text>
<rect x="256" y="136" width="88" height="14" rx="3" style="fill:var(--c-accent);fill-opacity:.12"/>
<text x="300" y="146.5" text-anchor="middle" style="fill:var(--c-accent);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">CODE</text>
<line x1="352" y1="101" x2="359" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 4 -->
<rect x="364" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="370" y="54" width="92" height="18" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="416" y="67" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10px;font-weight:700">④ TEST</text>
<text x="416" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Regression</text>
<text x="416" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Test PI-1 (ART)</text>
<text x="416" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">200 EN/TR variants</text>
<text x="416" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">0 external exfil</text>
<rect x="372" y="136" width="88" height="14" rx="3" style="fill:var(--c-accent-2);fill-opacity:.12"/>
<text x="416" y="146.5" text-anchor="middle" style="fill:var(--c-accent-2);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">CI/CD</text>
<line x1="468" y1="101" x2="475" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 5 -->
<rect x="480" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="486" y="54" width="92" height="18" rx="4" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1"/>
<text x="532" y="67" text-anchor="middle" style="fill:var(--c-success);font-size:10px;font-weight:700">⑤ EVIDENCE</text>
<text x="532" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Signed Log</text>
<text x="532" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Candidate rc-11.2</text>
<text x="532" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Run #7731 report</text>
<text x="532" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Sigstore SHA-256</text>
<rect x="488" y="136" width="88" height="14" rx="3" style="fill:var(--c-success);fill-opacity:.12"/>
<text x="532" y="146.5" text-anchor="middle" style="fill:var(--c-success);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">MERKLE</text>
<line x1="584" y1="101" x2="591" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 6 -->
<rect x="596" y="48" width="108" height="106" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="602" y="54" width="96" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1"/>
<text x="650" y="67" text-anchor="middle" style="fill:var(--c-warn);font-size:10px;font-weight:700">⑥ RISK</text>
<text x="650" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Residual Risk</text>
<text x="650" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Threat T1: R 20→4</text>
<text x="650" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Appetite: A=6 (OK)</text>
<text x="650" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Owner signed off</text>
<rect x="606" y="136" width="88" height="14" rx="3" style="fill:var(--c-warn);fill-opacity:.12"/>
<text x="650" y="146.5" text-anchor="middle" style="fill:var(--c-warn);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">SIGNED</text>

<!-- Bottom Educational Banner -->
<rect x="16" y="164" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="183" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 EVIDENCE CHAIN PRINCIPLE: A VERIFIED CLAIM REQUIRES ALL SIX LINKS</text>
<text x="28" y="201" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">A security claim is valid if and only if all six links exist and correlate to the exact same release candidate.</text>
<text x="28" y="217" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">When model weights, prompt templates, policies, or tool schemas change, prior audit evidence becomes instantly stale.</text>
</svg>

A requirement is **verified** only if all six links exist and the evidence belongs to the released version. The **coverage** of a verification target is:

$$
\text{Coverage} = \frac{\left|\, V \cap A \,\right|}{\left|\, A \,\right|}
$$

Read it like this: $A$ is the set of requirements applicable to the system (after documented exclusions). $V$ is the set of requirements with a complete chain for this release. $|\cdot|$ counts elements. Coverage is the fraction of applicable requirements that are verified.

**Walk** for an OWASP AISVS 1.0 Level 2 target. AISVS 1.0 assigns each of its 191 requirements to a level: 51 in Level 1, 95 in Level 2 and 45 in Level 3. Verifying at Level 2 means verifying Level 1 and Level 2 requirements:

```text
Level 1 + Level 2 requirements:            51 + 95 = 146
Not applicable (documented rationale,      e.g. no foundation training
  e.g. no pre-training of our own):        − 18
Applicable |A|:                            146 − 18 = 128
Complete chain for this release |V ∩ A|:   112
Coverage = 112 / 128 = 0.875               → 87.5%
Open items: 128 − 112 = 16                 → each must have an owner, a plan or a signed acceptance
```

Two properties make coverage useful rather than cosmetic. First, **exclusions are documented**: "not applicable" requires a rationale an auditor can check, rather than a way to hide gaps. Second, **coverage is version-bound**: a new model or policy version invalidates evidence built on the old one, so coverage can drop on release day if regression suites were not rerun. That is the intended behavior.

---

## 5. The support assistant, assembled

Twelve parts have added layers to one system. Assembled, the reference architecture looks like this:

<!-- fig:p12_arch -->
<svg viewBox="0 0 720 536" role="img" aria-label="Complete reference architecture in five bands. Governance: assets and RACI from Part 1, risk acceptance, evidence index and go-live gate from Part 12. Build time: threat register from Part 2, data lifecycle from Part 3, supply chain and release flow from Part 4, evaluation and red teaming from Part 10. Runtime request path: chat UI, gateway from Part 9, context labels and trust from Part 5, guardrails and policy engine from Part 6, permissioned retrieval and memory from Part 7, isolated model server from Part 9. Runtime action path: executor from Part 8 calling tools and MCP servers with privilege intersection, parameter-bound approvals, budgets and idempotency. Operations: audit events, masked telemetry, canaries and incident runbook from Part 11." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p12a-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p12a-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Secure AI by Design: the support assistant, assembled</text>
<rect x="16" y="44" width="688" height="64" rx="8" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="28" y="61" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:700">GOVERNANCE</text>
<rect x="28" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="156" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P1</text>
<text x="38" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Asset, owner, RACI</text>
<rect x="198" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P2</text>
<text x="208" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Risk acceptance</text>
<rect x="368" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P12</text>
<text x="378" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Evidence index</text>
<rect x="538" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P12</text>
<text x="548" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Go-live gate</text>
<rect x="16" y="118" width="688" height="64" rx="8" style="fill:var(--c-accent-2);fill-opacity:.05;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="28" y="135" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:700">BUILD TIME</text>
<rect x="28" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="156" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P2</text>
<text x="38" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Threat register</text>
<rect x="198" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P3</text>
<text x="208" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Data lifecycle</text>
<rect x="368" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P4</text>
<text x="378" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Supply, release</text>
<rect x="538" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P10</text>
<text x="548" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Evaluation</text>
<rect x="16" y="192" width="688" height="118" rx="8" style="fill:var(--c-warn);fill-opacity:.05;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="28" y="209" text-anchor="start" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">RUNTIME · REQUEST PATH</text>
<rect x="28" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="80" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Chat UI</text>
<rect x="66" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="80.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P5</text>
<line x1="132" y1="241" x2="138" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="140" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="192" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Gateway</text>
<rect x="178" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="192.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P9</text>
<line x1="244" y1="241" x2="250" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="252" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="304" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Labels, trust τ</text>
<rect x="290" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="304.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P5</text>
<line x1="356" y1="241" x2="362" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Guardrails</text>
<rect x="402" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="416.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P6</text>
<line x1="468" y1="241" x2="474" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="476" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="528" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Retrieval, memory</text>
<rect x="514" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="528.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P7</text>
<line x1="580" y1="241" x2="586" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="588" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="640" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Model server</text>
<rect x="626" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="640.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P9</text>
<text x="28" y="296" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Every model call, guard call and fallback path passes through the gateway; every chunk carries its source and trust.</text>
<rect x="16" y="320" width="688" height="70" rx="8" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="28" y="337" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:700">RUNTIME · ACTION PATH</text>
<rect x="28" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<rect x="156" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P8</text>
<text x="38" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Executor (PEP/PDP)</text>
<rect x="198" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P8</text>
<text x="208" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Param-bound approval</text>
<rect x="368" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P8</text>
<text x="378" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Budgets, idempotency</text>
<rect x="538" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P8</text>
<text x="548" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Tools, MCP</text>
<rect x="16" y="400" width="688" height="68" rx="8" style="fill:var(--c-success);fill-opacity:.05;stroke:var(--c-success);stroke-width:1.3"/>
<text x="28" y="417" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:700">OPERATIONS</text>
<rect x="28" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="156" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P11</text>
<text x="38" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Audit events</text>
<rect x="198" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P11</text>
<text x="208" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Masked telemetry</text>
<rect x="368" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P11</text>
<text x="378" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Canaries, alerts</text>
<rect x="538" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">P11</text>
<text x="548" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Runbook, levers</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="468" width="688" height="58" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="487" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 ARCHITECTURAL SYSTEM INTEGRITY: 12 PARTS SYNCHRONIZED ACROSS 5 BANDS</text>
<text x="28" y="504" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Governance establishes risk appetite, build-time guarantees verifiable artifacts, and the request path enforces trust boundaries.</text>
<text x="28" y="519" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">The action path constrains tools via deterministic PEP, and operations captures forensic proof. If any band is omitted, security collapses.</text>
</svg>

Reading from the outside in:

- **Governance (Parts 1, 12):** Assets, owners and RACI; risk acceptance; evidence index; go-live gate.
- **Build time (Parts 2, 3, 4, 10):** Threat register; data provenance, PII handling and deletion; artifact admission, signing and release flow; evaluation and red teaming.
- **Runtime, request path (Parts 5, 6, 7, 9):** Gateway with quotas and guards on every path; source-tagged context and trust propagation; detectors and policy engine; permissioned retrieval with quarantined memory; streaming rollback.
- **Runtime, action path (Part 8):** Executor holding credentials; privilege intersection; parameter validation; parameter-bound approval; budgets; idempotency.
- **Operations (Part 11):** Complete audit events, masked telemetry, canaries, containment levers and runbook.

---

## 6. A risk-based release gate

Going live is a decision under uncertainty. The gate turns the outputs of earlier parts into a decision rule the business owner can sign. Denoting the residual risk of threat $i$ by $R'_i$ (Part 2), the agreed risk appetite by $A$, a signed acceptance by $\text{accept}_i$, the 95% upper bound on harm rate by $\bar{h}$ (Part 10) and its limit by $\bar{h}_{\max}$:

$$
\text{Pass} = \Big( \bigwedge_{i} \big[\, R'_i \le A \;\lor\; \text{accept}_i \,\big] \Big) \land \big[\, \bar{h} \le \bar{h}_{\max} \,\big] \land T_{\text{accept}} \land E_{\text{full}}
$$

Read it like this: $\bigwedge$ means "for every threat"; every threat's residual risk is either within appetite or explicitly accepted by its owner. In addition, the measured harm rate's upper bound must be within its limit, all acceptance tests $T_{\text{accept}}$ (Part 9) must pass, and the evidence index $E_{\text{full}}$ must be complete for the released version.

**Walk** with the threat register from Part 2 and an appetite of $A = 6$:

```text
Residual scores R':      T1 = 4   T2 = 4   T3 = 4   T4 = 4   T5 = 5   T6 = 4
Appetite A = 6:          all R' ≤ 6                            → holds without exception

Evaluation (Part 10):    harm rate 0 on 1,200 attack cases
                         upper bound ≈ 3 / 1,200 = 0.0025 (0.25%)
                         limit h̄_max = 0.5%                    → 0.25% ≤ 0.5%, holds

Acceptance tests S-1..S-10 (Part 9):   all pass on release candidate rc-2026-11.2
Evidence index:          complete for rc-2026-11.2 (coverage 87.5%; 16 open items owned)

Pass = true ∧ true ∧ true ∧ true = true   → business owner signs, review date set
```

If T5 (tampered model artifact) had scored 8 after controls, the gate would have demanded either further controls or the business owner's signed acceptance of that specific risk with a review date, not a silent pass. And if the release candidate changed the model after evaluation, $E_{\text{full}}$ would be false until regression tests were rerun.

The gate does not claim the system is "secure". It records that for this version and this scope, known risks are within the appetite agreed by the people accountable for them, based on verifiable evidence.

---

## 7. Practical output: mapping matrix, evidence index and go-live checklist

### 7.1. Standard mapping matrix (excerpt)

| Series control | ETSI EN 304 223 phase | NIST AI RMF | OWASP LLM 2026 / Agentic | AISVS 1.0 | ATLAS | Other |
| --- | --- | --- | --- | --- | --- | --- |
| Assets, owners, RACI (P1) | Design | Govern, Map | – | – | – | ISO/IEC 42001 roles |
| Threat register (P2) | Design | Map | All categories as checklist | – | Technique ID per row | ISO/IEC 23894 process |
| Data provenance, PII, deletion (P3) | Development, decommissioning | Map, Manage | LLM02, LLM05 | C1 | `AML.T0020` | ISO/IEC 27701, KVKK, GDPR |
| Artifact admission, signing, release (P4) | Development, deployment | Manage | LLM04, ASI04 | C3, C6 | `AML.T0010` | SLSA 1.2, NIST SP 800-218A |
| Trust propagation, spotlighting (P5) | Deployment | Manage | LLM01, ASI01 | C2, C9 | `AML.T0051` | – |
| Detectors and policy engine (P6) | Deployment | Measure, Manage | LLM01, LLM02 | C2, C7 | – | – |
| Permissioned retrieval, memory gate (P7) | Deployment | Manage | LLM09, ASI06 | C8 | `AML.T0070`, `AML.T0080` | – |
| Executor, approvals, budgets (P8) | Deployment | Manage | LLM03, LLM06, ASI02, ASI03 | C5, C9, C10 | `AML.T0086` | MCP specification, RFC 8693 |
| Gateway, isolation, streaming (P9) | Deployment | Manage | LLM06, LLM10 | C4 | `AML.T0029`, `AML.T0034` | OWASP API Top 10, ASVS 5.0 |
| Evaluation and red teaming (P10) | Maintenance | Measure | All | C11 | – | – |
| Audit, monitoring, response (P11) | Maintenance | Manage | – | C12 | Tag on alerts | NIST SP 800-61r3, CSF 2.0 |

The matrix maps controls to *categories and phases*, which is what public descriptions of these documents support. Clause-level mappings to ISO texts are added only after the texts themselves have been examined.

### 7.2. Evidence index architecture

```text
evidence-index/rc-2026-11.2/
  manifest.json                 # system versions: model, adapter, template, prompt, policy, gateway, tools
  requirements/aisvs-l2.csv     # requirement → applicable? rationale → control IDs → test IDs → status
  controls/C-08.md              # design, implementation reference, owner
  tests/PI-1/run-7731.json      # test definition, dataset hash, results, Wilson interval, judge κ
  acceptance/S-1..S-10.json     # serving acceptance results
  evaluation/report-v4.pdf      # Part 10 report
  risk/register.csv             # threats, R, R', acceptance signatures, review dates
  approvals/go-live.sig         # business owner and security sign-off
```

Every file carries the release candidate's identifier and is stored immutably, so that the `rc-2026-11.2` directory says the same thing a year later.

### 7.3. Programmatic go-live verification engine

The following Python script reads the evidence index files and objectively evaluates the $\text{Pass}$ decision rule within the CI/CD pipeline:

```python
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class ReleaseCandidateEvidence:
    release_id: str
    manifest_hash: str
    residual_risks: Dict[str, int]
    risk_appetite: int
    signed_acceptances: List[str]
    harm_rate_upper_bound: float
    max_harm_limit: float
    acceptance_tests_passed: bool
    unverified_gap_count: int

def evaluate_release_gate(evidence: ReleaseCandidateEvidence) -> Tuple[bool, List[str]]:
    """Evaluates whether an AI release candidate satisfies all compliance & risk gates."""
    violations = []

    # 1. Residual risk appetite check
    for threat_id, score in evidence.residual_risks.items():
        if score > evidence.risk_appetite and threat_id not in evidence.signed_acceptances:
            violations.append(
                f"Unaccepted residual risk: {threat_id} (Score={score} > Appetite={evidence.risk_appetite})"
            )

    # 2. Measured harm rate upper bound check
    if evidence.harm_rate_upper_bound > evidence.max_harm_limit:
        violations.append(
            f"Harm rate upper bound exceeded: {evidence.harm_rate_upper_bound*100:.2f}% > {evidence.max_harm_limit*100:.2f}%"
        )

    # 3. Gateway and serving acceptance tests
    if not evidence.acceptance_tests_passed:
        violations.append("Serving and gateway acceptance test suite (S-1..S-10) failed!")

    # 4. Evidence index completeness check
    if evidence.unverified_gap_count > 0:
        violations.append(f"Evidence index contains {evidence.unverified_gap_count} unverified open items.")

    passed = len(violations) == 0
    return (passed, violations)
```

### 7.4. Go-live checklist

1. Asset inventory and RACI are current; each row has a single accountable owner.
2. Threat register reviewed for this release; every threat has a control, a test and a residual score.
3. Data inventory, PII flow and deletion test pass; no raw personal data in training or eval sets.
4. All artifacts in the AI-BOM signed and verified on load; previous approved set deployable.
5. Trust propagation, executor validation, approvals, budgets and idempotency tested on every tool with side effects.
6. Retrieval applies permissions inside the query; memory writes from untrusted contexts quarantined.
7. Gateway acceptance tests S-1 to S-10 pass; no path bypasses guards; streaming rollback enabled.
8. Evaluation report with intervals per language exists for this release candidate; regression gates pass.
9. Audit events complete; content capture off; canaries in place; containment levers rehearsed.
10. Evidence index complete for this release; coverage and open items stated.
11. Outward claims (website, contracts, questionnaires) use only the honest forms of Section 3.
12. Business owner signs the gate decision with a review date.

### 7.5. Two engineering lenses: Build time and run time

**Build time:** the mapping matrix, requirement applicability list, evidence index structure and gate rule are written once and updated with every change.

**Run time:** evidence is produced continuously by the regression suite, acceptance tests and audit trail, so that each release candidate's directory comes together on its own rather than being written by hand the week before an audit.

---

## The whole story in six lines

- Management system standards, guidance, frameworks, taxonomies, awareness lists and verification standards are different kinds of documents, each permitting a different claim.
- ISO/IEC 42001 governs the organization, 23894 guides risk management, 42005 impact assessment, and 27090 AI security controls together with 27001 and 42001.
- Never claim clause-level compliance from a public summary; replace "secure" with a scoped, versioned and measured statement.
- The evidence chain links requirement, control, implementation, test, evidence and residual risk; coverage counts only complete chains for the released version.
- In the walk, 112 of 128 applicable AISVS Level 2 requirements were verified: 87.5% coverage and 16 owned open items.
- Go-live is a signed decision: residual risks within appetite or accepted, harm rate within limit, acceptance tests passed, and evidence complete.

---

## Glossary

- **Management system standard:** A standard of organizational requirements (roles, processes, improvement) certifiable for a defined scope.
- **Guidance standard:** A standard offering recommendations on how to do something, against which one cannot be certified.
- **Framework:** A structure of functions and outcomes to organize work, such as the NIST AI RMF.
- **Taxonomy:** Shared classification and terms for attacks and mitigations, such as NIST AI 100-2 or MITRE ATLAS.
- **Awareness list:** A prioritized list of important risk categories, such as the OWASP Top 10, used for prioritization rather than compliance.
- **Verification standard:** A set of testable requirements, often tiered, against which a system can be verified.
- **Evidence chain:** The linked sequence from source requirement to control, implementation, test, evidence and residual risk.
- **Coverage:** The fraction of applicable requirements that have a complete evidence chain for the released version.
- **Risk appetite:** The level of residual risk an accountable owner is prepared to accept without a special sign-off.
- **Evidence index:** A versioned, immutable collection of documents and test results supporting a release decision.

---

## Going deeper

- Previous part: [Secure AI by Design (11): Observability and Incident Response for AI Operations](post.html?slug=ai-observability-and-incident-response)
- Start of the series: [Secure AI by Design (1): AI Security Starts at Design](post.html?slug=ai-security-starts-at-design)
- ISO/IEC, [*42001:2023 Artificial intelligence management system*](https://www.iso.org/standard/42001), [*23894:2023 Guidance on risk management*](https://www.iso.org/standard/77304.html), [*42005:2025 AI system impact assessment*](https://www.iso.org/standard/42005) and [*27090:2026*](https://www.iso.org/standard/27090); BSI, [*BS ISO/IEC 27090:2026 announcement*](https://www.bsigroup.com/en-GB/insights-and-media/media-centre/press-releases/2026/october/new-international-guidance-to-help-organizations-protect-ai-from-security-threats/) (October 2026).
- ETSI, [*EN 304 223*](https://www.etsi.org/newsroom/press-releases/2627-etsi-releases-world-leading-standard-for-securing-ai) (January 2026).
- NIST, [*AI RMF 1.0*](https://www.nist.gov/itl/ai-risk-management-framework), [*AI 600-1*](https://doi.org/10.6028/NIST.AI.600-1), [*AI 100-2 E2025*](https://csrc.nist.gov/pubs/ai/100/2/e2025/final), [*SP 800-218A*](https://csrc.nist.gov/pubs/sp/800/218/a/final), [*SP 800-61r3*](https://csrc.nist.gov/pubs/sp/800/61/r3/final).
- OWASP, [*AISVS 1.0*](https://owasp.github.io/www-project-artificial-intelligence-security-verification-standard-aisvs-docs/), [*ASVS 5.0*](https://asvs.owasp.org/), [*Top 10 for LLM Applications 2026*](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10), [*Top 10 for Agentic Applications*](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/), [*Agent Control Standard*](https://genai.owasp.org/resource/agent-control-standard-acs/).
- MITRE, [*ATLAS*](https://atlas.mitre.org/); SLSA, [*v1.2*](https://slsa.dev/spec/v1.2/).