Your RAG demo answered everything. Three weeks into production, a user
types "my dishwasher is leaking underneath" and the bot — calm, fluent,
certain — explains how to replace the door seal. Wrong page, wrong
part, delivered with a citation.

At that moment every fix looks equally plausible: bigger chunks? a
reranker? an agent? This article's one claim: **RAG failures look
infinite, but there are only four families — and once you name the
family, the fix picks itself.** For each family: the symptom you
observe, the root cause, the treatment, and the price on the label.

**In this article**

- [1. The machine you already have](#1-the-machine-you-already-have)
- [2. The question doesn't match the documents](#2-the-question-doesnt-match-the-documents)
  - [Close the gap from the document side: hypothetical questions](#close-the-gap-from-the-document-side-hypothetical-questions)
  - [Close it from the query side: HyDE](#close-it-from-the-query-side-hyde)
  - [Split it or zoom out: sub-queries and step-back](#split-it-or-zoom-out-sub-queries-and-step-back)
- [3. The index hides what it holds](#3-the-index-hides-what-it-holds)
  - [Retrieve small, read big](#retrieve-small-read-big)
  - [Two searchers and a judge](#two-searchers-and-a-judge)
  - [Embed one sentence, read the window](#embed-one-sentence-read-the-window)
  - [Filter before you search](#filter-before-you-search)
- [4. Retrieval worked — the answer is still wrong](#4-retrieval-worked-the-answer-is-still-wrong)
- [5. The pipeline treats every question the same](#5-the-pipeline-treats-every-question-the-same)
- [6. The decision, on one page](#6-the-decision-on-one-page)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

## 1. The machine you already have

**RAG (retrieval-augmented generation)** bolts a library onto a frozen
LLM (why a bare LLM makes things up: [the hallucination
story](post.html?slug=how-llms-work)). Documents are split into
**chunks**, each chunk becomes an **embedding** in a **vector
database**; the query is embedded onto the same map, the **top-k**
nearest chunks are pasted into the prompt, and the model answers from
them.

![The vanilla RAG pipeline: documents are chunked into a vector store; the query retrieves the top-k relevant chunks, which are injected into the prompt the LLM answers from](../../assets/img/rag/vanilla_rag.png)

That machine can break in exactly four places: the question you
search with (section 2), the index you search in (section 3), the way
the model reads what retrieval found (section 4), and the pipeline
that never checks itself (section 5).

Before treating anything, measure one number:

> **retrieval hit rate** = failed questions whose correct passage
> appears in the retrieved top-k ÷ all failed questions

Collect twenty failed questions, find each correct passage by hand,
and check whether it was among the retrieved chunks. **Not there → a
search problem (sections 2–3). There, and the answer still wrong → a
reading problem (sections 4–5).** That one test rules out half the
catalog before you touch anything.

## 2. The question doesn't match the documents

**Symptom:** the hit rate is low, yet you can find the passage by hand.

**Root cause:** the user writes casual symptoms ("leaking
underneath"); the manual writes formal solutions ("verify the drain
hose coupling"). An embedding model maps both faithfully — into two
different neighborhoods. The search answers exactly what you asked:
*what sounds like my sentence?*

### Close the gap from the document side: hypothetical questions

> **Hypothetical questions** = at indexing time, have the LLM write the
> questions each chunk answers, and embed *those* instead of the chunk —
> search becomes question-to-question.

For the drain-hose paragraph, one generated question will be "why is
water pooling under my dishwasher?" — the user's question now matches
questions, not manual prose. It is the shopkeeper labeling shelves by
what customers ask, not by the supplier's catalog.

![Hypothetical questions: at indexing time an LLM generates questions for each chunk into their own vector store; the user's query runs a query-to-query search, and the matched questions' chunks go to the LLM](../../assets/img/rag/hypothetical_question.png)

**The price:** 6,000 chunks × 3 questions ≈ 18,000 generations — paid
once, at write time, and again on every re-index.

### Close it from the query side: HyDE

> **HyDE (hypothetical document embeddings)** = have the LLM draft a
> *fake* answer to the query, embed the fake, and search with it — the
> real paragraphs sitting near it come back.

The mirror image. The fake answer is written in the manual's own
voice — "If water pools under the appliance, the drain hose coupling
may have loosened…" — its facts may be wrong, but its *shape* is
right, and shape is all the embedding sees. A police sketch: nobody
claims it *is* the suspect; it only has to point at the right photo.

![HyDE: the LLM first generates fake answer documents from the query; their embeddings run a doc-to-doc search against the vector store, and the real top-k chunks go to the LLM](../../assets/img/rag/hyde.png)

**The price:** +1 LLM call and its latency on *every* query — and in a
niche domain the fake answer can confidently retrieve the wrong
paragraphs. The decision rule for the pair:

> HyDE pays per query, forever. Hypothetical questions pay per corpus,
> once. Heavy traffic on a stable corpus → enrich the documents. A fast-
> changing corpus with light traffic → enrich the query.

### Split it or zoom out: sub-queries and step-back

> **Sub-queries** = split a compound question into simpler parts,
> retrieve for each part separately, and let the LLM merge the results.

**Symptom:** "compare the eco and intensive programs" — no chunk holds
that comparison; the eco specs live on page 41, the intensive on
page 57. Decompose, retrieve per part, and let the model do the
comparing.

![Sub-queries: the original query is decomposed into sub-query 1 and sub-query 2; each retrieves its own top-k chunks from the vector store, and the LLM answers from the merged results](../../assets/img/rag/sub_query.png)

> **Step-back prompting** = abstract an over-specific question one
> level up, retrieve with the general question, then answer the
> specific one from the general material.

**Symptom:** "why does my 2019 X-500 beep twice after a power cut?" —
no chunk is that specific. Ask "what do the beep codes mean?" first:
zoom the map out to find the neighborhood before hunting the street.

![Step-back prompting: the original question is abstracted into a step-back question, which retrieves chunks and yields a step-back answer; a second LLM pass combines it with the original question into the final answer](../../assets/img/rag/stepback.png)

**The price:** each decomposition is an extra LLM call, each sub-query
a full extra retrieval — cost and latency multiply with the parts.

## 3. The index hides what it holds

**Symptom:** the hit rate is low even with well-phrased queries — and
the junk has a *texture*: context-free fragments, lookalikes, or the
right page from the wrong year.

**Root cause:** how the text was cut and what was stored beside it
decided what could ever be found. The texture tells you which decision
to revisit.

### Retrieve small, read big

**Symptom:** a 128-token chunk embeds precisely but arrives naked
("Tighten it a quarter turn." — tighten *what*?); a 1,024-token chunk
has context but embeds as a blur of ten topics. The fix: **separate
the search unit from the reading unit** — a book's index: a pointer
built to be *findable*, pointing at a page built to be *readable*.

> **Parent-child retrieval (small-to-big)** = embed small child
> chunks; when a child matches, hand the LLM its larger parent.

![Merge chunks automatically: small child chunks are embedded in the vector store; matched children resolve to their parent chunks, and the parent goes to the LLM](../../assets/img/rag/merge_chunks.png)

> **Hierarchical index** = a two-level index: search document
> *summaries* first, then search chunks only inside the winning
> documents.

![Hierarchical index: a summary-vector index selects relevant documents first; the chunk-level search then runs only inside the selected documents](../../assets/img/rag/hierarchical_index.png)

**The price:** index bookkeeping (every child must know its parent),
larger prompts, and one generated summary per document.

### Two searchers and a judge

**Symptom:** "error E24" retrieves the *error codes overview*, not the
E24 entry — dense embeddings blur rare literal tokens (codes, part
numbers, names), while lexical **BM25** never misses an exact token
and never sees a paraphrase. Complementary blind spots → run both.

> **Hybrid retrieval** = run a dense (embedding) searcher and a
> sparse, lexical searcher (**BM25** or **SPLADE**) side by side;
> merge their ranked lists with **RRF**; let a **cross-encoder**
> reranker re-grade the finalists.

The two searchers return incomparable scores, so merge by *rank*:

> RRF(doc) = Σ over searchers 1 ÷ (60 + rank in that searcher's list)

| document | dense rank | lexical rank | RRF score |
|---|---|---|---|
| troubleshooting entry "E24" | 2 | 2 | 1/62 + 1/62 ≈ **0.0323** |
| error codes overview | 1 | — | 1/61 ≈ 0.0164 |
| drain pump replacement | — | 1 | 1/61 ≈ 0.0164 |

**A consistent second beats a lonely first** — and nobody ever compares
a cosine to a BM25 score. Then the judge: a **cross-encoder** reads
query and candidate *together*, far more accurately than the
**bi-encoder** that embedded them apart — and far too slowly for 6,000
chunks, so it only re-grades the top ~20 finalists.

![Hybrid retrieval and reranking: the query goes to both the vector store and BM25 retrieval in parallel; each returns its own top-k chunks, a reranker merges and re-grades them, and the winners go to the LLM](../../assets/img/rag/hybrid_and_rerank.png)

**The price:** a second index to keep in sync, and ~100 ms-class
reranker inference per query.

### Embed one sentence, read the window

> **Sentence-window retrieval** = embed a single sentence; on a match,
> hand the LLM a wider window of the sentences around it.

The chunk dilemma's sharpest resolution, one zoom finer than
parent-child: the single sentence is the most precise search unit
there is; the window restores the context it lacks.

![Sentence window retrieval: the vector store matches one sentence, but a wider window of surrounding sentences is what gets handed to the LLM](../../assets/img/rag/sentence_window.png)

**The price:** the window is a dial — too narrow brings back naked
fragments, too wide brings back noise.

### Filter before you search

**Symptom:** semantically perfect, factually wrong — the 2021 manual's
spec served to a 2024 owner. Embeddings cannot see the year, the
edition, or access rights.

> **Metadata filtering** = store structured fields (product, year,
> language, access level) beside every chunk; filter on them *first*,
> rank by similarity only within what survives.

Passport control before the face-recognition gate: however well the
face matches, the wrong passport ends the conversation.

![Metadata filtering: the vector store returns the top-k relevant chunks, which are then filtered by metadata before reaching the LLM](../../assets/img/rag/metadata_filtering.png)

**The price:** nearly free at query time; the real bill is metadata
discipline at ingestion.

Search problems, on one card:

| what you observe | example | root cause | treatment | price |
|---|---|---|---|---|
| findable by hand, missed by retrieval | "leaking underneath" vs the drain-hose paragraph | style gap between query and documents | HyDE / hypothetical questions | LLM calls per query / per corpus |
| compound or over-specific questions fail | "compare the eco and intensive programs" | wrong unit of search | sub-queries / step-back | calls × number of parts |
| context-free fragments, blurry matches | "Tighten it a quarter turn." — tighten *what*? | one chunk size doing two jobs | small-to-big / sentence-window / hierarchical | bookkeeping, larger prompts |
| exact IDs and codes missed | "error E24" → the overview chapter | dense search blurs rare tokens | hybrid + RRF + reranker | second index, rerank latency |
| right content, wrong version or audience | the 2021 manual → a 2024 owner | embeddings can't see metadata | metadata filtering | discipline at ingestion |

## 4. Retrieval worked — the answer is still wrong

**Symptom:** the section-1 test comes back *positive* — the passage is
in the top-k — and the answer is still bad. The problem moved from
finding to *reading*. Liu et al. (2023) moved the answer-bearing
document through twenty retrieved ones; accuracy traced a U — strong
at the edges, collapsing in the middle, at the dip *worse than giving
the model no documents at all*:

<svg viewBox="0 0 480 320" role="img" aria-label="U-shaped curve of answer accuracy versus the position of the correct document among 20 retrieved documents: about 75 percent at position 1, dipping to about 54 percent in the middle, recovering to about 63 percent at position 20. A dashed horizontal line at about 56 percent marks closed-book accuracy with no documents at all" style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<line x1="50" y1="260" x2="460" y2="260" style="stroke:var(--c-border);stroke-width:1.5"/>
<line x1="50" y1="260" x2="50" y2="20" style="stroke:var(--c-border);stroke-width:1.5"/>
<g style="stroke:var(--c-border);stroke-width:1">
<line x1="50" y1="260" x2="50" y2="265"/><line x1="134" y1="260" x2="134" y2="265"/><line x1="239" y1="260" x2="239" y2="265"/><line x1="345" y1="260" x2="345" y2="265"/><line x1="450" y1="260" x2="450" y2="265"/>
<line x1="45" y1="60" x2="50" y2="60"/><line x1="45" y1="140" x2="50" y2="140"/><line x1="45" y1="220" x2="50" y2="220"/>
</g>
<g style="fill:var(--c-text-mute);font-size:11px" text-anchor="middle">
<text x="50" y="277">1</text><text x="134" y="277">5</text><text x="239" y="277">10</text><text x="345" y="277">15</text><text x="450" y="277">20</text>
</g>
<g style="fill:var(--c-text-mute);font-size:11px" text-anchor="end">
<text x="40" y="64">75%</text><text x="40" y="144">65%</text><text x="40" y="224">55%</text>
</g>
<text x="455" y="296" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">position of the correct document among 20</text>
<text x="18" y="140" transform="rotate(-90 18 140)" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">answer accuracy</text>
<line x1="50" y1="212" x2="450" y2="212" style="stroke:var(--c-accent-2);stroke-width:1.8;stroke-dasharray:6 5"/>
<text x="160" y="204" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">no documents at all (closed-book)</text>
<path d="M 50 60 C 90 130, 110 196, 134 196 C 170 224, 205 228, 239 228 C 280 224, 315 218, 345 212 C 390 198, 425 172, 450 156" fill="none" style="stroke:var(--c-accent);stroke-width:2.5"/>
<circle cx="50" cy="60" r="4.5" style="fill:var(--c-text)"/>
<circle cx="239" cy="228" r="4.5" style="fill:var(--c-text)"/>
<circle cx="450" cy="156" r="4.5" style="fill:var(--c-text)"/>
<text x="62" y="52" text-anchor="start" style="fill:var(--c-text);font-size:13px">≈75%</text>
<text x="239" y="250" text-anchor="middle" style="fill:var(--c-text);font-size:13px">≈54%</text>
<text x="444" y="146" text-anchor="end" style="fill:var(--c-text);font-size:13px">≈63%</text>
</svg>

(Redrawn, approximate, from Liu et al. 2023 — "lost in the middle".)
It is the serial-position effect — read ten CVs, remember the first
and the last. The model learned to read from us. Two treatments — the
second is free:

> **Context compression** = an extraction pass that keeps only the
> retrieved sentences relevant to *this* query, shrinking the prompt.

Twenty chunks ≈ 6,000 tokens; maybe 600 bear on the question — the
rest is noise you pay for and the model gets distracted by. Extract:
6,000 in, 900 out, ~85% of input cost gone. Pack the item, leave the
box.

![Compress prompt: the retrieved top-k chunks are compressed into only the information relevant to the query before being passed to the LLM](../../assets/img/rag/compress_prompt.png)

**The price:** +1 call per query — and it may drop the sentence you
needed; compress *after* reranking so the confidence order protects
the finalists.

> **Reordering** = place the highest-confidence chunks at the start
> and end of the context, the weakest in the middle.

You control the paste order, and the model has edge bias — the
reranker already sorted candidates by confidence, so the order is
lying around waiting to be used. **The price: nothing.** The rare free
lunch — in practice, do it first.

![Adjust chunk sorting in the prompt: the top-k chunks arrive ranked 1, 2, 3 and are reordered to 1, 3, 2 so the highest-confidence chunks sit at both ends of the context](../../assets/img/rag/adjust_order.png)

## 5. The pipeline treats every question the same

**Symptom pair:** "thanks, that fixed it!" triggers a full
embed-search-stuff cycle; and when retrieval silently fails, the model
answers anyway — hallucination *with citations*.

**Root cause:** a fixed pipe. Nobody asks whether retrieval was
needed; nobody checks whether it worked. Two patterns give it
judgment — one at the exit, one at the entrance.

### The editor who checks sources: self-reflection

> **Self-reflection (corrective RAG)** = grade the retrieved chunks
> before answering; on a failing grade, rewrite the query and
> re-retrieve, or fall through to web search.

The grader is an LLM or a small natural-language-inference model;
**Self-RAG** goes further and trains the critique into the model as
reflection tokens. The newspaper editor who reads the reporter's
sources before publishing — and sends the reporter back out.

![Self-reflection: the retrieved top-k chunks are graded as correct or ambiguous; ambiguous ones are verified with an internet search, and only the final relevant chunks reach the LLM](../../assets/img/rag/self_reflection.png)

**The price:** the loop can run twice or three times — worst-case
latency ×2–3. You buy a quality *floor* with *tail* latency: the
pattern for "never confidently wrong", not for "always fast".

### The triage nurse: routing

> **Query routing** = a cheap classifier in front of the pipeline that
> decides, per message: answer directly, retrieve, decompose into
> sub-queries, or search the web.

The router can be an LLM call, a small model, or rules. Small talk →
answer directly; corpus question → RAG; fresh facts → web search.
Nobody gets an MRI for a paper cut.

![Query routing: an agent first decides whether the query needs RAG at all; yes goes through the retrieval pipeline, no goes straight to the LLM](../../assets/img/rag/query_routing.png)

The same router can choose *between* strategies — here, whether to
decompose into sub-queries first:

![Query routing with sub-queries: the agent decides whether to split the query; yes produces sub-queries that retrieve separately, no sends the original query through the normal path](../../assets/img/rag/query_routing_with_sub_query.png)

**The price:** the classifier runs on *every* request, and a misroute
is a brand-new failure mode — log its decisions.

## 6. The decision, on one page

Start with the test, not the brochure:

**THE TEST — is the correct passage in the retrieved top-k?**

- **No — and it isn't in the corpus at all.** Not a pattern problem:
  add sources, or route to web search (section 5).
- **No — but it is in the corpus.** Look at what the junk looks like:
  - exact IDs and codes missed → hybrid + reranker (section 3)
  - right content, wrong version or audience → metadata filter (section 3)
  - context-free fragments → retrieve small, read big (section 3)
  - compound or over-specific queries → sub-queries, step-back (section 2)
  - queries and documents sound nothing alike → HyDE, hypothetical questions (section 2)
- **Yes — the passage was there.** Look at where the answer goes wrong:
  - ignores evidence in the middle → reorder (section 4)
  - distracted by noise, or too expensive → compress (section 4)
  - easy questions overworked → routing (section 5)
  - confidently wrong when retrieval fails → reflection loop (section 5)

The whole clinic, card by card. On every card, read **Idea** as the
answer you would give in a system-design interview.

**hypothetical questions**
- **When:** users phrase questions casually and your documents are formal prose.
- **Idea:** pay the translation cost once, at indexing time — store the questions each chunk answers, so search matches question to question.
- **Example:** beside the drain-hose paragraph, store "why is water pooling under my dishwasher?" — the casual query now hits.
- **Cost:** one big LLM pass over the corpus, repeated on every re-index.
- **Skip when:** the corpus changes daily.

**HyDE**
- **When:** the same style gap, but the corpus changes often or traffic is light.
- **Idea:** have the LLM write a fake answer and search with its *shape*, not the question's words.
- **Example:** "leaking underneath" → a fake manual paragraph about the hose coupling → the real paragraph is retrieved.
- **Cost:** one extra LLM call on every query.
- **Skip when:** traffic is heavy on a stable corpus — enrich the documents once instead.

**sub-queries**
- **When:** one question actually contains several.
- **Idea:** the unit of retrieval is a single question — split the query until each part is one.
- **Example:** "compare the eco and intensive programs" → two spec lookups → the LLM compares.
- **Cost:** calls and retrievals multiply with the number of parts.
- **Skip when:** questions are already simple and single-topic.

**step-back**
- **When:** questions are more specific than anything written in the corpus.
- **Idea:** retrieve at the corpus's level of generality, answer at the user's level of detail.
- **Example:** "why does my 2019 X-500 beep twice?" → retrieve "what do the beep codes mean?" → answer the specific case.
- **Cost:** one extra call and one extra retrieval per query.
- **Skip when:** queries already match the corpus's altitude.

**small-to-big / sentence-window**
- **When:** matches are precise but arrive without their context.
- **Idea:** search with a small unit, read with a large one — the index card points, the page explains.
- **Example:** the child "Tighten it a quarter turn" matches → the LLM reads the 800-token repair section.
- **Cost:** index bookkeeping, larger prompts.
- **Skip when:** chunks already read as self-contained.

**hierarchical index**
- **When:** the right passage keeps coming from the wrong document.
- **Idea:** choose the right document first, and only then the passage inside it.
- **Example:** the summary index picks the X-500 manual → chunks are searched only inside it.
- **Cost:** one generated summary per document, at index time.
- **Skip when:** the corpus is small, or has no document structure.

**hybrid + RRF**
- **When:** exact IDs, codes, and names keep being missed.
- **Idea:** two searchers with opposite blind spots, merged by rank — a consistent second beats a lonely first.
- **Example:** "error E24" → the BM25 branch nails the literal entry → RRF puts it on top.
- **Cost:** a second index that must be kept in sync.
- **Skip when:** queries are purely conceptual, with no literal tokens.

**cross-encoder rerank**
- **When:** good candidates are retrieved, but the final ranking is mediocre.
- **Idea:** the cheap searcher finds candidates at scale; the expensive reader re-grades only the shortlist.
- **Example:** reading "error E24" against 20 candidates, the E24 entry is promoted above the overview chapter.
- **Cost:** roughly 100 ms of model inference per query.
- **Skip when:** the top-k is already reliably clean.

**metadata filtering**
- **When:** answers are semantically right but from the wrong version or audience.
- **Idea:** similarity cannot see the year — filter structured fields first, rank by meaning second.
- **Example:** `model_year = 2024` → the 2021 manual is excluded before the search even runs.
- **Cost:** nearly free at query time; the work is recording metadata at ingestion.
- **Skip when:** the corpus has one version and one audience.

**reordering**
- **When:** the model ignores evidence buried in the middle of the context.
- **Idea:** models read like people — the edges stick, the middle gets lost — so put the strongest chunks at the edges.
- **Example:** best chunk first, second-best last → the mid-context dip cannot hide them.
- **Cost:** nothing — the free lunch of this list.
- **Skip when:** you send only a few chunks anyway.

**context compression**
- **When:** prompts are long, noisy, and expensive.
- **Idea:** every irrelevant token costs money *and* attention — keep only the sentences that answer this query.
- **Example:** twenty chunks (~6,000 tokens) → only the ~900 about leaks and hoses survive.
- **Cost:** one extra call; it may drop a needed sentence — compress after reranking.
- **Skip when:** prompts are already short.

**routing**
- **When:** easy messages trigger the full pipeline, or the wrong tool gets used.
- **Idea:** not every question deserves retrieval — triage at the door.
- **Example:** "thanks, that fixed it!" → answered directly, no search at all.
- **Cost:** a classifier on every request; misroutes are a new failure mode.
- **Skip when:** every query genuinely needs retrieval.

**reflection loop**
- **When:** the bot answers confidently even when retrieval failed.
- **Idea:** check the sources before publishing — a slow correct answer beats a fast wrong one.
- **Example:** the chunks never mention beeping → rewrite the query and retrieve again.
- **Cost:** worst-case latency ×2–3.
- **Skip when:** speed matters more than the error floor.

And when two patterns look interchangeable, this is the tiebreaker —
why you pick one and *not* its look-alike:

- **Hypothetical questions or HyDE?** Same disease, opposite sides.
  Stable corpus and heavy traffic → pay once at write time
  (hypothetical questions). Fast-changing corpus and light traffic →
  pay per query (HyDE).
- **Sub-queries or step-back?** Count the questions. Several questions
  hiding in one ("compare A and B") → sub-queries. One question asked
  too specifically → step-back.
- **Parent-child, sentence-window, or hierarchical?** One idea, three
  zooms. Default → parent-child. Precision above all → sentence-window.
  The confusion is *across documents*, not within one → hierarchical.
- **Hybrid search or a better embedding model?** If the misses are
  exact tokens — IDs, error codes, names — no embedding upgrade cures
  literal blindness. Add the lexical searcher.
- **Cross-encoder rerank or a bigger top-k?** Raising k widens the
  prompt and feeds section 4's noise problem. The reranker adds
  precision without widening anything.
- **Metadata filter or the reranker?** The reranker judges *relevance*;
  only structured fields know *validity* — the year, the edition, the
  audience.
- **Compression or reordering?** Not rivals. Reordering is free —
  always do it. Compress when prompts are long or expensive.
- **Routing or a reflection loop?** Different ends of the pipe: routing
  saves money at the entrance, reflection saves quality at the exit —
  and they compose.

When several patterns still fit, climb the cost ladder from the bottom:

> **Free:** reordering; metadata filtering (if the metadata exists).
> **Paid once, at index time:** small-to-big, hierarchical, hybrid's second index, hypothetical questions.
> **Paid per query:** HyDE, compression, routing, the reranker.
> **Paid per query, multiplied:** sub-queries, reflection loops.

The binding rule: **measure first, buy the cheapest treatment that
fits the diagnosis, and stack the next pattern only when the hit rate
says the last one wasn't enough.**

## The whole story in six lines

1. Run the one test before touching anything: is the correct passage in
   the retrieved top-k?
2. If not, and the *question* is at fault: HyDE or hypothetical
   questions bridge the dialect gap; sub-queries split compound
   questions; step-back lifts over-specific ones.
3. If not, and the *index* is at fault: embed small and read big; add a
   lexical searcher with RRF and a cross-encoder judge; filter metadata
   before searching.
4. If the passage was found and the answer still failed: put the
   strongest chunks at the edges, and compress the noise away.
5. Give the pipe judgment: a router at the entrance, a reflection loop
   at the exit.
6. Climb the cost ladder from the free rung, and let the hit rate — not
   the brochure — tell you when to stop.

Back to the leaking dishwasher: the passage wasn't in the top-k, the
dialect gap is audible ("leaking underneath" vs "hose coupling
integrity"), the prescription is section 2 — hypothetical questions —
and the bill is one indexing pass. Four families, and you can name
them now.

## Glossary

The base vocabulary of the article, one line each:

- **token** — the smallest unit of text a model reads and writes; roughly three-quarters of an English word.
- **embedding** — a list of numbers placing a text's meaning as a point on a map; similar meanings land close together.
- **chunk** — a slice of a document, typically a few hundred tokens, stored and retrieved as one unit.
- **vector database** — the store that keeps embeddings and quickly finds the ones nearest to a query.
- **top-k** — the k nearest chunks a search returns; k is your choice.
- **prompt** — everything handed to the model in one request: the question, the retrieved chunks, the instructions.
- **corpus** — the whole document collection your system searches.
- **ingestion** — the write path: splitting documents, embedding them, and storing them with their metadata.
- **dense / sparse search** — dense compares embeddings (meaning); sparse — lexical, like BM25 — matches the exact words.
- **BM25** — the classic lexical ranking formula: rewards exact word matches, weighted by how rare each word is.
- **bi-encoder / cross-encoder** — a bi-encoder embeds query and document separately (fast, scales); a cross-encoder reads them together (accurate, slow).
- **reranking** — re-scoring a shortlist of retrieved candidates with a stronger model.
- **RRF (reciprocal rank fusion)** — merging several ranked lists by rank position instead of raw scores.
- **retrieval hit rate** — the share of failed questions whose correct passage *did* appear in the retrieved top-k; this article's one diagnostic.
- **hallucination** — a fluent answer shaped like the truth without being true; the failure RAG exists to prevent.

## Going deeper

- The Milvus team, [How to Enhance the Performance of Your RAG Pipeline](https://zilliz.com/learn/how-to-enhance-the-performance-of-your-rag-pipeline) — the survey this article's diagrams come from, organized by pipeline stage.
- Gao et al., [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (2022) — the HyDE paper.
- Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (2023) — the U-curve is theirs.
- Asai et al., [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) (2023).
- Yan et al., [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) (2024).
- Formal et al., [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) (2021).
- Cormack, Clarke & Büttcher, [Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods](https://dl.acm.org/doi/10.1145/1571941.1572114) (SIGIR 2009) — where k = 60 comes from.

---

*Diagram credit: all pipeline illustrations in this article are
reproduced from the Milvus team's guide [How to Enhance the Performance
of Your RAG Pipeline](https://zilliz.com/learn/how-to-enhance-the-performance-of-your-rag-pipeline)
and remain the work of their authors. The lost-in-the-middle U-curve is
this article's own drawing, redrawn approximately from Liu et al. 2023.*
