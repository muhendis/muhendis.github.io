Every answer a language model has ever given you was produced the same way:
by predicting the next small piece of text, over and over. Your phone keyboard
does a tiny version of this — type "see you" and it offers *tomorrow*. An LLM
is that trick scaled up by billions, and the interesting part is what the game
demands: to predict the next word in human text well, a model must absorb
grammar, facts, style, and a working imitation of reasoning. Everything below
is a footnote to that idea.

## 1. Text becomes numbers

A **tokenizer** (the standard algorithm is **byte-pair encoding**, BPE) chops text into pieces called **tokens** — "the" is one piece,
"unbelievable" may be "un + believ + able" — and gives each an ID number. Rule
of thumb: 100 tokens ≈ 75 English words, so a "128K context window" holds
roughly a novel. One consequence worth knowing: the model never sees letters,
only token IDs — which is why counting the r's in "strawberry" was famously
hard. It is like counting brushstrokes in a *photo* of a painting.

## 2. Numbers with meaning

Each token becomes an **embedding**: a long list of numbers that act as its
coordinates on a map of meaning. *King* sits near *queen* and far from
*spreadsheet*, and directions encode relationships: the arrow from *Paris* to *France* runs parallel to the arrow from *Rome* to *Italy* — a "capital-of" direction — and *king − man + woman* lands near *queen*. Nobody drew this map; it is learned. Because a bag of
coordinates has no order, each token's **position** is stamped in too: "dog
bites man" must stay different from "man bites dog".

## 3. The transformer: a context machine

An embedding alone cannot say what "bank" means — river bank, or the one
with the money? Meaning depends on neighbors. The **transformer** — the
design behind every modern model, the T in GPT — is built to read them.
Earlier architectures digested text left to right, squeezing everything seen
so far through one narrow running memory that faded with distance. The
transformer's move: let every token look *directly* at every other token,
all at once, and decide for itself what matters.

Watch that decision, in the original paper's own example:

> The animal didn't cross the street because **it** was too *tired*.
> The animal didn't cross the street because **it** was too *wide*.

One word changes and "it" switches sides. You resolved that instantly;
**attention** is how the model does. The whole trick reduces to one
sentence: **a word's context is a weighted blend of the other words, and
attention's entire job is choosing the weights.**

### Q, K, V — the mechanism, with numbers

To choose the weights, every token is given three roles, each a small
learned transformation of its vector:

- **query** — what am I looking for?
- **key** — how should others find me?
- **value** — what do I hand over if picked?

(Think YouTube: your search is the query, video titles are the keys, the
videos are the values.) And only two math operations are involved: the
**dot product** — multiply two vectors, get one similarity score — and the
**weighted sum** — blend vectors by percentages.

Take "The quick brown fox", with the model working on *fox*:

1. **Score.** *fox*'s query is dot-producted with every key, its own
   included: Q·K(The) = 0.5, Q·K(quick) = 2.1, Q·K(brown) = 4.0,
   Q·K(fox) = 5.4.
2. **Percentages.** Softmax converts the scores into weights: 2%, 8%, 30%,
   60%. The model has just *decided, in numbers*, that *brown* matters and
   *The* barely does. Note that a token attends to itself — often most of
   all.
3. **Blend.** fox_new = 0.02 × V(The) + 0.08 × V(quick) + 0.30 × V(brown)
   + 0.60 × V(fox). The result is no longer generic *fox*; it is
   *this-quick-brown-fox*, and it is what the next layer receives.

No rule ever told the model that foxes are brown. The Q, K, V
transformations were tuned, over trillions of guesses, until useful weights
came out on their own.

Two footnotes on this machinery. First, the raw scores are divided by
√(key dimension) before softmax — hence "*scaled* dot-product attention":
huge dot products would saturate the weights into all-or-nothing and stall
learning. Second, since every token scores every other token, the work grows
with the *square* of the length — O(n²). Double the context, quadruple the
cost; million-token windows are an engineering feat, not a checkbox.

### One direction only

A detail with large consequences: during generation, each token may only
look *backward*. *Fox* sees *brown*; *brown* never sees *fox*. That mask is
what makes the model a *next*-token predictor — and it is why a past
token's key and value, once computed, never change: nothing that arrives later can touch them. File that away; it becomes the KV cache in section 7. The mask is also the field's dividing line: models built with it (GPT-style **decoders**) generate; models built without it (BERT-style **encoders**) read in both directions and classify rather than generate. Modern LLMs are almost all decoder-only.

### Many heads

One weighting per layer would be crude — a word needs grammar from one
neighbor and a referent from another. So each layer runs many attention
**heads** in parallel, each with its own Q/K/V lenses, each learning its
own relationship to track: one follows syntax, one resolves "it", one binds
adjectives to nouns. Nobody assigns these roles; they emerge, because each
one helps predict what comes next.

## 4. Layers: where knowledge lives

A transformer is this block stacked dozens to a hundred-plus times — and
each layer *edits* the vectors rather than replacing them, so meaning
accumulates: *fox* becomes *brown-quick-fox*, then *subject about to act*,
layer by layer. Inside every layer, attention gathers context (the
librarian) while a **feed-forward network** — a small network applied to
each token on its own — stores learned patterns like "Paris pairs with
France" (the warehouse; most **parameters** live here). A modern twist, **mixture of experts (MoE)**: build many warehouses and let a router send each token to its best one or two — huge total capacity, only a fraction of it paying compute per token. Early layers handle
spelling and grammar; deeper layers, facts and long-range logic. GPT-2 made
news in 2019 with 1.5 billion parameters; frontier models now run to the
trillions.

At the very top, **softmax** — the same percentage converter attention
uses — turns the final scores into a probability for every token the model
knows. After "Once upon a", the mass piles onto "time". After "My favorite
city is", it spreads across hundreds of cities. Both correctly answer the
only question the model ever answers: *what likely comes next?*

## 5. Training and scale

**Pretraining**: show the model trillions of tokens, hide the next one, let
it guess. A **loss function** scores its surprise; **gradient descent**
nudges every parameter a tiny step toward less surprise; repeat trillions of times. (The standard report card is **perplexity** — the exponential of the average surprise on held-out text. Central to pretraining; a weak proxy for real tasks.) Nobody writes rules in — predicting a detective novel's last chapter
requires tracking who had a motive, so the tracking gets learned. The result
is the training data compressed like a JPEG: the picture survives, the
pixels do not.

Scale follows **scaling laws**: multiply compute by ten and the loss falls a
predictable amount, which is what turned nine-figure training runs from
gambles into plans — GPT-4's final loss was predicted from trials 10,000×
smaller. DeepMind's **Chinchilla** added the balance rule: parameters and
data must grow together (roughly 20 tokens per parameter); their 70B model
beat a 280B rival on that arithmetic alone. One honest caveat: the curve is smooth, but skills can arrive abruptly — a model may fail three-digit arithmetic at size after size, then handle it reliably at the next jump: an **emergent ability**. And the raw material is finite: high-quality public text is nearly exhausted, which is pushing the frontier toward synthetic data and toward spending compute at answer time instead — the reasoning models below.

## 6. From autocomplete to assistant

Pretraining yields a **base model**: a machine that continues text, nothing
more. Ask it "What is the capital of France?" and you may get "Paris." — or
nine more quiz questions, since quizzes travel in packs online — or a fictional scene: "asked the teacher, and nobody raised a hand." All are faithful continuations. Coaxing answers out once required writing "Q: … A:" so that an answer became the likeliest continuation; prompt engineering was born there. Two cheap
stages make it an assistant. **Instruction tuning**: train further on tens of
thousands of written examples of question → ideal answer, until "answer
helpfully" becomes the most probable continuation. **RLHF** (reinforcement
learning from human feedback): humans compare candidate answers, a reward
model learns their taste, and the LLM is tuned toward it — comparing is far easier for people than authoring perfection, and comparisons capture what examples cannot spell out: tone, honesty about uncertainty, refusing harm. Both stages cost a small fraction of pretraining's months on thousands of GPUs — and when even that is too much, **LoRA** freezes the model and trains tiny low-rank adapter matrices alongside it: near fine-tuning quality for a sliver of the parameters, with adapters you can swap like lenses. The punchline: GPT-3 existed
for two years before ChatGPT. The revolution was these stages, not a bigger
network.

## 7. Generation: a loop, not a plan

The model computes probabilities, **samples** one token (a weighted draw), appends it, and repeats — each new token instantly part of the next prediction's input — until a special stop token says "I'm done." Three dials govern the draw. After "The sky was":
*blue* 60%, *dark* 10%, … *potato* 0.0001%. **Temperature** sharpens or flattens the list — low for SQL, high for brainstorming; temperature 0 is greedy decoding, nearly deterministic, though batching and floating-point order still leave tiny run-to-run drift; **top-k** keeps only
the k most likely; **top-p** keeps the smallest set covering, say, 90% —
adapting between two tokens when the model is sure and eighty when it is
torn. Cut first, then draw: that is why answers vary day to day and the sky
is never a potato.

Two consequences of the loop: "think step by step" works because the page is
the model's only scratchpad — writing "17 × 24 = 340 + 68" makes each next
prediction easier, which reasoning models industrialize. And a contrast sharpens the last piece. In *training*, the model sees whole
documents and processes every token in parallel — that parallelism is what
let transformers, unlike their predecessors, soak up GPUs and scale. In
*inference* — chatting — text arrives one token at a time, and the **KV
cache** exists to make that serial loop cheap, cashing in the asymmetry from
section 3. Token number 1,000 must compare its query against 999 earlier keys — which looks like re-reading everything at every step. It is not: past keys and values never change, so they are computed once and stored. The pause before a long prompt's first word is **prefill**, building that cache; afterwards words stream quickly because each pays only for itself; long chats eat memory because the cache grows with every token; and "cached input" is cheaper because it is already paid for. The other big serving lever is **quantization**: store the weights in fewer bits (16 → 8 → 4). Inference is bound by moving bytes, not by arithmetic, so smaller weights mean faster, cheaper answers at a modest accuracy cost.

Here is the whole machine in one tiny trace. Input: **"I love"** — the model
must produce the next token.

1. **Cache check.** "I" and "love" were already processed; their keys and
   values (K₁V₁, K₂V₂) sit in the KV cache.
2. **Fresh query.** For the new position the model computes a query, Q₃ — in
   effect the question "given everything so far, what should come next?"
3. **Match.** Q₃ is compared against the cached keys:

   | comparison | attention weight |
   |---|---|
   | Q₃ · K₁ ("I") | 30% |
   | Q₃ · K₂ ("love") | 70% |

4. **Blend.** The output is built from the cached values:
   0.30 × V₁ + 0.70 × V₂ — a vector representing *this context*.
5. **Predict.** That vector runs through the final layers and softmax:
   *you* 85%, *it* 7%, *her* 5%, … The draw picks **"you"**.
6. **Extend the cache.** K₃ and V₃ are computed for "you" and appended, and
   the loop restarts with "I love you" as the context.

Q is always computed fresh; K and V are always fetched from the cache. That
one sentence is the entire KV-cache story.

## 8. It does not remember you

After training, parameters are **frozen**. Every message replays the whole
conversation through the network — like a brilliant consultant with no
long-term memory who needs the full case file every morning. What feels like memory is the context window — which is why very long chats slow down and lose track of their beginnings. The flip side is **in-context learning**: show
"sea → mer, house → maison, cat → ?" and the model answers *chat*, having learned the task from the prompt alone, no parameter changed. Most practical prompt engineering is exactly this: arranging the context so the desired continuation becomes the most probable one.

## 9. Why it makes things up

In 2023, lawyers in *Mata v. Avianca* filed six precedents invented by
ChatGPT — which, asked if they were real, said yes. A $5,000 fine made "AI
hallucination" famous. The mechanics are no mystery: the model is a
probability engine, not a database. Where training data is rich, the most probable continuation is usually the truth. Where it is thin, there is no
entry to fail to find — it produces something *shaped* like an answer,
because plausible, not true, is what it optimizes. The fixes change the input, and they form a decision ladder: **prompting** shapes behavior in context; **RAG** (retrieval) supplies knowledge that changes often, no weights touched; **fine-tuning** bakes in style or domain that must be permanent. Reach for them in that order — each step up costs more. Add tools, and check sources yourself: the step the lawyers skipped.

## The whole story in five lines

1. Text → **tokens** → **embeddings** (coordinates of meaning, position included).
2. **Attention** (query·key·value) blends each token's vector with its
   context — looking only backward; feed-forward layers store the knowledge.
3. **Pretraining** = next-token prediction at scale; scaling laws make the
   gains predictable; instruction tuning + RLHF turn the base model into an
   assistant.
4. Generation = sample, append, repeat — temperature, top-k, top-p tune the
   draw; the KV cache makes it affordable.
5. Frozen weights; memory is the context window; fluent because it optimizes
   *plausible* — hallucinating for the same reason.

And the next time someone asks how these models work — an interviewer, a
student, or the curious voice in your own head — start where the model
starts: with the next token.
