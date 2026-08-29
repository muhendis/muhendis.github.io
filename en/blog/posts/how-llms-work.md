Before you finish this sentence, your mind has already started guessing
the next ___. See? "Once upon a ___." You could not stop it: the blank
filled itself, without asking your permission.

That reflex has a name — next-token prediction — and this article makes
one claim: every sentence a language model has ever written to you, every
essay, every code snippet, every apology for a mistake it just made, is
that same reflex scaled up by a factor of billions. Your phone keyboard
plays the pocket version when it offers *tomorrow* after "see you". How
does a game this simple pass exams and write software? Because its demand
is merciless: to predict the next word of human text *well*, you must
absorb grammar, facts, style, and a working imitation of reasoning.
Everything below is a footnote to that idea.

**In this article**

- [1. Text becomes numbers](#1-text-becomes-numbers)
- [2. Numbers with meaning](#2-numbers-with-meaning)
- [3. The transformer: a context machine](#3-the-transformer-a-context-machine) — [Q, K, V](#q-k-v-the-mechanism-with-numbers) · [One direction only](#one-direction-only) · [Many heads](#many-heads)
- [4. Layers: where knowledge lives](#4-layers-where-knowledge-lives)
- [5. Training and scale](#5-training-and-scale)
- [6. From autocomplete to assistant](#6-from-autocomplete-to-assistant)
- [7. Generation: a loop, not a plan](#7-generation-a-loop-not-a-plan)
- [8. It does not remember you](#8-it-does-not-remember-you)
- [9. Why it makes things up](#9-why-it-makes-things-up)
- [The whole story in five lines](#the-whole-story-in-five-lines)
- [Going deeper](#going-deeper)

## 1. Text becomes numbers

A **tokenizer** (the standard algorithm is **byte-pair encoding**, BPE) chops text into pieces called **tokens** — "the" is one piece,
"unbelievable" may be "un + believ + able" — and gives each an ID number. Think LEGO: language is snapped apart into its most reusable bricks — common words stay whole, rare words are assembled from smaller pieces. Rule
of thumb: 100 tokens ≈ 75 English words, so a "128K context window" holds
roughly a novel. One consequence worth knowing: the model never sees letters,
only token IDs — which is why counting the r's in "strawberry" was famously
hard. It is like counting brushstrokes in a *photo* of a painting.

## 2. Numbers with meaning

Each token becomes an **embedding**: a long list of numbers that act as its
coordinates on a map of meaning. Think of the thousands of numbers as dials — one for formality, one for tense, one for sentiment, most for qualities no human ever named. *King* sits near *queen* and far from *spreadsheet*. Directions carry meaning too: the arrow from *Paris* to *France* runs parallel to the arrow from *Rome* to *Italy* — a "capital-of" direction. Even word arithmetic works: *king − man + woman* lands near *queen*. Nobody drew this map; it is learned. Because a bag of
coordinates has no order, each token's **position** is stamped in too: "dog
bites man" must stay different from "man bites dog".

## 3. The transformer: a context machine

An embedding alone cannot say what "bank" means — river bank, or the one
with the money? Meaning depends on neighbors. The **transformer** — the
design behind every modern model, the T in GPT — is built to read them.
Earlier architectures digested text left to right, squeezing everything seen
so far through one narrow running memory that faded with distance. The 2017 paper's radical move — and the literal meaning of its title
*Attention Is All You Need* — was to throw that machinery away and keep a
single mechanism, **attention**: let every token look *directly* at every
other token, all at once, and decide for itself what matters.

Watch that decision, in the original paper's own example:

> The animal didn't cross the street because **it** was too *tired*.
> The animal didn't cross the street because **it** was too *wide*.

One word changes and "it" switches sides. You resolved that instantly;
**attention** is how the model does. The whole trick reduces to one
sentence: **a word's context is a weighted blend of the other words, and attention's entire job is choosing the weights.** And the weights cannot come from *distance* — the word that settles "it" may sit twenty tokens back while the next-door word is noise. They must be computed from *content*, and learned.

The transformer's answer has a precise shape. Every layer of the stack
holds exactly two sub-layers. The first is **self-attention** — "self"
because the sentence attends to *itself*: each word looks at the other
words of the same text and rewrites its own vector in their light. The
second is a **feed-forward network**: a small neural network applied to
each position on its own, no looking around — first gather context
together, then digest it alone. This section unpacks the first sub-layer;
the second takes over in section 4.

### Q, K, V — the mechanism, with numbers

How does a token choose its weights? By playing three roles at once, each
one a small learned transformation of its own vector:

- **query** — what am I looking for?
- **key** — how should others find me?
- **value** — what do I hand over if picked?

YouTube runs on the same trio: the text you type is a query, every video's
title is a key, and the videos themselves are the values. In the model, all
three come from one place — the token's embedding — multiplied by three
learned tables of numbers, the weight matrices **W_Q, W_K, W_V** (each just
an ordinary dense layer). Same word, three outfits; and every token wears
all three at once, simultaneously a searcher, findable, and content to
hand over.

Why not skip the outfits and compare raw embeddings? Because an embedding
mixes every aspect of a word — grammar, meaning, position — and a good
search needs only one of them. The three tables extract *just the aspect
this job requires*: a query and key that match on "could be tired", not on
"starts with f". The value is a selection too — a chosen token hands over
not its whole embedding but the slice worth passing on, and W_V decides
which slice. What emerges behaves like a **soft dictionary**: where a real
dictionary matches a key exactly or returns nothing, attention matches
every key *partially* and takes a proportional slice of every value.

That is the cast. The arithmetic behind it is only two moves:

- **Dot product** — multiply two vectors position by position, then add:
  a · b = a₁b₁ + a₂b₂ + a₃b₃ + … For example,
  [2, 1, 0] · [3, 1, 4] = 6 + 1 + 0 = **7**. One number falls out — large
  when the vectors point the same way, small when they do not. A
  similarity meter.
- **Weighted sum** — mix several vectors by percentages, like a recipe:
  60% of this, 30% of that.

Time to run it. Trace "The quick brown fox" while the model works on
*fox* — four steps, the same four you will meet in the official formula at
the end.

**Step 1 — Score: who matters to me?** *Fox*'s query is dot-producted with
every word's key, its own included:

> scoreᵢ = Q(fox) · K(wordᵢ)

| pair | dot product | reading |
|---|---|---|
| Q(fox) · K(The) | 0.5 | barely relevant |
| Q(fox) · K(quick) | 2.1 | somewhat relevant |
| Q(fox) · K(brown) | 4.0 | very relevant |
| Q(fox) · K(fox) | 5.4 | itself — most of all |

**Step 2 — Scale: tame the numbers.** Before any percentages, every score
is divided by **√dₖ**, the dimension of the key vector:

> scaledᵢ = scoreᵢ ÷ √dₖ

With big vectors, dot products grow huge; huge scores would saturate the
softmax into all-or-nothing weights and stall learning. This division is
the "scaled" in *scaled dot-product attention*. (To keep our toy numbers
readable, read the table's scores as already scaled.)

**Step 3 — Percentages: turn scores into a recipe.** This is **softmax**,
and it is just two moves. First, raise *e* to each score — that makes every
number positive and stretches the gaps between them. Then divide each result
by the total — now they sum to exactly 100%:

> weightᵢ = e^scoreᵢ ÷ (e^score₁ + e^score₂ + … )

| pair | score | e^score | share of total |
|---|---|---|---|
| Q(fox) · K(The) | 0.5 | 1.6 | **1%** |
| Q(fox) · K(quick) | 2.1 | 8.2 | **3%** |
| Q(fox) · K(brown) | 4.0 | 54.6 | **19%** |
| Q(fox) · K(fox) | 5.4 | 221.4 | **77%** |
| | | total ≈ 285.8 | 100% |

Check the last row against the formula: 221.4 ÷ 285.8 ≈ 0.77 — the 77%. And notice what the exponential did: 5.4 is only a little above 4.0, yet 77% is
four times 19% — softmax rewards the leaders and starves the laggards. The
model has just decided, in numbers, how much attention each word deserves.
(Yes, a token attends to itself — usually most of all.)

**Step 4 — Blend: cook the recipe.** The new *fox* vector is the weighted
sum of the *values*:

> new vector = weight₁ × V(word₁) + weight₂ × V(word₂) + …
> fox_new = 0.01 × V(The) + 0.03 × V(quick) + 0.19 × V(brown) + 0.77 × V(fox)

The result is no longer the dictionary word *fox*; it is
*this-particular-quick-brown-fox*, and that enriched vector is what the
next layer receives.

Two remarks before that formula. First, the *only* learned parts in this
whole dance are the three tables behind Q, K, and V; the dot products, the
softmax, the weighted sum are fixed arithmetic, with no learning in them.
No rule ever told the model that foxes are brown — over trillions of
guesses, the tables were tuned until useful weights came out on their own.
Second, the price: every token scores every other token, so the work grows
with the *square* of the length, O(n²). Double the context, quadruple the
cost — million-token windows are an engineering feat, not a checkbox.

Now the payoff. Everything above compresses into a single line — the
formula printed in every paper since 2017:

> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**

Read left to right, it is the fox trace in symbols: Step 1 is QKᵀ, Step 2
the division by √dₖ, Step 3 the softmax, Step 4 the multiplication by V —
same four moves, same order, every time. The capital letters carry one
last gift: Q, K, and V here are matrices, every token's vectors stacked
into a block, so this one line runs the search for *all* tokens at once as
pure matrix multiplication. No loops — exactly the shape of work GPUs
devour.

### One direction only

A detail with large consequences: during generation, each token may only
look *backward*. *Fox* sees *brown*; *brown* never sees *fox*. That mask is
what makes the model a *next*-token predictor — and it is why a past
token's key and value, once computed, never change: nothing that arrives later can touch them. File that away; it becomes the KV cache in section 7.

This mask is also the switch behind the two famous model families. Built *with* the mask, a model reads left to right and therefore writes — the **decoder** family: GPT, and virtually every modern LLM. Built *without* it, a model sees both directions at once and classifies instead of writing — the **encoder** family: BERT. (The names are leftovers from the transformer's original translation design, which paired the two.) One architectural switch, the whole family tree.

### Many heads

One weighting per layer would be crude — a word needs grammar from one
neighbor and a referent from another. So each layer runs many attention
**heads** in parallel, each with its own Q/K/V lenses, each learning its
own relationship to track: one follows syntax, one resolves "it", one binds
adjectives to nouns. Nobody assigns these roles; they emerge, because each one helps predict
what comes next. Two concrete details complete the picture. Heads work in
slices: in the original design, each head projects the 512-number embedding
down to 64, so eight heads cost roughly the same as one full-width one. And
the paper's own visualizations show the division of labor — encoding "it",
one head locks onto *the animal* while another locks onto *tired*: the
referent and the reason, tracked at the same time.

All of section 3, on one card:

| question | answer |
|---|---|
| What is self-attention? | The sentence attending to itself: every token rewrites its vector in the light of the others |
| What are Q, K, V? | Three roles per token — query: *what am I looking for?* · key: *how am I found?* · value: *what do I hand over?* |
| How are they made? | embedding × W_Q, W_K, W_V — three learned linear layers; same word, three outfits |
| Why three separate vectors? | An embedding mixes every aspect of a word; each role extracts only the aspect its job needs |
| Why are the weights learned? | The right weights cannot come from word distance; they are computed from content |
| In what order does the math run? | score (Q·K) → scale (÷√dₖ) → softmax → blend (×V) |
| What is the feed-forward network? | The small network that digests each token alone after context is gathered — the knowledge warehouse of section 4 |
| Where, when, who? | Every layer, every head, all tokens at once — Vaswani et al., 2017 |

## 4. Layers: where knowledge lives

One attention sub-layer plus one feed-forward sub-layer: that pair is a
**layer**, and a transformer is this layer stacked dozens to a
hundred-plus times. Why stack? Because a single pass gives each word only
one round of context-gathering. Stacked passes let meaning build: after
the early layers, *fox* has become *brown-quick-fox*; after the deep
ones, *the subject about to act*.

The stacking follows one rule: a layer never *replaces* the vectors — it
*edits* them. This is the **residual connection**: each layer's output is
*added* on top of its input, like notes in a book's margin. Nothing an
earlier layer built gets erased, so refinements accumulate.

Inside every layer, the two sub-layers split the work. **Attention**
moves information *between* words — the librarian, fetching each word the
context it needs. The **feed-forward network** digests each word *alone*,
no looking around — the warehouse, where training left its patterns:
"Paris pairs with France". Roughly two-thirds of a model's
**parameters** — its learned numbers — sit in these warehouses. That
answers this section's title: an LLM's knowledge lives in no sentence you
can point to; it is smeared across billions of feed-forward weights.

Nobody assigns the layers their jobs; a division of labor emerges from
training on its own. Early layers handle spelling and grammar; deep
layers, facts and long-range logic — like a photograph developing in the
tray: outlines first, faces later.

Growth, then, mostly means more warehouse. GPT-2 made news in 2019 with
1.5 billion parameters; frontier models now run to the trillions. A
modern twist, **mixture of experts (MoE)**, builds many warehouses per
layer and trains a small router to send each token to the best one or
two of them: huge total capacity, of which each token pays for only a
fraction.

One question completes the machine: how does a stack of vector-editors
produce a *prediction*? At the very top, the model takes the final
vector of the **last** token — after all those layers, it encodes not
one word but the whole context that led here. That single vector is then
scored against every token the model knows, with the same move attention
uses — a dot product. In many models the scoring even reuses the
embedding table of section 2, run in reverse:

> score(candidate) = final vector · candidate's embedding — one score per token in the vocabulary (~50,000 of them in GPT-2)

**Softmax** — the same percentage converter as attention's Step 3 —
turns those scores into probabilities that sum to 100%. After "Once upon
a", the mass piles onto "time". After "My favorite city is", it spreads
across hundreds of cities. Both are correct answers to the only question
the model ever answers: *what likely comes next?*

## 5. Training and scale

**Pretraining**: show the model trillions of tokens, hide the next one, let
it guess. A **loss function** scores its surprise at the truth:

> loss = −log p(correct token)

Had the model given the true next token a 90% chance, the loss is
−log 0.9 ≈ 0.1 — barely surprised. Had it given 20%, the loss is
−log 0.2 ≈ 1.6 — badly surprised. **Gradient descent** then nudges every
parameter a tiny step in the direction that shrinks this number; repeat trillions of times. Picture it as descending a mountain in fog: you cannot see the valley, you only feel the slope under your feet — so you take one small step downhill, and then a trillion more. (The standard report card: **perplexity = e^(average loss)**. An average loss of 1.6 gives e^1.6 ≈ 5 — as
unsure as picking among five equally likely words. Central to pretraining; a
weak proxy for real tasks.)

Nobody writes rules in — predicting a detective novel's last chapter
requires tracking who had a motive, so the tracking gets learned. The result
is the training data compressed like a JPEG: the picture survives, the
pixels do not.

Scale follows **scaling laws**: loss falls as a power law in compute —
roughly loss ≈ a · C^(−α), a straight line on log-log paper — so multiplying
compute by ten buys a predictable drop. That is what turned nine-figure
training runs from gambles into plans: GPT-4's final loss was predicted from
trials 10,000× smaller. DeepMind's **Chinchilla** added the balance rule:
parameters and data must grow together (roughly 20 tokens per parameter);
their 70B model beat a 280B rival on that arithmetic alone.

One honest caveat: the curve is smooth, but skills can arrive abruptly — a
model may fail three-digit arithmetic at size after size, then handle it
reliably at the next jump: an **emergent ability**. And the raw material is
finite: high-quality public text is nearly exhausted, which is pushing the
frontier toward synthetic data and toward spending compute at answer time
instead — the reasoning models below.

## 6. From autocomplete to assistant

Pretraining yields a **base model**: a machine that continues text, nothing
more. Ask it "What is the capital of France?" and you may get "Paris." — or
nine more quiz questions, since quizzes travel in packs online — or a
fictional scene: "asked the teacher, and nobody raised a hand." All are
faithful continuations. Coaxing answers out once required writing "Q: … A:"
so that an answer became the likeliest continuation; prompt engineering was
born there.

Two cheap stages make it an assistant. **Instruction tuning**: train further
on tens of thousands of written examples of question → ideal answer, until
"answer helpfully" becomes the most probable continuation. **RLHF**
(reinforcement learning from human feedback): humans compare candidate
answers, a reward model learns their taste, and the LLM is tuned toward it —
comparing is far easier for people than authoring perfection, and
comparisons capture what examples cannot spell out: tone, honesty about
uncertainty, refusing harm.

Both stages cost a small fraction of pretraining's months on thousands of
GPUs — and when even that is too much, **LoRA** freezes the model and trains
tiny low-rank adapter matrices alongside it: near fine-tuning quality for a
sliver of the parameters, with adapters you can swap like lenses. The
punchline: GPT-3 existed for more than two years before ChatGPT. The revolution was
these stages, not a bigger network.

## 7. Generation: a loop, not a plan

The model computes probabilities, **samples** one token (a weighted draw),
appends it, and repeats — each new token instantly part of the next
prediction's input — until a special stop token says "I'm done."

Three dials govern the draw. After "The sky was", the list might read
*blue* 60%, *dark* 10%, …, *potato* 0.0001%:

- **Temperature** divides every score by T before softmax — the draw uses
  softmax(score ÷ T). Below 1, the
  gaps stretch and the leader takes almost everything — T = 0 is greedy
  decoding, nearly deterministic (batching and floating-point order still
  leave tiny drift). Above 1, the gaps shrink and *dark* and *grey* get to
  compete. Low for SQL, high for brainstorming.
- **Top-k** keeps only the k most likely tokens and deletes the tail —
  *potato* included.
- **Top-p** keeps the smallest set covering, say, 90% of the probability —
  two tokens when the model is sure, eighty when it is torn.

Cut first, then draw: that is why answers vary day to day, and why the sky
is never a potato.

The loop explains why "think step by step" works: the page is the model's
only scratchpad. Asked for 17 × 24 in one leap, it must land the answer in
a single guess; allowed to write "17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 =
408", every intermediate step joins the context and sharpens the next
prediction. Reasoning models industrialize exactly this.

A contrast completes the picture. In *training*, the model sees whole
documents and processes every token in parallel — the parallelism that let
transformers soak up GPUs and scale. In *inference* — chatting — text
arrives one token at a time, and the **KV cache** exists to make that
serial loop cheap, cashing in the asymmetry from section 3. Token number
1,000 must compare its query against 999 earlier keys, which looks like
re-reading everything at every step. It is not: past keys and values never
change, so they are computed once and stored.

You have felt this cache. The pause before a long prompt's first word is
**prefill**, building it; afterwards words stream quickly because each pays
only for itself. It is also why long chats eat memory — the cache grows
with every token, in every layer. The arithmetic is sobering:

> cache = 2 (K and V) × layers × context length × vector width × bytes

For a 32-layer, 4,096-wide model in 16-bit precision holding 100K tokens:
2 × 32 × 100,000 × 4,096 × 2 bytes ≈ **52 GB** — for one conversation. That
is also why "cached input" is priced cheaper: it is already paid for. The
other big serving lever is **quantization** — store the weights in fewer
bits (16 → 8 → 4); inference is bound by moving bytes, not arithmetic, so
smaller weights mean faster, cheaper answers at a modest accuracy cost.

Here is the whole machine in one tiny trace. Input: **"I love"** — the model
must produce the next token.

1. **Cache check.** "I" and "love" were already processed; their keys and
   values (K₁V₁, K₂V₂) sit in the KV cache.
2. **Fresh query.** For the new position the model computes a query, Q₃ — in
   effect the question "given everything so far, what should come next?"
3. **Match, scale, softmax.** Q₃ is dot-producted with the cached keys,
   divided by √dₖ, and softmaxed — Steps 1–3 of the fox trace — landing at:

   | comparison | attention weight |
   |---|---|
   | Q₃ · K₁ ("I") | 30% |
   | Q₃ · K₂ ("love") | 70% |

4. **Blend.** Step 4 as before — the weighted sum of the cached values:
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
entry to fail to find — it produces something *shaped* like an answer, because plausible, not true, is what it optimizes. This is not lying — lying requires knowing the truth. It is completing the sentence. The fixes change the input, and they form a decision ladder: **prompting** shapes behavior in context; **RAG** (retrieval) supplies knowledge that changes often, no weights touched; **fine-tuning** bakes in style or domain that must be permanent. Reach for them in that order — each step up costs more. Add tools, and check sources yourself: the step the lawyers skipped.

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
student, or the curious voice in your own head — start where the model starts: with the next token.

One last thing. In this article's first line, your mind wrote "time" into
the blank — instantly, confidently, from pure pattern. Now you know
exactly how a machine does the same. That is the whole story.

## Going deeper

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — the original transformer paper; the tired/wide example is theirs.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the classic visual walkthrough.
- Ebrahim Pichka, [What are Query, Key, and Value in the Transformer Architecture?](https://medium.com/data-science/what-are-query-key-and-value-in-the-transformer-architecture-and-why-are-they-used-acbe73f731f2) — a careful unpacking of the QKV intuition, including the soft-dictionary view.
- Andrej Karpathy, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the whole machine, written in code before your eyes.
