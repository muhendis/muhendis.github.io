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

A **tokenizer** chops text into pieces called **tokens** and gives each
an ID number — "the" is one piece, "unbelievable" may be "un + believ +
able". Think LEGO: language snapped apart into its most reusable bricks,
common words whole, rare ones assembled from parts (the standard
algorithm is **byte-pair encoding**, BPE). Rule of thumb: 100 tokens ≈
75 English words, so a "128K context window" holds roughly a novel. And
because the model sees only token IDs, never letters, counting the r's
in "strawberry" was famously hard — like counting brushstrokes in a
*photo* of a painting.

## 2. Numbers with meaning

Each token then becomes an **embedding**: a long list of numbers, its
coordinates on a map of meaning. Think of them as thousands of dials —
one for formality, one for tense, most for qualities no human ever
named. On this map *king* sits near *queen*; the arrow from *Paris* to
*France* runs parallel to the arrow from *Rome* to *Italy*; *king − man
+ woman* lands near *queen*. Nobody drew the map — it is learned. One
last stamp goes in, each token's **position**, because "dog bites man"
must stay different from "man bites dog".

## 3. The transformer: a context machine

An embedding alone cannot say what "bank" means — river bank, or the
one with the money? Meaning lives in context, and the **transformer** —
the T in GPT — is a machine for reading it. Older designs digested text
left to right through one narrow memory that faded with distance. The
2017 paper *Attention Is All You Need* threw that away and kept a
single mechanism, **attention**: every token looks *directly* at every
other token, all at once, and decides for itself what matters.

Watch that decision in the paper's own example:

> The animal didn't cross the street because **it** was too *tired*.
> The animal didn't cross the street because **it** was too *wide*.

One word changes and "it" switches sides. You resolved that instantly;
attention is how the model does. The whole trick is one sentence: **a
word's new meaning is a weighted blend of the other words — and
attention's entire job is choosing the weights.** The right weights
cannot come from distance (the deciding word may sit twenty tokens
back); they are computed from *content*, and learned.

The machinery comes in stacked layers, each holding two sub-layers:
**self-attention**, where every word rewrites its vector in the light
of the others, then a **feed-forward network**, a small network that
digests each word alone. Gather together, digest alone. This section
opens the first; section 4 takes the second.

### Q, K, V — the mechanism, with numbers

To choose its weights, every token plays three roles at once — three
small learned makeovers of its own embedding:

- **query** — what am I looking for?
- **key** — how should others find me?
- **value** — what do I hand over if picked?

YouTube runs on the same trio: your search text is a query, every
video's title is a key, the videos themselves are the values. In the model, all three come from the token's embedding multiplied by three learned tables, **W_Q, W_K, W_V** — same word, three outfits, all worn at once. Raw embeddings would not do: they mix everything about a word, while a search needs one aspect at a time — a query and key that match on "could be tired", not "starts with f". The result is a **soft dictionary**: every key matches *partially*, and a slice of every value is taken.

The arithmetic needs two moves only: the **dot product** (multiply two
vectors position by position and add — the more aligned, the bigger the
number: a similarity meter) and the **weighted sum** (mix vectors by
percentages, like a recipe). Four steps run them. Trace "The quick
brown fox" while the model works on *fox*:

**Step 1 — Score.** *Fox*'s query meets every word's key:
scoreᵢ = Q(fox) · K(wordᵢ).

**Step 2 — Scale.** Every score is divided by **√dₖ**, the key vector's
dimension, keeping big vectors from jamming softmax into all-or-nothing
weights: scaledᵢ = scoreᵢ ÷ √dₖ.

**Step 3 — Softmax.** Raise *e* to each score, divide by the total, and
the scores become percentages: weightᵢ = e^scoreᵢ ÷ (e^score₁ + …).

| pair | score | e^score | share |
|---|---|---|---|
| Q(fox) · K(The) | 0.5 | 1.6 | **1%** |
| Q(fox) · K(quick) | 2.1 | 8.2 | **3%** |
| Q(fox) · K(brown) | 4.0 | 54.6 | **19%** |
| Q(fox) · K(fox) | 5.4 | 221.4 | **77%** |

Check the last row: 221.4 ÷ 285.8 ≈ 77%. And note the exponential at
work — 5.4 is only a little above 4.0, yet 77% is four times 19%:
softmax rewards leaders.

**Step 4 — Blend.** The new *fox* is the weighted sum of the values:
fox_new = 0.01·V(The) + 0.03·V(quick) + 0.19·V(brown) + 0.77·V(fox) —
no longer the dictionary word *fox*, but
*this-particular-quick-brown-fox*.

Everything above is one famous line:

> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**

Step 1 is QKᵀ, Step 2 the division, Step 3 the softmax, Step 4 the multiplication by V. The *only* learned parts are the three tables; the rest is fixed arithmetic — and with all tokens stacked as matrices, this one line runs every search at once: exactly the work GPUs devour. The price is **O(n²)** — everyone scores everyone, so double the context, quadruple the cost.

### One direction only

During generation, a token may only look *backward*: *fox* sees
*brown*; *brown* never sees *fox*. That mask is what makes the model a
*next*-token predictor — and it means a past token's key and value,
once computed, never change. File that away; it becomes the KV cache in
section 7. The mask is also the switch behind the model families: built
*with* it, a model writes — the **decoder** family, GPT and virtually
every modern LLM; built *without* it, a model reads both ways and
classifies — the **encoder** family, BERT.

### Many heads

One weighting per layer would be crude — a word needs grammar from one
neighbor and a referent from another. So each layer runs many attention
**heads** in parallel, each with its own Q/K/V lenses, each on a thin
slice of the vector (512 ÷ 8 heads = 64 in the original design, so
eight heads cost about one). Encoding "it", one head locks onto *the
animal* while another locks onto *tired* — referent and reason, tracked
at the same time.

All of section 3, on one card:

| question | answer |
|---|---|
| What is self-attention? | The sentence attending to itself: every token rewrites its vector in the light of the others |
| What are Q, K, V? | Three roles per token — query: *what am I looking for?* · key: *how am I found?* · value: *what do I hand over?* |
| Why three separate vectors? | An embedding mixes every aspect of a word; each role extracts only what its job needs |
| In what order does the math run? | score (Q·K) → scale (÷√dₖ) → softmax → blend (×V) |
| What is the feed-forward network? | The network that digests each token alone — the knowledge warehouse of section 4 |

## 4. Layers: where knowledge lives

One attention sub-layer plus one feed-forward sub-layer make a
**layer**; a transformer is a tower of them, dozens to a hundred-plus
floors. On every floor, attention — the librarian — gathers context,
and the feed-forward network — the warehouse — digests it alone,
holding learned patterns like "Paris pairs with France". Each floor's
output is *added* to its input (the **residual connection**), so
nothing is erased: *fox* grows, floor by floor, into *brown-quick-fox*,
then *the subject about to act*. Lower floors handle spelling and
grammar; upper floors, facts and logic. And knowledge lives mostly in
the warehouses — roughly two-thirds of all **parameters** — smeared
across billions of weights, never stored as sentences. Growing a model
mostly means growing warehouse: GPT-2's famous 1.5 billion parameters
(2019) became today's trillions, and **mixture of experts (MoE)** puts
many warehouses on each floor, routing every token to the best one or
two.

At the roof, a *prediction* is due. The last token's final vector — by
now encoding the whole context — is dot-producted against every token
the model knows (~50,000 in GPT-2):

> score(candidate) = final vector · candidate's embedding

**Softmax** turns the scores into probabilities. After "Once upon a",
the mass piles onto "time"; after "My favorite city is", it spreads
across hundreds of cities. Either way the model answers its only
question: *what likely comes next?*

## 5. Training and scale

Every number in the machine starts as random noise. **Pretraining**
sets them, like a school with no lessons, only exams: show the model
trillions of tokens of real text, hide the next one, demand a guess.
The answer key is free — it is simply the token that actually came
next; the data grades itself. Each guess is scored by the **loss**:

> loss = −log p(correct token)

A 90% chance on the truth costs −log 0.9 ≈ 0.1 — barely surprised; 20%
costs −log 0.2 ≈ 1.6 — badly surprised. **Gradient descent** then
nudges every parameter a tiny step downhill — a descent in fog, feeling
only the slope underfoot — trillions of times over. (Report card:
**perplexity** = e^(average loss); e^1.6 ≈ 5, like choosing among five
words.) Skills nobody programmed appear because they lower the loss —
predicting a detective novel's ending requires tracking motives, so
tracking is learned — until the model is the training data compressed
like a JPEG: picture kept, pixels gone.

Scale pays predictably. **Scaling laws** — loss ≈ a · C^(−α), a
straight line on log-log paper — let you read a giant model's quality
off cheap trial runs: OpenAI predicted GPT-4's final loss from trials
10,000× smaller, and DeepMind's **Chinchilla** showed parameters and
data must grow together, ~20 tokens per parameter — its 70B beat a
280B Gopher. Two caveats: skills can still jump out abruptly
(**emergent abilities**), and high-quality public text is running
out — pushing compute toward answer time instead: the reasoning models
of section 7.

## 6. From autocomplete to assistant

Pretraining yields a **base model**: a machine that continues text,
nothing more. Ask it "What is the capital of France?" and you may get
"Paris." — or nine more quiz questions — or "asked the teacher, and
nobody raised a hand." All faithful continuations; coaxing an answer
once meant writing "Q: … A:" yourself, and prompt engineering was born
there. Two cheap stages make an assistant. **Instruction tuning**:
further training on tens of thousands of question → ideal-answer pairs,
until answering helpfully is the likeliest continuation. **RLHF**: humans compare candidate answers, a reward model learns their taste, and the LLM is tuned toward it — capturing what examples cannot spell out: tone, honesty, refusal. (Cheaper still,
**LoRA** freezes the model and trains tiny adapter matrices alongside —
near fine-tuning quality for a sliver of the parameters.) The
punchline: GPT-3 existed for more than two years before ChatGPT. The
revolution was these stages, not a bigger network.

## 7. Generation: a loop, not a plan

The model computes probabilities, **samples** one token, appends it,
and repeats until a special stop token — each new token instantly part
of the next prediction's input. Three dials govern the draw. After "The
sky was": *blue* 60%, *dark* 10%, …, *potato* 0.0001%.

- **Temperature** divides every score by T before softmax — T = 0 is greedy and near-deterministic, high T lets *dark* and *grey* compete. Low for SQL, high for brainstorming.
- **Top-k** keeps the k likeliest tokens — *potato* deleted.
- **Top-p** keeps the smallest set covering, say, 90% of the probability — two tokens when sure, eighty when torn.

Cut first, then draw: that is why answers vary day to day, and why the
sky is never a potato.

The loop also explains "think step by step" — the page is the model's
only scratchpad. Asked for 17 × 24 in one leap, it must land the answer
in a single guess; allowed to write 17 × 24 = 340 + 68 = 408, every
step joins the context and sharpens the next prediction. Reasoning
models industrialize exactly this.

One economic fact completes the picture. Training processes whole
documents in parallel; chat produces tokens one at a time — and the
**KV cache** keeps that serial loop cheap, cashing in section 3's
promise: past keys and values never change, so they are computed once
and stored. Watch one turn with context "I love": the cached K, V of
"I" and "love" meet a fresh query Q₃ (the weights land 30% / 70%), the
blend rides up the tower, softmax says *you* 85%, the draw picks
"you", and its K, V join the cache for the next turn. **Q is computed fresh; K and V are fetched from the cache** — that sentence is the whole story. You have felt it: the pause before a long prompt's first word is **prefill**, building the cache; afterwards, words stream. The bill is memory:

> cache = 2 × layers × context × width × bytes ≈ 2 × 32 × 100,000 × 4,096 × 2 ≈ **52 GB** for one long conversation

— which is why cached input is priced cheaper, and why the other big
serving lever is **quantization**: store the weights in fewer bits
(16 → 8 → 4). Inference is bound by moving bytes more than by math, so
smaller weights answer faster and cheaper.

## 8. It does not remember you

After training, the parameters are **frozen**. Every message replays
the whole conversation through the network — a brilliant consultant
with no long-term memory, handed the full case file every morning. What
feels like memory is the context window. The flip side is **in-context
learning**: show "sea → mer, house → maison, cat → ?" and out comes
*chat* — a task learned from the prompt alone, no parameter changed.
Practical prompt engineering is exactly this: arranging the context so
the desired continuation becomes the most probable one.

## 9. Why it makes things up

In 2023, lawyers in *Mata v. Avianca* filed six precedents invented by
ChatGPT — which, asked if they were real, said yes. A $5,000 fine made
"AI hallucination" famous. No mystery: the model is a probability
engine, not a database. Where training data is rich, the likeliest
continuation is usually true; where it is thin, the model still
produces something *shaped* like an answer — plausible, not true, is
what it optimizes. This is not lying — lying requires knowing the
truth. It is completing the sentence. The fixes form a ladder:
**prompting** shapes behavior in context; **RAG** fetches fresh
knowledge at answer time; **fine-tuning** bakes in what must be
permanent. Climb in that order — each step costs more — and check the
sources yourself: the step the lawyers skipped.

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

One last thing. In this article's first line, your mind wrote "time" into
the blank — instantly, confidently, from pure pattern. Now you know
exactly how a machine does the same. That is the whole story.

## Going deeper

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — the original transformer paper; the tired/wide example is theirs.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the classic visual walkthrough.
- Ebrahim Pichka, [What are Query, Key, and Value in the Transformer Architecture?](https://medium.com/data-science/what-are-query-key-and-value-in-the-transformer-architecture-and-why-are-they-used-acbe73f731f2) — a careful unpacking of the QKV intuition, including the soft-dictionary view.
- Andrej Karpathy, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the whole machine, written in code before your eyes.
