Every answer a language model has ever given you was produced the same way:
by predicting the next small piece of text, over and over. Your phone keyboard
does a tiny version of this — type "see you" and it offers *tomorrow*. An LLM
is that trick scaled up by billions, and the interesting part is what the game
demands: to predict the next word in human text well, a model must absorb
grammar, facts, style, and a working imitation of reasoning. Everything below
is a footnote to that idea.

## 1. Text becomes numbers

A **tokenizer** chops text into pieces called **tokens** — "the" is one piece,
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

## 3. Attention: reading in context

An embedding alone cannot say what "bank" means — river bank, or the one
with the money? Meaning depends on neighbors, and reading the neighbors is
attention's job. Compare, from the original Transformer paper:

> The animal didn't cross the street because **it** was too *tired*.
> The animal didn't cross the street because **it** was too *wide*.

One word changes and "it" switches sides. You resolved that instantly;
**attention** is how the **transformer** — the design behind every modern
model, the T in GPT — does the same. Every token plays three roles: a
**query** ("what am I looking for?"), a **key** ("how should others find
me?"), a **value** ("what do I hand over if picked?"). Think YouTube: your
search is a query, video titles are keys, the videos are values. In "it was
too tired", *it* asks for something earlier that could be tired; *animal*'s
key matches strongly, *street*'s weakly; the scores become percentages and
*it* rebuilds its vector as a weighted blend — say 85% *animal*, 10%
*street*. Every layer runs many attention "heads" in parallel, each learning its own relationship to track: grammar, references, which adjective belongs to which noun. Nobody assigns these roles; they emerge, because each one helps predict what comes next.

Note one asymmetry: a query fires once, but a token's key and value stay
relevant to every later token that looks back. Remember that — it becomes
money in a moment.

## 4. Layers: where knowledge lives

Stack this dozens to a hundred-plus times. In each layer, attention mixes in
context (the librarian) and a **feed-forward network** stores learned
patterns like "Paris pairs with France" (the warehouse — most **parameters** live here). Early layers pick up spelling and grammar; deeper layers, facts and long-range logic. GPT-2 made news in 2019 with 1.5 billion parameters; frontier
models now run to the trillions. At the top, **softmax** turns scores into
percentages summing to 100 — the model's entire output is a probability for
every token it knows. After "Once upon a", the mass piles onto "time"; after
"My favorite city is", it spreads across hundreds of cities. Both correctly
answer the only question the model ever answers: *what likely comes next?*

## 5. Training and scale

**Pretraining**: show the model trillions of tokens, hide the next one, let
it guess. A **loss function** scores its surprise; **gradient descent**
nudges every parameter a tiny step toward less surprise; repeat trillions of
times. Nobody writes rules in — predicting a detective novel's last chapter
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
model learns their taste, and the LLM is tuned toward it — comparing is far easier for people than authoring perfection, and comparisons capture what examples cannot spell out: tone, honesty about uncertainty, refusing harm. Both stages cost a small fraction of pretraining's months on thousands of GPUs. The punchline: GPT-3 existed
for two years before ChatGPT. The revolution was these stages, not a bigger
network.

## 7. Generation: a loop, not a plan

The model computes probabilities, **samples** one token (a weighted draw), appends it, and repeats — each new token instantly part of the next prediction's input — until a special stop token says "I'm done." Three dials govern the draw. After "The sky was":
*blue* 60%, *dark* 10%, … *potato* 0.0001%. **Temperature** sharpens or
flattens the list (low for SQL, high for brainstorming); **top-k** keeps only
the k most likely; **top-p** keeps the smallest set covering, say, 90% —
adapting between two tokens when the model is sure and eighty when it is
torn. Cut first, then draw: that is why answers vary day to day and the sky
is never a potato.

Two consequences of the loop: "think step by step" works because the page is
the model's only scratchpad — writing "17 × 24 = 340 + 68" makes each next
prediction easier, which reasoning models industrialize. And the **KV cache** cashes in the asymmetry from section 3. Token number 1,000 must compare its query against 999 earlier keys — which looks like re-reading everything at every step. It is not: past keys and values never change, so they are computed once and stored. The pause before a long prompt's first word is **prefill**, building that cache; afterwards words stream quickly because each pays only for itself; long chats eat memory because the cache grows with every token; and "cached input" is cheaper because it is already paid for.

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
because plausible, not true, is what it optimizes. The fixes change the
input: retrieval (RAG), tools, and checking sources yourself — the step the
lawyers skipped.

## The whole story in five lines

1. Text → **tokens** → **embeddings** (coordinates of meaning, position included).
2. **Attention** (query·key·value) blends each token's vector with its
   context; feed-forward layers store the knowledge.
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
