Every answer a large language model has ever given you — every essay, every
code snippet, every apology for a mistake it just made — was produced the same
way: by predicting the next small piece of text, over and over, one piece at a
time.

That sounds too simple to explain something that can pass exams and write
software. The interesting part is *what it takes* to predict the next word
well. To do that at a high level, a model ends up having to absorb grammar,
facts, style, and a working imitation of reasoning — because all of those are
needed to guess what comes next in human text. Hold on to that idea; everything
below is a footnote to it.

## Text becomes numbers: tokens

Computers do not read words. The first step is a **tokenizer**, which chops
text into pieces called tokens and gives each piece an ID number. A token is
often a whole common word ("the", "model"), sometimes a fragment of a rarer one
("quantization" might become "quant" + "ization"), sometimes punctuation.

Two practical consequences fall out of this:

- **Context limits are measured in tokens**, not words. When a model has a
  "128K context window", that is how many tokens it can consider at once.
- **Odd failures often live at the token level.** Counting the letters in a
  word is genuinely hard for a model that never sees letters — it sees token
  IDs. "Strawberry" is not s-t-r-a-w-b-e-r-r-y to the model; it may be two
  opaque chunks.

## Numbers with meaning: embeddings

Each token ID is then mapped to an **embedding**: a long list of numbers — a
vector, often thousands of dimensions — that represents the token's meaning.
These vectors are learned, not hand-designed, and they end up organized so that
similar meanings sit near each other: *king* near *queen*, *Paris* near
*Rome*.

This is the model's native language. From here on, everything is arithmetic on
vectors.

## Attention: reading in context

An embedding alone cannot say what "bank" means — river bank or the one with
the money? The word's meaning depends on its neighbors. This is the problem the
**transformer** architecture solves, and its central tool is **attention**.

Attention lets every token look at every other token in the input and decide
which ones matter for interpreting it. Processing the sentence "The engineer
finished her review because it was due", attention is what connects *her* back
to *the engineer* and *it* back to *the review*. No one programmed those links;
the model learned that resolving pronouns helps predict what comes next.

A transformer stacks this in **layers** — dozens to over a hundred of them.
Each layer refines the vectors using two blocks: attention (mix in context from
other tokens) and a small feed-forward network (transform each token's vector
on its own, where much of the model's stored knowledge lives). Early layers
capture grammar and local structure; deeper layers capture facts,
relationships, and longer-range logic. The "large" in *large language model*
counts the learned numbers in all these layers — the **parameters** — which run
from billions to trillions.

At the very top, the model turns the final vector into a score for every token
in its vocabulary, and converts the scores into probabilities: "given
everything so far, here is how likely each possible next token is."

## Training: where the knowledge comes from

How do the parameters get their values? **Pretraining**: show the model
enormous amounts of text — much of the public internet, books, code — hide the
next token, and let it guess. Wrong guess? Nudge every parameter a tiny amount
in the direction that would have made the right answer more likely (this is
gradient descent). Repeat trillions of times.

Nobody writes rules into the model. Grammar, geography, chemistry, Python — all
of it is absorbed as a side effect of getting better at one game: guess the
next token.

A pretrained model, though, is raw autocomplete. Ask it "What is the capital of
France?" and it might answer — or continue with nine more quiz questions,
because that is also a plausible continuation of the text. Two more stages turn
it into an assistant:

1. **Instruction tuning** — further training on examples of questions paired
   with good answers, teaching the *format* of being helpful.
2. **Learning from human preferences** (RLHF and its relatives) — people
   compare candidate answers, and the model is tuned toward the ones humans
   prefer: helpful, honest, harmless.

Same architecture, same next-token machinery — different behavior.

## Generation: one token at a time

When you send a prompt, the model does not plan an answer and then type it. It
computes the probability of every possible next token, **samples** one, appends
it to the text, and repeats — each new token immediately becoming part of the
input for the next prediction — until it emits a stop token.

It does not always pick the single most likely token. Always taking the top
choice produces repetitive, stilted text, so a bit of controlled randomness is
mixed in. The **temperature** setting scales it: low temperature makes output
focused and deterministic, higher temperature makes it varied and creative.
This is also why the same question can get different answers on different days.

## Why models make things up

Now the pieces explain the most famous failure mode. The model is a
probability machine over text, not a database with a lookup table. When you ask
about something well covered in training data, the most probable continuation
is usually the truth. When you ask about something obscure, there is no entry
to fail to find — the machinery keeps doing the only thing it does, and
produces a continuation that is *shaped* like a correct answer. A plausible
citation. A confident date. This is **hallucination**, and it is not a bug
bolted onto the system; it is the default behavior of the system, tamed but
not eliminated by the training stages above.

The practical fixes mostly work by changing the input, not the model: retrieval
(fetch real documents into the context and let the model answer from them),
tool use (let it call a search engine or run code), and asking for sources you
can check.

## The whole story in five lines

If you are revising for an interview, this is the skeleton worth keeping:

1. Text is split into **tokens**; each becomes a learned vector (**embedding**).
2. Stacked **transformer** layers use **attention** so every token's vector is
   informed by the context around it.
3. **Pretraining** on next-token prediction over vast text is where all the
   knowledge comes from; instruction tuning and human feedback shape it into an
   assistant.
4. Generation is a loop: predict probabilities for the next token, **sample**
   one, append, repeat. **Temperature** controls the randomness.
5. The model optimizes for *plausible*, not *true* — which is exactly why it is
   fluent, and exactly why it hallucinates.

None of the individual steps is magic. The surprise of the last few years is
what emerges when you do something this simple at sufficient scale.
