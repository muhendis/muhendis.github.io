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
similar meanings sit near each other: *king* near *queen*, *Paris* near *Rome*.
Even relationships become directions in this space — the famous party trick is
that the vector arithmetic *king − man + woman* lands near *queen*.

This is the model's native language. From here on, everything is arithmetic on
vectors.

One thing is still missing: order. A bag of vectors does not know that "dog
bites man" differs from "man bites dog", so each token's **position** in the
sequence is encoded into its representation too. Word order survives the trip
into math.

## Attention: reading in context

An embedding alone cannot say what "bank" means — river bank or the one with
the money? The word's meaning depends on its neighbors. This is the problem the
**transformer** architecture solves, and its central tool is **attention**.

The mechanism is best understood as matchmaking. Every token puts out a
**query** — a description of what it is looking for ("I am a pronoun; I need a
person mentioned earlier"). Every token also advertises a **key** — a
description of what it is ("I am a person, mentioned two words ago"). Where a
query and a key match strongly, information flows: the matched token hands over
its **value**, and the pronoun's vector is updated with it. Processing "The
engineer finished her review because it was due", this is what connects *her*
back to *the engineer* and *it* back to *the review*.

And it does not happen once. Each layer runs many attention "heads" in
parallel, and each head learns to track a different kind of relationship — one
follows grammar, another resolves references, another notices which adjective
modifies which noun. Nobody programs these roles; they emerge, because
each one helps predict what comes next.

## Layers: where the knowledge lives

A transformer stacks this machinery in **layers** — dozens to over a hundred.
Each layer has two blocks: attention (mix in context from other tokens) and a
**feed-forward network** (transform each token's vector on its own). The
feed-forward blocks hold most of the parameters, and a useful mental model is
that much of the model's stored *knowledge* lives there — patterns like "Paris
pairs with France" — while attention decides *which* knowledge the current
sentence calls for.

Early layers capture spelling and grammar; deeper layers capture facts,
relationships, and longer-range logic. The "large" in *large language model*
counts the learned numbers in all these layers — the **parameters** — from
billions to trillions.

At the very top, the model converts the final vector into a score for every
token in its vocabulary and squashes the scores into probabilities (the
**softmax** step). That is the model's entire output at each step: not a
sentence, not an idea — a probability for every token it knows.

## Training: where the knowledge comes from

How do the parameters get their values? **Pretraining**: show the model
enormous amounts of text — much of the public internet, books, code — hide the
next token, and let it guess. The **loss function** measures how surprised the
model was by the true answer; **gradient descent** then nudges every parameter
a tiny amount in the direction that would have made it less surprised. Repeat
trillions of times.

Nobody writes rules into the model. Grammar, geography, chemistry, Python — all
of it is absorbed as a side effect of getting better at one game: guess the
next token. You can think of the result as a lossy compression of its training
data — patterns kept, exact copies mostly not.

Scale is the other half of the story. Make the model bigger, feed it more data,
spend more compute, and the loss falls in a smooth, almost lawlike way — these
are the **scaling laws** that justified the enormous training runs. Along the
way, abilities show up that nobody targeted: translation, arithmetic, working
code. They emerge because each one helps with the only goal the model has.

## From autocomplete to assistant

A pretrained model is raw autocomplete. Ask it "What is the capital of France?"
and it might answer — or continue with nine more quiz questions, because that
is also a plausible continuation of the text. Two more stages turn it into an
assistant:

1. **Instruction tuning** — further training on examples of questions paired
   with good answers, teaching the *format* of being helpful.
2. **Learning from human preferences** (RLHF and its relatives) — people
   compare candidate answers, and the model is tuned toward the ones humans
   prefer: helpful, honest, harmless.

Same architecture, same next-token machinery — different behavior.

## Generation: a loop, not a plan

When you send a prompt, the model does not plan an answer and then type it. It
computes the probability of every possible next token, **samples** one, appends
it to the text, and repeats — each new token immediately becoming part of the
input for the next prediction — until it emits a stop token.

It does not always pick the single most likely token. Always taking the top
choice produces repetitive, stilted text, so a bit of controlled randomness is
mixed in. **Temperature** scales it — low is focused and deterministic, high is
varied and creative — and settings like top-p trim the truly unlikely options
before sampling. This is why the same question can get different answers on
different days.

This loop also explains why "think step by step" works. The model has no
scratchpad in its head — its only working memory is the text itself. When it
writes out intermediate steps, each step lands in the context and improves the
predictions that follow. Reasoning models industrialize exactly this: they are
trained to spend many tokens thinking out loud before committing to an answer.

## The model does not remember you

One more piece completes the picture: after training, the parameters are
**frozen**. Your conversation does not retrain the model. Each time you send a
message, the *entire* conversation is fed back through the network, and the
reply is predicted from that. What feels like memory is just the context
window — which is why very long chats slow down, hit limits, or lose track of
their beginnings.

The flip side is a genuine superpower called **in-context learning**: show the
model a few examples of a task inside the prompt, and it picks up the pattern
and continues it — without a single parameter changing. Much of practical
prompt engineering is exactly this: arranging the context so the desired
continuation becomes the most probable one.

## Why models make things up

Now the pieces explain the most famous failure mode. The model is a probability
machine over text, not a database with a lookup table. When you ask about
something well covered in training data, the most probable continuation is
usually the truth. When you ask about something obscure, there is no entry to
fail to find — the machinery keeps doing the only thing it does, and produces a
continuation that is *shaped* like a correct answer. A plausible citation. A
confident date. This is **hallucination**, and it is not a bug bolted onto the
system; it is the default behavior of the system, tamed but not eliminated by
the training stages above.

The practical fixes mostly work by changing the input, not the model: retrieval
(fetch real documents into the context and let the model answer from them),
tool use (let it call a search engine or run code), and asking for sources you
can check.

## The skeleton worth keeping

1. Text is split into **tokens**; each becomes a learned vector
   (**embedding**), with **position** encoded so word order survives.
2. Stacked **transformer** layers use **attention** — queries matching keys,
   many heads in parallel — so every token's vector is informed by its context;
   feed-forward blocks store most of the knowledge.
3. **Pretraining** on next-token prediction over vast text is where the
   knowledge comes from; **scaling laws** say more model, data, and compute
   predictably help; instruction tuning and human feedback shape the result
   into an assistant.
4. Generation is a loop: softmax gives a probability for every token,
   **sampling** picks one, the choice feeds the next step. **Temperature**
   controls the randomness; step-by-step "thinking" is the model using its own
   output as working memory.
5. Parameters are **frozen** after training — apparent memory is the context
   window, and **in-context learning** is patterns picked up from the prompt
   alone.
6. The model optimizes for *plausible*, not *true* — which is exactly why it is
   fluent, and exactly why it hallucinates.

None of the individual steps is magic; the surprise is what emerges when
something this simple is done at sufficient scale. And the next time someone
asks you how these models work — an interviewer across the table, a student
after class, or the curious voice in your own head — you can start exactly
where the model itself starts: with the next token.
