Every answer a large language model has ever given you — every essay, every
code snippet, every apology for a mistake it just made — was produced the same
way: by predicting the next small piece of text, over and over, one piece at a
time.

Your phone already does a tiny version of this. Type "see you" and the keyboard
suggests *tomorrow*, *soon*, *later*. A large language model is that same trick
scaled up by a factor of billions — and the interesting part is *what it takes*
to play the game that well. To predict the next word at a high level, a model
ends up having to absorb grammar, facts, style, and a working imitation of
reasoning, because all of those are needed to guess what comes next in human
text. Hold on to that idea; everything below is a footnote to it.

## Text becomes numbers: tokens

Computers do not read words. The first step is a **tokenizer**, which chops
text into pieces called tokens and gives each piece an ID number. Common words
survive whole — "the" is one token, "model" is one token. Rarer words get
split: "unbelievable" might become "un" + "believ" + "able", and a Turkish word
like "kuantizasyon" might become "kuant" + "izasyon". A useful rule of thumb
for English: **100 tokens is roughly 75 words**, so a "128K context window"
means the model can hold about 96,000 words in view at once — a full novel.

Two practical consequences fall out of this:

- **Context limits are measured in tokens**, not words or pages.
- **Odd failures often live at the token level.** For years, models famously
  miscounted the r's in "strawberry". That is not stupidity about fruit — the
  model never sees letters. To it, "strawberry" is one or two opaque chunks,
  and asking it to count letters is like asking you to count the brushstrokes
  in a photo of a painting.

## Numbers with meaning: embeddings

Each token ID is then mapped to an **embedding**: a long list of numbers — a
vector, often thousands of dimensions — that acts like the token's coordinates
on a map of meaning. On a real map, Paris sits near Brussels and far from
Sydney. On the meaning-map, *king* sits near *queen* and *throne*, far from
*spreadsheet*. Even relationships become consistent directions: the arrow from
*Paris* to *France* points the same way as the arrow from *Rome* to *Italy* —
a "capital-of" direction. The famous party trick is vector arithmetic:
*king − man + woman* lands near *queen*.

Nobody drew this map by hand. It is learned, and it is the model's native
language — from here on, everything is arithmetic on vectors.

One thing is still missing: order. A bag of coordinates does not know that
"dog bites man" is news and "man bites dog" is a headline, so each token's
**position** in the sequence is encoded into its representation too. Word
order survives the trip into math.

## Attention: reading in context

Coordinates alone cannot say what "bank" means — river bank or the one with
the money? The word's meaning depends on its neighbors. This is the problem the
**transformer** architecture solves, and its central tool is **attention**.

The cleanest demonstration comes from the original Transformer paper's own
example. Compare:

> The animal didn't cross the street because **it** was too *tired*.
> The animal didn't cross the street because **it** was too *wide*.

Swap one word at the end, and "it" flips its meaning — tired points at the
animal, wide points at the street. You resolved that instantly and
unconsciously. Attention is the machinery that lets a model do the same.

It works like matchmaking. Every token puts out a **query** — a description of
what it is looking for ("I am a pronoun; I need a thing mentioned earlier that
could be *tired*"). Every token also advertises a **key** — a description of
what it is ("I am an animal, four words back"). Where a query and a key match
strongly, information flows: the matched token hands over its **value**, and
the pronoun's vector is updated to mean, effectively, *the animal*.

And it does not happen once. Each layer runs many attention "heads" in
parallel, and each head learns to track a different kind of relationship — one
follows grammar, another resolves references like the example above, another
notices which adjective modifies which noun. Nobody assigns these roles; they
emerge, because each one helps predict what comes next.

## Layers: where the knowledge lives

A transformer stacks this machinery in **layers** — dozens to over a hundred.
Each layer has two blocks: attention (mix in context from other tokens) and a
**feed-forward network** (transform each token's vector on its own). A useful
mental model: the feed-forward blocks are the warehouse where learned patterns
are stored — "Paris pairs with France", "code after `def` is a function
name" — and attention is the librarian deciding which shelf the current
sentence needs.

Early layers capture spelling and grammar; deeper layers capture facts,
relationships, and longer-range logic. The "large" in *large language model*
counts the learned numbers in all these layers — the **parameters**. GPT-2
made headlines in 2019 with 1.5 billion of them; today's frontier models are
measured in the hundreds of billions to trillions.

At the very top, the model converts the final vector into a score for every
token in its vocabulary and squashes the scores into probabilities — the
**softmax** step. That is the model's entire output at each step: not a
sentence, not an idea — a probability for every token it knows. After "Once
upon a", nearly all of the probability piles onto "time". After "My favorite
city is", it spreads across hundreds of plausible cities. Both of those
distributions are the correct answer to the only question the model ever
answers.

## Training: where the knowledge comes from

How do the parameters get their values? **Pretraining**: show the model
enormous amounts of text — much of the public internet, books, code; trillions
of tokens, more than a human could read in ten thousand lifetimes — hide the
next token, and let it guess. The **loss function** measures how surprised the
model was by the true answer; **gradient descent** then nudges every parameter
a tiny amount in the direction that would have made it less surprised. Repeat,
trillions of times.

Nobody writes rules into the model. Grammar, geography, chemistry, Python — all
of it is absorbed as a side effect of getting better at one game: guess the
next token. To predict the next word of a detective novel's final chapter, it
helps to have tracked who had a motive; to predict the next line of a physics
textbook, it helps to have internalized some physics. You can think of the
result as a lossy compression of the training data — patterns kept, exact
copies mostly not.

Scale is the other half of the story. Make the model bigger, feed it more
data, spend more compute, and the loss falls in a smooth, almost lawlike way —
these are the **scaling laws** that justified the enormous training runs. Along
the way, abilities show up that nobody targeted: translation, arithmetic,
working code. They emerge because each one helps with the only goal the model
has.

## From autocomplete to assistant

A pretrained model is raw autocomplete, and it behaves like it. Ask it "What
is the capital of France?" and it might answer — or it might continue with
nine more quiz questions, because on the internet, one quiz question is
usually followed by another. Two more stages turn it into an assistant:

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

It does not always pick the single most likely token; always taking the top
choice produces repetitive, stilted text. Instead a bit of controlled
randomness is mixed in, and **temperature** scales it. At low temperature,
"The sky was" continues with *blue* almost every time — the right setting for
extracting data or writing SQL. At high temperature it might continue with *a
bruised shade of purple over the harbor* — the right setting for
brainstorming. Settings like top-p trim the truly unlikely options before
sampling. This is also why the same question can get different answers on
different days.

The loop also explains why "think step by step" genuinely works. Ask a model
for 17 × 24 in one leap, and it must hit the answer in a single next-token
guess. Let it write "17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = ..." and every
intermediate step lands in the context, sharpening the predictions that
follow — the model has no scratchpad in its head, so it uses the page as one.
Reasoning models industrialize exactly this: they are trained to spend many
tokens thinking out loud before committing to an answer.

## The model does not remember you

One more piece completes the picture: after training, the parameters are
**frozen**. Your conversation does not retrain the model. It is like working
with a brilliant consultant who has no long-term memory — every single
morning, you must hand over the entire case file again. That is literally what
happens: each time you send a message, the *whole* conversation is fed back
through the network, and the reply is predicted from that. What feels like
memory is just the context window — which is why very long chats slow down,
hit limits, or lose track of their beginnings.

The flip side is a genuine superpower called **in-context learning**. Put this
in a prompt:

> sea → mer, house → maison, cat → ?

and the model answers *chat* — French — having inferred the task from two
examples, without a single parameter changing. Show it three support tickets
labeled *urgent* or *routine* and it will label the fourth. Much of practical
prompt engineering is exactly this: arranging the context so the desired
continuation becomes the most probable one.

## Why models make things up

Now the pieces explain the most famous failure mode. In 2023, two New York
lawyers filed a brief in *Mata v. Avianca* citing six airline cases — complete
with case names, docket numbers, and quotable judicial reasoning. None of them
existed. ChatGPT had invented all six, and when the lawyers asked it whether
the cases were real, it assured them they were. The court fined the lawyers
$5,000, and "AI hallucination" entered the mainstream vocabulary.

The machinery above explains it. The model is a probability engine over text,
not a database with a lookup table. Ask about something well covered in
training data, and the most probable continuation is usually the truth. Ask
for case law that does not exist, and there is no entry to fail to find — the
machinery keeps doing the only thing it does and produces a continuation
*shaped* like a correct answer: plausible names, plausible citations,
confident tone. **Hallucination** is not a bug bolted onto the system; it is
the default behavior of the system, tamed but not eliminated by the training
stages above.

The practical fixes mostly change the input, not the model: retrieval (fetch
real documents into the context and let the model answer from them), tool use
(let it call a search engine or run code), and asking for sources you can
check yourself — the step the lawyers skipped.

## The skeleton worth keeping

1. Text is split into **tokens**; each becomes a learned vector
   (**embedding**) — coordinates on a map of meaning — with **position**
   encoded so word order survives.
2. Stacked **transformer** layers use **attention** — queries matching keys,
   many heads in parallel — so every token's vector is informed by its
   context ("it was too tired" vs "it was too wide"); feed-forward blocks
   store most of the knowledge.
3. **Pretraining** on next-token prediction over vast text is where the
   knowledge comes from; **scaling laws** say more model, data, and compute
   predictably help; instruction tuning and human feedback shape the result
   into an assistant.
4. Generation is a loop: softmax gives a probability for every token,
   **sampling** picks one, the choice feeds the next step. **Temperature**
   controls the randomness; step-by-step "thinking" is the model using the
   page as its scratchpad.
5. Parameters are **frozen** after training — apparent memory is the context
   window, and **in-context learning** is patterns picked up from the prompt
   alone (*sea → mer, cat → chat*).
6. The model optimizes for *plausible*, not *true* — which is exactly why it
   is fluent, and exactly how six fake cases ended up in a federal court
   filing.

None of the individual steps is magic; the surprise is what emerges when
something this simple is done at sufficient scale. And the next time someone
asks you how these models work — an interviewer across the table, a student
after class, or the curious voice in your own head — you can start exactly
where the model itself starts: with the next token.
