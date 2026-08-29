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
split: "unbelievable" might become "un" + "believ" + "able". A useful rule of thumb
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

Each token ID is then mapped to an **embedding** — a vector: a list of numbers, often thousands of them, that acts like the token's coordinates on a map of meaning. On a real map, Paris sits near Brussels and far from
Sydney. On the meaning-map, *king* sits near *queen* and *throne*, far from
*spreadsheet*. Even relationships become consistent directions: the arrow from
*Paris* to *France* points the same way as the arrow from *Rome* to *Italy* —
a "capital-of" direction. A famous demonstration is vector arithmetic:
*king − man + woman* lands near *queen*.

Nobody drew this map by hand. It is learned, and it is the model's native
language — from here on, everything is arithmetic on vectors.

One thing is still missing: order. A bag of coordinates does not know that
"dog bites man" is news and "man bites dog" is a headline, so each token's **position** in the sequence is stamped into its vector as well. Word
order survives the trip into math.

## Attention: reading in context

Coordinates alone cannot say what "bank" means — river bank or the one with
the money? The word's meaning depends on its neighbors. This is the problem the **transformer** solves — the neural-network design behind every modern language model, and the T in GPT. Its central tool is **attention**.

The cleanest demonstration comes from the original Transformer paper's own
example. Compare:

> The animal didn't cross the street because **it** was too *tired*.
> The animal didn't cross the street because **it** was too *wide*.

Swap one word at the end, and "it" flips its meaning — tired points at the
animal, wide points at the street. You resolved that instantly and
unconsciously. Attention is the machinery that lets a model do the same.

How? Inside the model, every token wears three hats — each one a different
small transformation of its vector:

- a **query**: "what am I looking for?"
- a **key**: "how should others find me?"
- a **value**: "what do I hand over if I'm picked?"

The closest everyday machine is a search engine. What you type into YouTube is
a query. Every video's title and description is a key. The videos themselves
are the values. The engine compares your query against all the keys, scores
the matches, and serves you the content behind the best ones.

Attention runs that search for every token at once. In "it was too *tired*",
the token *it* issues a query that says, roughly, "I need a thing mentioned
earlier that could be tired." The key of *animal* matches that query strongly;
the key of *street* matches it weakly. The match scores are turned into
percentages that sum to 100, and *it* updates its vector with a blend of the
values weighted by those percentages — say 85% of *animal*'s, 10% of
*street*'s, a little from the rest. Nothing is ever copied whole: every update
is a mixture, weighted by relevance. Swap *tired* for *wide*, and the very
same machinery flips the weights toward *street*.

Notice the asymmetry in the three roles. A token's query is used exactly
once — at the moment that token looks around. Its key and value, though, stay
relevant for the rest of the text: every later token that looks back must
compare against this key, and may draw on this value. Hold that thought; in
the generation section it turns into real money.

And it does not happen once. Each layer runs many attention "heads" in
parallel, and each head learns to track a different kind of relationship — one
follows grammar, another resolves references like the example above, another
notices which adjective modifies which noun. Nobody assigns these roles; they
emerge, because each one helps predict what comes next.

## Layers: where the knowledge lives

A transformer stacks this machinery in **layers** — dozens to over a hundred.
Each layer has two blocks: attention (mix in context from other tokens) and a **feed-forward network** — a small neural network applied to each token's vector on its own. A useful
mental model: the feed-forward blocks are the warehouse where learned patterns
are stored — "Paris pairs with France", "code after `def` is a function
name" — and attention is the librarian deciding which shelf the current
sentence needs.

Early layers capture spelling and grammar; deeper layers capture facts,
relationships, and longer-range logic. The "large" in *large language model*
counts the learned numbers in all these layers — the **parameters**. GPT-2
made headlines in 2019 with 1.5 billion of them; today's frontier models are
measured in the hundreds of billions to trillions.

At the very top, the model turns the final vector into a score for every token in its vocabulary, then converts the scores into percentages that add up to 100% — the **softmax** step, the same scores-to-percentages move attention used internally. That is the model's entire output at each step: not a
sentence, not an idea — a probability for every token it knows. After "Once
upon a", nearly all of the probability piles onto "time". After "My favorite
city is", it spreads across hundreds of plausible cities. Both are correct answers to the only question the model ever answers: what is likely to come next?

## Training: where the knowledge comes from

How do the parameters get their values? **Pretraining**: show the model
enormous amounts of text — much of the public internet, books, code; trillions
of tokens, more than a human could read in ten thousand lifetimes — hide the
next token, and let it guess. The **loss function** measures how surprised the
model was by the true answer; an algorithm called **gradient descent** then nudges every parameter
a tiny amount in the direction that would have made it less surprised. Repeat,
trillions of times.

Nobody writes rules into the model. Grammar, geography, chemistry, Python — all
of it is absorbed as a side effect of getting better at one game: guess the
next token. To predict the next word of a detective novel's final chapter, it
helps to have tracked who had a motive; to predict the next line of a physics
textbook, it helps to have internalized some physics. You can think of the result as a compressed copy of the training data — compressed the way a JPEG compresses a photo: the overall picture is kept, the exact pixels are not.

## Scaling laws: why bigger kept getting better

Through the 2010s, "make it bigger" was a hunch. In 2020, researchers at
OpenAI measured it and found something stronger: the relationship between
scale and performance follows a **power law** — smooth and remarkably
regular. Multiply the compute budget by ten, and the loss — the model's
average surprise — falls by a predictable amount. Not a guarantee that bigger
is smarter in every respect, but a measured, repeatable curve that holds
across many orders of magnitude.

That regularity changed the economics of the field. If you can forecast how
good a model will be before spending the money, a hundred-million-dollar
training run stops being a gamble and becomes an engineering plan. OpenAI
later reported predicting GPT-4's final loss in advance, from trial models
trained with less than 1/10,000 of the compute — the curve, extended, landed
where it said it would.

A second finding refined the recipe. In 2022, DeepMind's **Chinchilla** study
showed that the field had the balance wrong: models had grown too large for
the amount of data they were trained on. Parameters and training tokens
should grow together — as a rule of thumb, roughly twenty tokens of text per
parameter. The proof was direct: their 70-billion-parameter model, trained on
far more data, outperformed a 280-billion-parameter rival. Since then, "how
big is it?" is always paired with "trained on how much?"

One caveat keeps the story honest. The loss falls smoothly, but individual
skills do not always appear smoothly. A model can score near zero on
three-digit arithmetic at several sizes in a row, then handle it reliably at
the next jump — an **emergent ability**. Researchers still debate whether
these jumps are truly sudden or partly an artifact of how the skills are
graded; the practical point is that a smooth curve of prediction quality can
conceal abrupt arrivals of capability. And the raw material is not infinite:
high-quality public text is close to exhausted, which is why the frontier has
been shifting toward synthetic training data and toward spending compute at
answer time — the reasoning models that appear later in this article.

## From autocomplete to assistant

What pretraining produces is called a **base model**, and it is worth being
precise about what that is: a machine that continues text — nothing more. It
has no job description, no notion that a question aimed at it is *its* to
answer. It has simply read the internet, where text follows text.

Ask a base model "What is the capital of France?" and you might get "Paris."
You might just as easily get "What is the capital of Germany? What is the
capital of Spain?" — on the internet, quiz questions travel in packs — or
"asked the teacher, and nobody raised a hand," continuing the scene as
fiction. All three are faithful continuations. In the base-model era, getting
answers out required tricks like writing "Q: ... A:" so that an answer became
the likeliest continuation. Prompt engineering was born there.

Turning this raw material into an assistant takes two further stages. Neither
changes the architecture; both are simply more next-token training, on
carefully chosen text.

**Stage one: instruction tuning.** People — increasingly helped by models —
write tens of thousands of example dialogues, each an instruction paired with
an ideal response:

> **User:** Summarize this email in two sentences.
> **Assistant:** (a genuinely good two-sentence summary)

Train on enough of these, and "I am an assistant; a question is for answering;
this is what helpful looks like" becomes the most probable continuation. The
format of being helpful is learned the same way everything else was — from
examples.

**Stage two: learning from human preferences**, known as **RLHF**
(reinforcement learning from human feedback). Have the model produce several
answers to the same prompt. Show pairs to human reviewers: *which one is
better?* Train a second model — a **reward model** — to predict those
judgments, then tune the LLM toward answers the reward model scores highly.
Why the detour? Because people are far better at *comparing* two answers than
at writing perfect ones, and because comparisons capture what examples
struggle to spell out: tone, honesty about uncertainty, declining harmful
requests.

The two halves are sharply unequal in cost: pretraining takes months on
thousands of GPUs; the assistant stages are a small fraction of that. And the
gap between them is a gap you have personally felt. GPT-3 — the base model —
existed for more than two years before ChatGPT. What turned a research
curiosity into the fastest-growing product in history was not a bigger
network. It was these two stages, bolted onto the same next-token machine.

## Generation: a loop, not a plan

When you send a prompt, the model does not plan an answer and then type it. It
computes the probability of every possible next token, **samples** one — draws it at random, weighted by its probability — appends it to the text, and repeats — each new token immediately becoming part of the
input for the next prediction — until it produces a special stop token that means "I'm done."

This loop hides an engineering problem. When token number 1,000 is being
generated, its query must be compared against the keys of all 999 tokens
before it — which, at first glance, seems to require re-processing the entire
text at every step. It does not, and the reason is the asymmetry noted in the
attention section: the keys and values of past tokens never change once
computed. So the model computes them once and keeps them in memory — the
**KV cache**. Each new token then does only its own small share of work: it
compares its fresh query against the stored keys, blends the stored values,
and appends its own key and value to the cache for the tokens still to come.

You have felt this cache without knowing its name. Send a long prompt and
there is a pause before the first word appears — that is **prefill**, the
model building the cache for your entire input. After it, words stream out
quickly, because each one pays only for itself. The cache is also why long
conversations consume serious memory — it grows with every token, in every
layer — and why API providers charge less for "cached" input: when the
beginning of your prompt has not changed, its keys and values are already
sitting there, paid for.

It does not always pick the single most likely token — always taking the top
choice produces repetitive, stilted text — so the pick is a controlled
lottery, governed by three dials worth knowing by name. Concretely: after
"The sky was", the model's list might read *blue* 60%, *clear* 20%, *dark*
10%, *grey* 5%, then a tail of thousands of tokens at tiny probabilities —
including, somewhere far down, *potato* at 0.0001%.

- **Temperature** reshapes the list before the draw. Low temperature
  exaggerates the leader: *blue* wins almost every time — what you want when
  extracting data or writing SQL. High temperature flattens the list: *dark*
  and *grey* get real chances, and once in a while you get *a bruised shade
  of purple over the harbor* — what you want when brainstorming.
- **Top-k** cuts the list to a fixed length before sampling. With k = 50,
  only the 50 most likely tokens stay in the draw; the tail — *potato*
  included — is simply deleted.
- **Top-p** cuts by probability mass instead of count: keep the smallest set
  of tokens whose percentages add up to p — say 90% — and discard the rest.
  The clever part is that this set adapts. When the model is confident
  ("Once upon a"), the 90% set may hold two tokens; when it is genuinely
  torn ("My favorite city is"), it may hold eighty. That adaptiveness is why
  top-p is the more common choice.

Cut first, then draw from the survivors. This is why the same question can
get different answers on different days — and why the sky is never completed
with a potato.

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

The practical fixes mostly change the input, not the model: retrieval (often called RAG: fetch real documents into the context and let the model answer from them), tool use
(let it call a search engine or run code), and asking for sources you can
check yourself — the step the lawyers skipped.

## The skeleton worth keeping

1. Text is split into **tokens**; each becomes a learned vector
   (**embedding**) — coordinates on a map of meaning — with **position**
   encoded so word order survives.
2. Stacked **transformer** layers use **attention** — queries matching keys,
   values blended by relevance, many heads in parallel — so every token's vector is informed by its
   context ("it was too tired" vs "it was too wide"); feed-forward blocks
   store most of the knowledge.
3. **Pretraining** on next-token prediction over vast text is where the
   knowledge comes from; **scaling laws** make the gains
   from more model, data, and compute predictable — provided the three grow
   in balance. Pretraining alone yields a **base model** — raw
   autocomplete; instruction tuning and RLHF shape it into an assistant.
4. Generation is a loop: softmax gives a probability for every token,
   **sampling** picks one, the choice feeds the next step. **Temperature**, **top-k**, and **top-p** control the randomness; step-by-step "thinking" is the model using the
   page as its scratchpad; the **KV cache** stores past tokens' keys and
   values so each step pays only for the newest token.
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
