Ask a frontier model how many r's are in "strawberry" and it may still
get it wrong. Ask it to reverse a word and it fumbles. Meanwhile the
same model writes a working compiler pass. The failure looks like a
reasoning gap, but it is not: the model never saw the letters. By the
time "strawberry" reached the first layer it had become three integers
— `str`, `aw`, `berry` — and you cannot count what was never in the
input.

Every prompt passes through a desk before it reaches the network. At
that desk, text is chopped into pieces from a fixed list and each
piece is stamped with an integer. Nothing crosses without a stamp, and
the network only ever sees the stamps. This article works through that
desk: why the fixed list exists, how BPE builds it, how the algorithms
differ, what vocabulary size costs, why Turkish pays more than English
at the same desk, the failure modes the desk creates — and how the
integers finally become vectors.

**In this article**

- [1. Why a fixed list exists at all](#1-why-a-fixed-list-exists-at-all)
- [2. How BPE builds the list](#2-how-bpe-builds-the-list)
- [3. Four algorithms, one table](#3-four-algorithms-one-table)
- [4. The token ID is just a row number](#4-the-token-id-is-just-a-row-number)
- [5. What vocabulary size buys and costs](#5-what-vocabulary-size-buys-and-costs)
- [6. Why Turkish pays more at the same desk](#6-why-turkish-pays-more-at-the-same-desk)
- [7. What the desk breaks](#7-what-the-desk-breaks)
- [8. From stamp to vector](#8-from-stamp-to-vector)
- [The whole story in six lines](#the-whole-story-in-six-lines)
- [Glossary](#glossary)
- [Going deeper](#going-deeper)

## 1. Why a fixed list exists at all

A neural network multiplies matrices. It cannot multiply the letter
"h". So something must map text to numbers, and the mapping has to be
a lookup into a finite list — because the model's first layer is a
table with one row per list entry, and tables need a fixed number of
rows.

> **Vocabulary** = the fixed, finite list of string pieces a tokenizer
> knows. Every entry has a row number. Text that cannot be built from
> these pieces cannot enter the model.

The obvious list is *every word*. It fails twice. English alone runs
to hundreds of thousands of forms once you count inflections, proper
nouns, code identifiers and typos, and each one needs its own row in
the embedding matrix. Worse, the list is closed at training time: the
first word your users type that is not on it — a new product name, a
misspelling — maps to a single `[UNK]` token and its meaning is gone.

The opposite list is *every character*. The vocabulary shrinks to a
few hundred, and nothing is ever unknown. But now "tokenization" costs
twelve slots instead of two, and attention cost grows roughly with the
square of sequence length. You pay quadratically for pieces that carry
almost no meaning on their own.

Subword tokenization takes the middle. Common words stay whole;
rare ones break into parts that are still meaningful.

```text
"tokenization"   →  ["token", "ization"]     2 pieces, both meaningful
"antidisestablishmentarianism"
                 →  ["anti", "dis", "establish", "ment", "arian", "ism"]
"Zyxwvut"        →  ["Zy", "x", "w", "vut"]  never unknown, just longer
```

The deal is explicit: frequent text is cheap, rare text is expensive,
and nothing is impossible. That trade shapes everything below.

## 2. How BPE builds the list

**Byte pair encoding (BPE)** builds the vocabulary by starting from
single characters and repeatedly gluing together whichever adjacent
pair appears most often. It is a counting loop, not a grammar.

Here is the walkthrough from the Hugging Face tokenizer summary, with
a corpus of five words and their frequencies:

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

Base vocabulary: ["b", "g", "h", "n", "p", "s", "u"]
Split into characters:
("h" "u" "g", 10) ("p" "u" "g", 5) ("p" "u" "n", 12)
("b" "u" "n", 4)  ("h" "u" "g" "s", 5)
```

Now count adjacent pairs. The pair `"u" "g"` appears in *hug* (10),
*pug* (5) and *hugs* (5) — 20 times. The pair `"u" "n"` appears in
*pun* (12) and *bun* (4) — 16 times. So `ug` wins the first merge:

```text
Merge 1:  u + g → "ug"     (20 occurrences)
("h" "ug", 10) ("p" "ug", 5) ("p" "u" "n", 12)
("b" "u" "n", 4) ("h" "ug" "s", 5)

Merge 2:  u + n → "un"     (16 occurrences)
("h" "ug", 10) ("p" "ug", 5) ("p" "un", 12)
("b" "un", 4)  ("h" "ug" "s", 5)

Vocabulary now: ["b","g","h","n","p","s","u","ug","un"]
```

Repeat until the vocabulary hits its target size. Two consequences
matter in practice.

First, **the merge list is the tokenizer**. Encoding new text means
replaying those merges in the order they were learned, so the merge
order is part of the artifact — not just the final word list.

Second, **vocabulary size is arithmetic**: base characters plus number
of merges. GPT-1 used 40,478 tokens, which is 478 base tokens plus
40,000 merges. GPT-2 used byte-level BPE: 256 byte values as the base,
plus 50,000 merges, plus one end-of-text token — 50,257 exactly.

That byte-level base is the quiet fix for `[UNK]`. If your atoms are
the 256 possible byte values, then *every* possible input — emoji,
Chinese, corrupted bytes, a PNG pasted as text — is representable.
Unknown tokens stop existing. This is why GPT-4 and Llama 3 both use
byte-level BPE.

## 3. Four algorithms, one table

BPE is the most common but not the only option. The four differ mainly
in **how they decide what to merge or keep**.

| Algorithm | Starts from | Decision rule | Used by |
| :--- | :--- | :--- | :--- |
| **BPE** | characters | merge the **most frequent** adjacent pair | GPT-2, Llama, Qwen2 |
| **Byte-level BPE** | 256 byte values | same rule, but no `[UNK]` is possible | GPT-2/4, Llama 3 |
| **WordPiece** | characters + `##` | merge the pair that is most **surprising** | BERT, DistilBERT |
| **Unigram** | a large candidate pool | **prune** the tokens the corpus needs least | T5, Pegasus |

The WordPiece row is the one worth understanding, because "most
surprising" sounds vague and is actually a formula:

```text
score(a, b) = freq(ab) / ( freq(a) × freq(b) )
```

BPE asks *how often do these two appear together*. WordPiece asks *how
much more often than chance*. In the corpus above, `u`+`g` occurs 20
times and scores 20/(36×20) = 0.028, while `g`+`s` occurs only 5 times
but scores 5/(20×5) = 0.050 — so WordPiece merges `gs` first. A pair
of common characters has to work much harder to earn a merge, which
pushes the vocabulary toward pieces that genuinely belong together.

**Unigram** runs backwards. It starts with far too many candidate
subwords and repeatedly asks, for each one, *how much worse would the
corpus be tokenized if I deleted this?* Then it deletes the bottom
10–20%. Removing `"pu"` is nearly free, because `pug` and `pun` can
still be spelled `["p","ug"]` and `["p","un"]`. Removing `"ug"` is
expensive, because three words lean on it. One side effect: Unigram
can spell a word several ways and pick the most probable, so it is
probabilistic where BPE is deterministic.

**SentencePiece** is not a fourth algorithm — it is a library that
runs BPE or Unigram on the raw character stream. Its actual
contribution is treating the space as a normal character, written
`▁`. That matters for Chinese, Japanese and Thai, which do not put
spaces between words, and it makes decoding exactly reversible: glue
the tokens together, turn `▁` back into a space, done.

Before any of this runs, a **pre-tokenizer** splits the raw text into
chunks — usually at whitespace and punctuation — so that merges can
never span a word boundary and glue the end of one word to the start
of the next. GPT-family tokenizers use a regex here that also isolates
contractions like `'s` and caps digit runs at three, which is why
numbers tokenize in a way that quietly hurts arithmetic.

## 4. The token ID is just a row number

Once the vocabulary exists, each entry gets an integer. These integers
are the model's entire view of your text — and they are arbitrary. I
ran these through `tiktoken` while writing this section:

```python
import tiktoken

gpt4  = tiktoken.get_encoding("cl100k_base")   # GPT-4
gpt4o = tiktoken.get_encoding("o200k_base")    # GPT-4o

gpt4.encode("hello")    # [15339]
gpt4o.encode("hello")   # [24912]
```

Same word, same company, different tokenizer, unrelated numbers. The
ID is a row number in a table, nothing more. ID 24912 is not "bigger"
or "later" or "more positive" than ID 15339, and the model never does
arithmetic on it.

Two sensitivities surprise people, and both follow directly from the
fact that the vocabulary stores *exact strings*:

```python
gpt4.encode("hello")   # [15339]          one token
gpt4.encode("Hello")   # [9906]           one token, unrelated ID
gpt4.encode("HELLO")   # [51812, 1623]    two tokens: "HEL" + "LO"

gpt4.encode(" egg")    # [19151]          one token — space included
gpt4.encode("egg")     # [29468]          different token entirely
```

Capitalisation changes the string, so it changes the token. Shouting
costs extra, because `HELLO` is rare enough that no single row was
spent on it. And the leading space is *part of the token* — `" egg"`
and `"egg"` are different rows. Byte-level BPE writes that space as
`Ġ`, SentencePiece as `▁`; both are just visible stand-ins for a
character that would otherwise be invisible in the vocabulary file.

This is the mechanical reason a trailing space in your prompt can
change a model's output. You did not add whitespace; you selected a
different set of rows.

## 5. What vocabulary size buys and costs

Vocabulary size (`V`) is one of the few architectural numbers you
choose before training and can never change afterwards. It trades two
costs against each other.

```text
Small V (32,000)                     Large V (256,000)
├─ small embedding matrix            ├─ large embedding + output matrix
├─ cheap output softmax              ├─ expensive output softmax
└─ LONGER sequences                  └─ SHORTER sequences
   (more tokens per sentence)           (more text per context window)
```

The embedding matrix has `V × d` parameters, and the output layer that
predicts the next token is the same shape. At a hidden size of 4,096,
going from 32k to 256k vocabulary adds roughly 918M parameters per
matrix — real memory, spent before a single layer of attention.

What you buy is shorter sequences. Since attention cost grows with the
square of sequence length, and your context window is denominated in
tokens, fewer tokens per sentence means more text fits and each
forward pass is cheaper. The industry has been walking steadily
toward larger vocabularies as a result:

| Generation | Vocabulary | Notes |
| :--- | :--- | :--- |
| Llama 1 / 2, Mistral 7B | 32,000 | SentencePiece; non-English fragments badly |
| GPT-2 | 50,257 | 256 bytes + 50,000 merges + `<|endoftext|>` |
| GPT-4 (`cl100k_base`) | 100,277 | measured with `tiktoken` |
| GPT-4o (`o200k_base`) | 200,019 | measured with `tiktoken` |
| Gemma | 256,000 | SentencePiece, byte-level fallback |

Note the shape of that progression: it is driven almost entirely by
multilingual and code text, which is exactly where small vocabularies
hurt most. Which brings us to the bill.

## 6. Why Turkish pays more at the same desk

Turkish is **agglutinative** — a word is built by stacking suffixes
onto a root, so one Turkish word often carries what English spreads
across a phrase. A tokenizer trained mostly on English has spent its
merge budget on English strings, so the same meaning costs more
tokens.

> **Fertility** = the average number of tokens a tokenizer spends per
> word. Higher fertility means the same sentence costs more money,
> more latency, and more of your context window.

I measured this rather than repeating the folklore. The same sentence
pair, through both GPT tokenizers:

```text
EN: "Hello world, the weather is very nice today."
TR: "Merhaba dünya, bugün hava çok güzel."

cl100k_base (GPT-4):   EN 10 tokens  →  TR 14 tokens   (1.4×)
o200k_base  (GPT-4o):  EN 10 tokens  →  TR  9 tokens   (0.9×)
```

A second, longer pair shows the same direction: 13 EN vs 23 TR tokens
under `cl100k_base`, but 13 vs 18 under `o200k_base`. The vocabulary
expansion between GPT-4 and GPT-4o did real work for Turkish — the
penalty did not vanish, but it roughly halved.

It is worth being precise about *why* the penalty exists, because the
common explanation is wrong. BPE is not incapable of finding Turkish
morphemes. Look at what `cl100k_base` actually does with
*evlerimizden* ("from our houses"):

```text
cl100k_base:  ev | ler | im | iz | den
o200k_base:   ev | ler | imiz | den

Real morphemes: ev (house) + ler (plural) + imiz (our) + den (from)
```

That is a near-perfect morphological split, found by pure frequency
counting with no grammar. The problem is not that BPE cannot find
Turkish structure; it is that with only 32k or 100k rows to spend,
most of them go to English, so Turkish gets split into *more, smaller*
correct pieces than English does. It is a budget problem, not a
competence problem.

The practical consequences are concrete. If you price an API call per
token, Turkish text costs more per unit of meaning. If you size a
context window, you fit less Turkish in it. And if you are choosing
between models for a Turkish workload, tokenizer fertility is a real
selection criterion — measure it on your own text with `tiktoken`
before you commit, the same way you would measure latency.

## 7. What the desk breaks

Four well-known model failures are tokenizer artifacts, not reasoning
failures. Knowing which is which tells you whether prompting can fix
them.

**Counting letters.** The model sees `["str","aw","berry"]`, three row
numbers. The letters are not in the input, so "how many r's" is a
question about data the model does not have. It answers from
statistical memory of similar questions, which is why it is
unreliable. *Fix:* make the characters visible — ask for
`s-t-r-a-w-b-e-r-r-y` first, or hand the job to code. No amount of
"think carefully" adds information that was discarded at the desk.

**Arithmetic on long numbers.** GPT-family pre-tokenizers cap digit
runs at three, so a number splits at boundaries that have nothing to
do with place value. Column-wise arithmetic on misaligned chunks is
genuinely hard. *Fix:* a calculator tool, not a better prompt.

**Glitch tokens.** In 2023 Jessica Rumbelow and Matthew Watkins found
tokens in the GPT-2/GPT-3 vocabulary — `SolidGoldMagikarp`,
`TheNitromeFan` — that made models produce bizarre output or refuse to
repeat them. The cause is a mismatch between two datasets: the strings
were frequent enough in the tokenizer's Reddit-derived training data
to earn a row, but were filtered out of the model's pre-training
corpus. Those embedding rows were therefore initialised and almost
never updated. Prompting one is reading a row of near-random numbers.
*Fix:* none at inference; it is a pipeline hygiene problem.

**Tokenizer tampering.** `tokenizer.json` ships alongside the weights
in a model repository and is loaded automatically at inference. It is
a config file, not a signed artifact. HiddenLayer and NVIDIA's AI Red
Team have both documented attacks where changing a *single* string
mapping — the token for `://`, say — silently alters every URL the
model emits, every argument a tool receives, every command a
downstream system runs. No weights are touched, so weight checksums
prove nothing. *Fix:* version and checksum the tokenizer as you would
the weights, and verify it at load time.

| Symptom | Real cause | What actually helps |
| :--- | :--- | :--- |
| Miscounts letters in a word | characters never entered the model | spell it out, or use code |
| Fails long multiplication | digit runs split at 3, breaking place value | a calculator tool |
| Bizarre output on an odd string | untrained embedding row | remove the token from inputs |
| Non-English costs 2× | merge budget spent on English | measure fertility; pick a bigger vocabulary |
| Output subtly rewritten | modified `tokenizer.json` | checksum the tokenizer file |

## 8. From stamp to vector

The desk hands the network a list of integers. Those integers cannot
go into a matrix multiply, for the reason from section 4: they are row
numbers, and arithmetic on row numbers is meaningless. ID 24912 minus
ID 15339 is not a quantity.

The bridge is the **embedding matrix** `E`, with shape `V × d` — one
row per vocabulary entry, each row a vector of `d` trainable numbers.
Tokenization's output is used as an index into it:

```text
token ID k  →  row k of E  →  a vector of d numbers
```

That is the whole operation: a lookup, not a computation. But the
rows are *learned*, so training pulls the rows for tokens that behave
alike toward each other, and the arbitrary integer becomes a position
in a meaningful space. That space is the subject of
[the embeddings article](post.html?slug=how-embeddings-work), which
picks up exactly here.

Two things happen before the transformer layers proper. Positional
information is added — the lookup is order-blind, so "dog bites man"
and "man bites dog" would otherwise be identical bags of rows — and
the result goes into attention. From there the story is
[how LLMs actually work](post.html?slug=how-llms-work).

It is worth seeing the full round trip in code, since padding and
attention masks are where the pipeline usually bites:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=True)

batch = tok(
    ["Tokenization is the desk before the model.",
     "BPE merges frequent pairs."],
    padding=True,        # pad the short one to match the long one
    truncation=True,
    max_length=16,
    return_tensors="pt",
)

print(batch["input_ids"])       # integer rows, padded
print(batch["attention_mask"])  # 1 = real token, 0 = padding

print(tok.decode(batch["input_ids"][0], skip_special_tokens=True))
```

The `attention_mask` is the part worth internalising. Padding exists
only to make a rectangular tensor out of sentences of different
lengths; the mask marks those positions `0` so attention ignores them.
Forget the mask and the model attends to filler.

A last word on **special tokens**. Alongside the learned pieces, every
vocabulary reserves rows for structure: `[PAD]` for the filler above,
`<bos>`/`[CLS]` to mark a sequence start, `<eos>`/`[SEP]` for the end
or a boundary, `[MASK]` for the blanks BERT-style training predicts.
They occupy ordinary rows and get ordinary embeddings — the model
learns what they mean like any other token. When a chat template
wraps your message in role markers, it is writing these tokens.

## The whole story in six lines

- A neural network cannot read text, so a tokenizer chops every prompt
  into pieces from a fixed list and replaces each with an integer.
- Words as tokens need too many rows and break on unseen input;
  characters need too few rows but make sequences too long. Subwords
  are the compromise: common text cheap, rare text expensive, nothing
  impossible.
- BPE builds the list by repeatedly merging the most frequent adjacent
  pair; WordPiece merges the most surprising pair instead, and Unigram
  prunes down from too many candidates.
- A token ID is a row number with no magnitude — and the row is an
  exact string, so `"hello"`, `"Hello"` and `" hello"` are three
  different rows.
- A bigger vocabulary costs parameters in two matrices and buys
  shorter sequences, which is why vocabularies grew from 32k to 256k
  as multilingual and code workloads mattered more.
- Letter-counting failures, digit arithmetic, glitch tokens and
  tokenizer tampering are all artifacts of this desk — so prompting
  fixes none of them.

## Glossary

The base vocabulary of the article, one line each:

- **tokenization** — splitting text into pieces from a fixed list and replacing each with an integer.
- **token** — one piece from that list; often a whole word, often a fragment, sometimes a single byte.
- **vocabulary** — the fixed list itself; its size `V` is chosen before training and fixed forever.
- **token ID** — the row number of a token in the vocabulary; arbitrary, with no magnitude or order.
- **BPE (byte pair encoding)** — building a vocabulary by repeatedly merging the most frequent adjacent pair.
- **byte-level BPE** — BPE whose atoms are the 256 byte values, so no input is ever unknown.
- **WordPiece** — BERT's variant; merges the pair that co-occurs most relative to chance.
- **Unigram** — starts with too many candidates and prunes the least useful ones.
- **SentencePiece** — a library running BPE or Unigram on raw text, treating the space as a character (`▁`).
- **pre-tokenizer** — the first split, usually on whitespace and punctuation, that stops merges crossing word boundaries.
- **OOV / `[UNK]`** — input the vocabulary cannot represent; eliminated by byte-level fallback.
- **fertility** — average tokens spent per word; the number that makes non-English text cost more.
- **agglutinative** — a language that builds words by stacking suffixes on a root, like Turkish or Finnish.
- **attention mask** — the 1/0 vector telling the model which positions are real tokens and which are padding.
- **embedding matrix** — the `V × d` table whose row `k` is the trainable vector for token ID `k`.
- **glitch token** — a token with a vocabulary row but almost no training signal, so its embedding stayed near-random.

## Going deeper

- Hugging Face, [Tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — the source for this article's BPE, WordPiece, Unigram and SentencePiece walkthroughs.
- Hugging Face, [LLM course, chapter 6](https://huggingface.co/learn/llm-course/chapter6/1) — training a tokenizer from scratch, step by step.
- Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (2015) — the paper that brought BPE to NLP.
- Kudo, [Subword Regularization](https://arxiv.org/abs/1804.10959) (2018) — the Unigram language model tokenizer.
- Kudo & Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226) (2018) — language-independent tokenization and the `▁` convention.
- OpenAI, [tiktoken](https://github.com/openai/tiktoken) — the library used for every measurement in sections 4 and 6; run it on your own text.
- Rumbelow & Watkins, [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) (2023) — the glitch-token discovery and its cause.
- HiddenLayer, [Tokenizer tampering](https://www.hiddenlayer.com/research/tokenizer-tampering) and NVIDIA AI Red Team, [Secure LLM tokenizers](https://developer.nvidia.com/blog/secure-llm-tokenizers-to-maintain-application-integrity/) — the attack surface in `tokenizer.json`.
- On this blog: [A Prompt's Journey (2): The Embedding Layer](post.html?slug=inside-the-embedding-layer) — discrete tokens to continuous geometry —, [A Prompt's Journey (3): Semantic Embeddings](post.html?slug=how-embeddings-work) — what happens to the integer after the lookup —, [A Prompt's Journey (4): Inside Self-Attention](post.html?slug=inside-self-attention) — the attention engine and context vectors —, [How LLMs actually work](post.html?slug=how-llms-work) — the layers that consume those vectors —, and [Optimizing LLM cost and latency](post.html?slug=optimizing-llm-cost-and-latency) — why token counts are the unit your bill is written in.
