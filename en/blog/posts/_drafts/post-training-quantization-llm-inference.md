> **Draft.** Scaffolded from my MSc thesis as a starting point. Edit before
> setting `"draft": false` in `posts.json`.

Serving a large language model is mostly a memory problem. Parameters have to
travel from HBM to the compute units for every token generated, and on modern
GPUs that transfer — not the arithmetic — is what you wait for. Autoregressive
decoding makes this worse: one token at a time, the full weight matrix moves for
each step, and arithmetic intensity stays low no matter how fast the tensor cores are.

Quantization attacks this directly. Store weights in fewer bits, move fewer
bytes, wait less.

## Why post-training, not quantization-aware

Quantization-aware training gives better accuracy at low bit widths, but it
needs the training pipeline, the data, and the compute to fine-tune. For a model
you did not train — which is most models most of the time — that is not on offer.

Post-training quantization (PTQ) works from a trained checkpoint and a small
calibration set. It is the practical option, and the interesting question is how
much accuracy it costs and where that cost comes from.

## Outliers are the whole problem

The reason naive PTQ falls apart on transformers is activation outliers. In
larger models a small number of feature dimensions carry values orders of
magnitude above the rest, and they appear consistently in the same channels.

Quantization maps a floating-point range onto a small integer grid. One extreme
value stretches that range, so every ordinary value collapses into a handful of
levels and the effective precision is destroyed. The average is fine; the tail
ruins it.

This is why per-tensor scaling underperforms and why the useful methods work at
finer granularity — per-channel or per-group scales — or move the difficulty
between weights and activations rather than absorbing it in one place.

## What actually matters in practice

- **Weights quantize far more easily than activations.** Weight-only schemes at
  4 bits are relatively forgiving. Activations at 8 bits are where careful work
  is needed.
- **Calibration data matters more than its quantity.** A few hundred samples
  that resemble the deployment distribution beat a large mismatched set.
- **Kernel support decides whether any of it helps.** A quantization scheme with
  no efficient kernel for your target architecture is a paper result. Memory
  saved is real; latency improvements depend on the dequantization path being
  fused rather than materialising a floating-point copy before the matmul.

## Measuring it honestly

Perplexity is a weak proxy. It moves little while task accuracy moves a lot,
which makes it comfortable to report and unwise to trust alone. Downstream
evaluation on the tasks you actually serve is the number that matters.

The same applies to speed. Report end-to-end latency at a realistic batch size
and sequence length, separating prefill from decode — they are bound by
different things, and quantization does not help them equally.

## Reference

Full method and results are in my MSc thesis at Bahçeşehir Üniversitesi,
*Post-Training Quantization for Efficient Inference of Large Language Models on
Modern GPU Architectures*, supervised by Asst. Prof. Fatih Kahraman.
