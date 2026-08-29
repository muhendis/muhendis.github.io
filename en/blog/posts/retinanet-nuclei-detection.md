> **Draft.** This note is scaffolded from the published paper as a starting point.
> Edit it before setting `"draft": false` in `posts.json`.

Counting cell nuclei in breast cancer histopathology slides is one of those tasks
that looks like a solved problem until you try it at scale. A single whole-slide
image can hold hundreds of thousands of nuclei. Pathologists do not count them all;
they sample. If detection is automated, the sampling bias goes away — but only if
the detector is reliable enough to be trusted.

## Why a one-stage detector

We used [RetinaNet](https://arxiv.org/abs/1708.02002) for nuclei detection. The
appeal of a one-stage detector here is not raw speed, it is the loss function.
Nuclei detection is an extreme foreground/background imbalance problem: the vast
majority of anchor boxes over a tissue patch contain no nucleus at all. Focal loss
is designed for exactly that imbalance, down-weighting the easy negatives that
would otherwise dominate the gradient.

## What mattered in practice

A few observations from the work:

- **Patch extraction strategy** affects results more than most architecture
  choices. Nuclei near patch borders are systematically harder, and how you handle
  overlap at inference time shows up directly in the counts.
- **Stain variation** between slides is real and it hurts. Normalizing to a
  reference tissue before training removes a source of variance that the model
  would otherwise have to spend capacity on.
- **Anchor scales** need to match the actual nucleus size distribution in your
  data. The COCO defaults are wrong for histopathology — nuclei occupy a much
  narrower and smaller size range than everyday objects.

## Reading the results honestly

Detection metrics on nuclei need care. A high mAP can hide systematic failure on
the clinically interesting cases — the pleomorphic, irregular nuclei that carry
diagnostic weight. Aggregate numbers are necessary but not sufficient; per-class
and per-region breakdowns tell you more about whether a model is usable.

## Reference

The full method and results are in the paper:

- *Nuclei Detection on Breast Cancer Histopathology Images Using RetinaNet*,
  SIU 2021. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9477852)

A companion paper covers pleomorphism scoring at the patch level:

- *Patch-Level Nuclear Pleomorphism Scoring Using Convolutional Neural Networks*,
  CAIP 2021. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-89128-2_18)
