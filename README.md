# paperflash

A decoder-only Transformer built entirely from scratch — not to ship a product, but to understand *how* the GPT-1 architecture actually works, and how a small transformer learns to do arithmetic.

Every component here is implemented and explained, not just wired together. The deliverable is understanding, not a benchmark score.

## Why this exists

It's easy to import a pretrained model and call it done. This project goes the other way: build the architecture by hand, then use two focused experiments to answer a real question — *what does a transformer actually learn when it learns arithmetic, and what changes when the arithmetic needs more than one step?*

## What's in here

| Stage | What it does |
|---|---|
| **Baseline + training run** | GPT-1-style decoder-only Transformer from scratch — causal self-attention, post-LN, own BPE tokenizer, training loop — trained on Tiny Shakespeare to confirm it learns and generates |
| **Grokking / modular arithmetic** | A one-layer transformer trained on `(a + b) mod p`, tracked through the memorization → generalization transition, with an inspection of learned embeddings for the known Fourier-basis structure |
| **Multi-digit addition** | `a + b = c` on multi-digit numbers, trained from scratch with a digit-level tokenizer, comparing answer formats (most significant digit first, reversed, and with a scratchpad) by number length, including lengths longer than any seen in training |
| **Write-up** | Consolidated findings and plots |

GPT-2/3-era architecture changes (pre-LN, GELU, learned positional embeddings) retrained against the baseline were planned as a further stage and are left as future work.

## The core question

Modular addition and multi-digit addition both look like "adding numbers" from the outside, but a transformer doesn't solve them the same way:

- **Bounded arithmetic** (`a + b mod p`) is small enough that a model can learn an actual generalizable algorithm — grokking shows this directly: after continued training past memorization, test accuracy jumps, and the model's embeddings show real internal structure (a Fourier basis), not memorized lookup.
- **Multi-digit addition** is a different regime. The answer depends on carries that propagate from the least significant digit, a single forward pass only has so much computation to spend, and how the digits are tokenized and ordered decides whether each step is easy or hard. That's why answer format matters — writing the answer in the order the carries flow, or spelling out intermediate steps first, spreads the work across more generated tokens rather than making the model "smarter" in one pass. This is the same reason chain-of-thought helps.

This repo tries to make that distinction concrete rather than just cited — build the clean case (grokking), then measure the multi-step case against it.

The original plan used GSM8K grade-school word problems for the second experiment. A ~5M-parameter model trained on Tiny Shakespeare would score close to 0% there, and a 0% can't tell failures of language, reasoning and arithmetic apart, so multi-digit addition replaced it.

## Repo structure

```
paperflash/
├── phase1_baseline/     # GPT-1-style Transformer, tokenizer, training loop
├── phase2_training/     # sampling from the trained baseline (training merged into phase 1)
├── phase3_grokking/     # modular arithmetic + interpretability
├── phase4_math/         # multi-digit addition, answer formats, length generalization
├── phase5_deltas/       # future work: GPT-2/3-era architecture changes
├── phase6_writeup/      # findings, plots
├── data/                # tokenizer.json (committed); the corpus is downloaded
└── notebooks/           # Colab drivers — see notebooks/README.md for setup
```

## Findings

_(filled in as each stage completes — raw results in [`phase6_writeup/findings.md`](phase6_writeup/findings.md))_

- **Baseline** — A 5.3M-parameter post-LN GPT-1 trained for 5000 steps on Tiny Shakespeare reaches validation loss 3.96 nats/token (a bigram baseline gets 4.81). Its samples follow the play format, depend on the prompt — `KING HENRY:` brings in the *Henry VI* cast — and are not memorized: no generated line appears in the corpus.
- **Grokking** — TBD
- **Multi-digit addition** — TBD

## Setup

```bash
git clone https://github.com/nabilrakaiza/paperflash.git
cd paperflash
pip install -r requirements.txt
```

Training was run on free-tier Google Colab (T4 GPU) — everything here is intentionally small-scale, so no heavy compute is required to reproduce it. To train the baseline, open [`notebooks/phase1.ipynb`](notebooks/phase1.ipynb) in Colab; it downloads the corpus and saves the checkpoint to Google Drive.

## Acknowledgments

- The baseline follows [Radford et al., "Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1), checked against OpenAI's released code.
- The grokking experiment replicates [Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/abs/2201.02177) and follows the setup and Fourier analysis of [Nanda et al., "Progress measures for grokking via mechanistic interpretability"](https://arxiv.org/abs/2301.05217).
- The addition experiment builds on [Lee et al., "Teaching Arithmetic to Small Transformers"](https://arxiv.org/abs/2307.03381).
- Tiny Shakespeare is from [Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn).
