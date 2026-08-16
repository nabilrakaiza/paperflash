# paperflash

A decoder-only Transformer built entirely from scratch — not to ship a product, but to understand *how* the GPT-1 → GPT-3 architecture actually works, and how a language model learns to do math.

Every component here is implemented and explained, not just wired together. The deliverable is understanding, not a benchmark score.

## Why this exists

It's easy to import a pretrained model and call it done. This project goes the other way: build the architecture by hand, then use two focused experiments to answer a real question — *what does a transformer actually learn when it learns arithmetic, and does that generalize to harder math?*

## What's in here

| Stage | What it does |
|---|---|
| **Baseline architecture** | GPT-1-style decoder-only Transformer from scratch — causal self-attention, post-LN, custom tokenizer, training loop |
| **Training run** | Trained on a public text corpus to confirm the baseline learns and generates coherently |
| **Grokking / modular arithmetic** | A tiny transformer trained on `(a + b) mod p`, tracked through the memorization → generalization transition, with an inspection of learned embeddings for the known Fourier-basis structure |
| **GSM8K / highschool math** | The same model family fine-tuned on grade-school word problems, evaluated and compared against the clean grokking result |
| **GPT-2/3-era deltas** | Pre-LN and other architecture changes implemented and retrained against the original baseline, to see whether architecture alone moves the needle or whether it's mostly scale |
| **Write-up** | Consolidated findings, plots, and a live demo |

## The core question

Grade-school word problems and bounded modular arithmetic both look like "math" from the outside, but a transformer doesn't solve them the same way:

- **Bounded arithmetic** (`a + b mod p`) is small enough that a model can learn an actual generalizable algorithm — grokking shows this directly: after continued training past memorization, test accuracy jumps, and the model's embeddings show real internal structure (a Fourier basis), not memorized lookup.
- **Multi-step word problems** are a different regime. Tokenization splits digits inconsistently, and a single forward pass only has so much computation to spend — which is why chain-of-thought helps: it spreads reasoning across more generated tokens rather than making the model "smarter" in one pass.

This repo tries to make that distinction concrete rather than just cited — build the clean case (grokking), then measure the messier case (GSM8K) against it.

## Repo structure

```
paperflash/
├── phase1_baseline/     # GPT-1-style Transformer, tokenizer, training loop
├── phase2_training/     # training run on a public corpus
├── phase3_grokking/     # modular arithmetic + interpretability
├── phase4_math/         # GSM8K fine-tuning + evaluation
├── phase5_deltas/       # GPT-2/3-era architecture changes, retrained
├── phase6_writeup/      # findings, plots, demo
└── notebooks/           # Colab drivers — see notebooks/README.md for setup
```

## Findings

_(filled in as each stage completes)_

- **Baseline** — TBD
- **Grokking** — TBD
- **GSM8K vs. grokking** — TBD
- **Architecture deltas vs. baseline** — TBD

## Setup

```bash
git clone https://github.com/<your-username>/paperflash.git
cd paperflash
pip install -r requirements.txt
```

Training was run on free-tier Google Colab (T4 GPU) — everything here is intentionally small-scale, so no heavy compute is required to reproduce it.

## Demo

A live demo of the trained model will be hosted on Hugging Face Spaces — link added once available.

## Acknowledgments

The grokking experiment replicates and builds on findings from [Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/abs/2201.02177) and Neel Nanda's mechanistic interpretability work on modular addition.
