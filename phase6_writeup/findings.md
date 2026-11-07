# Findings

Raw results per phase, recorded as each one finishes. The README's
Findings section is written from this file in phase 6.

## Phase 1 + 2: GPT-1 baseline on Tiny Shakespeare

Run on 2026-11-07, Colab T4.

### Setup

| | |
|---|---|
| Architecture | decoder-only, post-LN (as GPT-1), sinusoidal positions, ReLU FFN, `sqrt(d_model)` embedding scale, LM head tied to the input embedding |
| Size | 6 layers, d_model 256, 4 heads (d_k 64), d_ff 1024; ~5.3M parameters |
| Tokenizer | own BPE, GPT-4 split pattern, 2048 tokens, trained on `"\n\n"`-separated chunks; 3.03 characters per token |
| Data | Tiny Shakespeare, 1,115,394 characters = 367,917 tokens; contiguous 90/10 split (331,125 / 36,792) |
| Training | 5000 steps, batch 32 x 128 tokens (~62 epochs), AdamW lr 3e-4, weight decay 0.01, grad clip 1.0, dropout 0.1 |
| Schedule | linear warmup over 250 steps (5%), cosine decay to 0 |
| Loss estimate | mean over 20 batches per split, dropout off |

### Loss

Initial loss 7.689 against ln(2048) = 7.625. The 0.06 gap matches the
prediction for small random logits: tied std-0.02 weights against a
normalized output give logit std of about 0.32, which raises the expected
loss by about 0.32^2 / 2 = 0.05.

| step | train | val |
|---|---|---|
| 0 | 7.690 | 7.681 |
| 250 | 5.522 | 5.496 |
| 500 | 4.593 | 4.623 |
| 750 | 4.244 | 4.386 |
| 1000 | 4.023 | 4.240 |
| 1500 | 3.718 | 4.073 |
| 2000 | 3.481 | 3.992 |
| 2500 | 3.285 | 3.981 |
| 3000 | 3.183 | 3.987 |
| 3500 | 3.051 | 3.942 |
| 4000 | 3.020 | 3.998 |
| 4500 | 3.005 | 3.967 |
| 4750 | 3.011 | 3.915 (best) |
| 5000 | 3.026 | 3.959 (final) |

Full log, every 250 steps:

```
step    0 | train 7.6895 | val 7.6810
step  250 | train 5.5218 | val 5.4964
step  500 | train 4.5930 | val 4.6228
step  750 | train 4.2438 | val 4.3859
step 1000 | train 4.0227 | val 4.2400
step 1250 | train 3.8506 | val 4.1123
step 1500 | train 3.7176 | val 4.0731
step 1750 | train 3.5975 | val 4.0104
step 2000 | train 3.4808 | val 3.9924
step 2250 | train 3.3749 | val 3.9850
step 2500 | train 3.2849 | val 3.9807
step 2750 | train 3.2255 | val 3.9344
step 3000 | train 3.1830 | val 3.9870
step 3250 | train 3.1253 | val 3.9202
step 3500 | train 3.0513 | val 3.9424
step 3750 | train 3.0534 | val 3.9243
step 4000 | train 3.0202 | val 3.9977
step 4250 | train 3.0187 | val 3.9465
step 4500 | train 3.0048 | val 3.9666
step 4750 | train 3.0105 | val 3.9148
step 5000 | train 3.0263 | val 3.9588  (final)
```

Reference points, computed on the same tokens:

| | val loss (nats/token) | nats/character |
|---|---|---|
| unigram (token frequencies only) | 6.374 | 2.103 |
| smoothed bigram (previous token only) | 4.809 | 1.586 |
| this model, 500-step CPU run | 5.037 | 1.661 |
| **this model, 5000 steps** | **3.959** | **1.306** |

For loose outside context only: Karpathy's character-level nanoGPT
reports 1.47 nats/character on this corpus (10.7M parameters). Tokenizer,
split and context length all differ, so it is not a like-for-like
comparison.

### Observations

- **Post-LN trained stably** with a 250-step warmup: no loss spikes, no NaNs.
- **The unigram plateau.** In the earlier 500-step CPU run, loss sat at
  6.36 for steps 50-100, matching the unigram baseline (6.37): the model
  had learned token frequencies but not yet to use context.
- **The 500-step run was under-trained**, ending above the bigram
  baseline, because the cosine schedule reached lr 0 while loss was still
  falling. That is why this run uses 5000 steps.
- **Overfitting without val rising.** Val improves until about step 2000,
  then wanders between 3.91 and 4.00 for the remaining 3000 steps while
  train keeps falling, 3.48 to 3.03. The generalization gap grows from
  0.5 to 0.9, but val does not get worse: dropout, weight decay and the
  decaying learning rate hold it. Saving at the end rather than at best
  val costs 0.04, within the noise of the estimate.
- **This is the ordinary version of the plateau grokking studies.** Train
  improves while held-out performance stays flat. In grokking, held-out
  performance eventually jumps; here it never does.

### Samples

Seed 0, 200 new tokens, temperature 1.0:

```
ROMEO: lack of robers
Belires chapluships: finger the one
That's upon a fellow, makes the account of
To the condemnable abrobation. Look'd him!
One that valour infant bids him do him speak;
And he hath thought held the other in the priest
Of his own meanness: so he is died; he knows me,
...

CAMILLO:
Dear many goodENES:
Her mother I pent his part,
Of under this deed is your visitation.

HERMIONE:
My lord,
```

```
KING HENRY: bear'st thy hand that be woo'd;
I'll leave this by Edward finger of France:
That Warwick take it not, therefore I am.

Son:
Nay, if thy lordWe come to Lady--to save thy way;
Methinks, Verona, Tullus! Can Ereen Margaret,
Doubted, content, mean as those hooks up.
My Lord of Gloucester, as this divine in blood,
...
Where I porized my brother's daughter,
Away with her good opinion!

GLOUCESTER:
```

- **It uses its context.** The two prompts give unrelated continuations
  with the same seed, and the `KING HENRY:` sample brings in the cast of
  *Henry VI*: Warwick, Edward, Margaret, Gloucester, France. The `ROMEO:`
  sample drifts to *The Winter's Tale* (Camillo, Hermione), not *Romeo and
  Juliet*.
- **It is not reciting.** None of the 30 generated lines appears in the
  corpus. The longest stretch it reuses is 23 characters
  (`My Lord of Gloucester, `, a stock phrase); on average the longest
  reused stretch is 13 characters of a 39-character line.
- **What it learned:** the play format (capitalized speaker names, verse
  line breaks, blank lines between speeches), mostly real words, and
  locally grammatical phrases. No meaning beyond a line or two.
- **BPE artifacts.** Invented words such as `chapluships`, `abrobation`,
  `porized` and `parears` are real subword pieces joined in combinations
  that do not exist; `goodENES`, `lordWe` and `FairLame` are the same at a
  word boundary. A character-level model would make different mistakes.
