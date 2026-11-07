# paperflash: revised plan (Nov 2026)

Decided in a planning chat on 2026-11-06. Where this conflicts with the phase
list in CLAUDE.md, this file wins until CLAUDE.md is updated (see the last
section).

## Timeline and priorities

- Budget: about 1 week, during the finals period in late Nov 2026.
- Hard stop: a separate project (cube RL post-training) starts in December.
- If time runs short, cut in this order: phase 6 polish, then phase 4.
  Do not shorten grokking (phase 3). A finished grokking result is the core
  of the project.

## Current state (local working tree, 2026-11-05)

- `phase1_baseline/layers.py`, `model.py`, `tokenizer.py`: done and committed.
- Tokenizer trained on Tiny Shakespeare: `data/tokenizer.json`, 1792 merges,
  so vocab is exactly 2048.
- `phase1_baseline/train.py` (uncommitted): `load_data`, `get_batch`,
  `estimate_loss`, `lr_lambda` are written. Still to do: the body of
  `train()`, `save_checkpoint`, `load_checkpoint`.
- Phases 2 to 6: empty stub files.

## Open issues from review (Nabil fixes these himself)

1. **Weight tying conflict.** CLAUDE.md says phase 1 has an untied output
   projection and lists weight tying as a phase 5 delta. `model.py` ties the
   head to the embedding, citing GPT-1's `softmax(h_n W_e^T)`, which does
   reuse `W_e`. Nabil to decide which is correct and update the other.
2. **`estimate_loss` returns tensors, not floats.** It appends `loss` itself,
   so the output dict holds 0-dim GPU tensors. Needs `.item()` per batch.
3. **`lr_lambda(0)` returns 0.** LambdaLR evaluates it at construction, so
   the first optimizer step runs at lr 0. Harmless with 25 warmup steps;
   `(step + 1) / warmup` avoids it.
4. **Sanity value for the first forward pass:** loss should be about
   ln(2048) ≈ 7.62. Post-LN output is normalized, and tied std-0.02 weights
   give logits with std around 0.3, so this should be close. A value like
   9 or 10 points at the embedding scale.

## Revised scope

### Phase 1 + 2, merged (about 1 day)

- Finish `train()` and the checkpoint functions, then run the smoke test on
  Tiny Shakespeare.
- Phase 2 is folded in: at this scale, "loss goes down and samples look
  vaguely Shakespearean" is the phase 2 result. No separate larger-corpus run.
- Nabil to decide what happens to the `phase2_training/` stubs (delete, or
  keep `sample.py` for generation).
- Done when: initial loss ≈ 7.62, train and val loss fall, samples are
  readable.

### Phase 3: grokking (about 2 to 3 days, top priority)

- Task: `(a + b) mod p` over all p² pairs, split into train and test by a
  fixed fraction.
- Reference setups to start from (verify against the papers; hyperparameter
  choices are Nabil's call per CLAUDE.md):
  - Power et al. 2022: p = 97, small transformer, AdamW with weight decay
    around 1.
  - Nanda et al. 2023 ("Progress measures for grokking via mechanistic
    interpretability"): p = 113, 1 layer, d_model 128, 4 heads, no
    LayerNorm, AdamW lr 1e-3, weight decay 1.0, full batch, 30% train
    fraction, tens of thousands of epochs.
- Decision for Nabil: reuse the `GPT1` class with a small config, or build a
  stripped-down 1-layer model closer to Nanda's. Either way, dropout should
  be 0 for this experiment.
- Log over training: train/test accuracy and loss (log-scale x axis), total
  weight norm.
- Analysis: Fourier transform of the token embedding matrix over the token
  index; check for a few sparse key frequencies.
- Expected divergences from the references (real variation, not bugs):
  - Grokking timing is sensitive to weight decay, train fraction and seed.
  - No grokking at all usually means weight decay too low or train fraction
    too high. Check those before hunting for bugs.
  - Post-LN `GPT1` differs from Nanda's no-LayerNorm model, so the Fourier
    structure may be less clean or appear in different places.
- Done when: a plot where train accuracy hits 100% early and test accuracy
  jumps much later, plus a Fourier plot of the embeddings.

### Phase 4: multi-digit addition (about 2 days, replaces GSM8K)

- Why GSM8K was dropped: a ~5M-param model pretrained on Tiny Shakespeare
  will score close to 0%, and a 0% cannot separate failure on language,
  reasoning or arithmetic. The addition experiment tests the same README
  claims (tokenization, compute per forward pass, why chain-of-thought
  helps) with clean, interpretable results.
- Task: `a+b=c` with synthetic numbers generated on the fly, trained from
  scratch with a digit-level tokenizer (not the BPE one).
- Conditions (answer format):
  - **Plain:** answer most significant digit first.
  - **Reversed:** answer least significant digit first, so the carry flows
    in generation order.
  - **Scratchpad:** write per-digit sums and carries before the answer.
  - Optional extra axis: digit tokens vs the BPE tokenizer.
- Eval: exact-match accuracy by number length, both in-distribution (for
  example trained on 1 to 5 digits) and for length generalization (6 to 8
  digits).
- Expected: reversed and scratchpad clearly beat plain; length
  generalization is poor for all three with sinusoidal absolute positions.
  Reference: Lee et al. 2023, "Teaching Arithmetic to Small Transformers".
- Nabil to decide file naming inside `phase4_math/` (`finetune.py` no longer
  fits).
- Done when: an accuracy plot or table of format x digit length.

### Phase 5: architecture deltas (cut, listed as future work)

### Phase 6: write-up (about 1 day)

- Fill in the README findings (currently TBD) and add 3 to 4 plots.
- One line explaining why GSM8K was replaced.
- List phase 5 as future work.
- Hugging Face Spaces demo only if time is left.

## CLAUDE.md updates needed

- Phase 4 description: GSM8K -> multi-digit addition experiment.
- Phase 5: deferred / future work.
- The weight tying line, once decided.
- "Current focus": finishing the phase 1 training loop.

## After this (separate project, not this repo)

December: an LLM post-training project to learn SFT, DPO, GRPO and PPO on a
single task, solving the 2x2 Rubik's cube with a small pretrained model
(Qwen around 0.5B, LoRA). A simulator gives the reward and BFS gives optimal
solutions for SFT data and an optimality metric, with a curriculum on
scramble depth. Order: SFT, DPO, GRPO, PPO as a stretch goal. DPO and GRPO
losses written from scratch and checked against TRL. Full plan to be written
separately.
