"""Generate text from a trained phase 1 checkpoint.

Phase 2 was folded into phase 1 (see PLAN.md), so this is the only file
in phase2_training/ that does anything: it turns checkpoints/phase1.pt
back into a model and prints what it writes.

Meant to be imported from the repo root, like the rest of the code:

    from phase2_training.sample import sample
    print(sample("ROMEO:"))
"""

import torch

from phase1_baseline.model import GPT1
from phase1_baseline.tokenizer import RegexTokenizer
from phase1_baseline.train import CHECKPOINT_PATH, DEVICE, TOKENIZER_PATH


def sample(prompt="\n", max_new_tokens=300, temperature=1.0,
           checkpoint_path=CHECKPOINT_PATH, seed=None):
  """Load a checkpoint, continue `prompt`, and return the decoded text.

  Args:
    prompt:         text to start from. Must not be empty, because the
                    model needs at least one token to predict the next.
                    The default is a newline: a neutral starting point.
    max_new_tokens: how many tokens to add (about 3 characters each).
    temperature:    below 1 is more predictable, above 1 more random.
    seed:           set it to get the same text twice.

  The checkpoint must have been trained with the tokenizer saved at
  data/tokenizer.json: the weights are indexed by token id, and a
  different tokenizer would assign different ids.
  """
  if not prompt:
    raise ValueError("prompt must not be empty")
  if seed is not None:
    torch.manual_seed(seed)

  tokenizer = RegexTokenizer().load(TOKENIZER_PATH)

  # the state_dict holds weights but not the architecture, so rebuild the
  # model from the saved config first, then copy the weights in
  checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
  model = GPT1(**checkpoint["config"]).to(DEVICE)
  model.load_state_dict(checkpoint["model"])

  # encode() returns a flat list; the extra [] makes it (1, T) for the model
  idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
  out = model.generate(idx, max_new_tokens, temperature)

  return tokenizer.decode(out[0].tolist())
