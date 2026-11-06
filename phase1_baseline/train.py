"""Training loop for the phase 1 baseline — smoke test, not the real run.

Goal here is only to prove the model learns: a few hundred steps on
tiny-shakespeare, loss going down. The full run with a proper schedule and
resumable checkpoints belongs in phase2_training/.

Meant to be imported, not executed — the relative imports below only
resolve when this is loaded as part of the phase1_baseline package. From a
notebook at the repo root:

  from phase1_baseline.train import train
  train()

Build it in this order, testing each step before starting the next:
  1. load_data      -> decode a slice, confirm it reads as Shakespeare
  2. get_batch      -> confirm y is x shifted by exactly one token
  3. one forward    -> loss should be ~ln(vocab_size)
  4. naive loop     -> loss goes down. you now have a trainer.
  5. estimate_loss  -> watch train and val separate
  6. schedule + checkpoints
"""

import os
import math

import torch

from .model import GPT1
from .tokenizer import RegexTokenizer


# --------------------------------------------------------------- config

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer.json")

VOCAB_SIZE  = 2048    # BPE target, e.g. 1024-4096. tokenizer may end up smaller
BLOCK_SIZE  = 128    # context length, 128-256. must equal the model's max_len
BATCH_SIZE  = 32    # 32-64

N           = 6    # decoder blocks, 4-6
D_MODEL     = 256    # 128-384, must be divisible by h and even
H           = 4    # heads. 256/4 gives d_k=64, matching GPT-1
D_FF        = 1024    # conventionally 4 * D_MODEL
DROPOUT     = 0.1

LEARNING_RATE = 3e-4  # 3e-4 to 1e-3 at this scale. GPT-1 pretrained at 2.5e-4
WEIGHT_DECAY  = 0.01  # GPT-1's l2
GRAD_CLIP     = 1.0   # GPT-1's max_grad_norm
MAX_STEPS     = 500   # ~6 epochs at ~81 steps/epoch on this corpus
WARMUP_FRAC   = 0.05  # 0.05-0.10. post-LN needs this; too short and loss spikes

EVAL_INTERVAL = 50    # 10 eval points over the run, ~every half epoch
EVAL_ITERS    = 20    # 2 splits x 20 batches per eval, ~27% overhead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 1337


# ----------------------------------------------------------- step 1: data

def load_data():
  """Read the corpus, train (or load) the tokenizer, encode, split.

  Returns:
    train_data, val_data: 1-D torch.long tensors of token ids
    tokenizer:            the fitted RegexTokenizer
  """

  with open(DATA_PATH) as f:
    content = f.read()

  # both branches converge on a trained tokenizer. BPE training is slow, and
  # a retrained tokenizer can produce different ids, which would silently
  # invalidate any checkpoint trained against the old ones.
  if os.path.exists(TOKENIZER_PATH):
    tokenizer = RegexTokenizer().load(TOKENIZER_PATH)
  else:
    tokenizer = RegexTokenizer()
    # split on blank lines, not single newlines: chunks then still contain
    # "\n" characters, so BPE can learn merges involving them. splitting on
    # "\n" strips them entirely and costs ~15% compression on this corpus.
    tokenizer.train(content.split("\n\n"), VOCAB_SIZE)  # train() wants a LIST
    tokenizer.save(TOKENIZER_PATH)

  ids = tokenizer.encode(content)                       # encode() wants a STRING
  tensor = torch.tensor(ids, dtype=torch.long)

  # contiguous split, no shuffling: this is one continuous stream of text
  split = int(tensor.size(0) * 0.9)
  train_data = tensor[:split]
  val_data = tensor[split:]

  return train_data, val_data, tokenizer


# -------------------------------------------------------- step 2: batching

def get_batch(data, block_size, batch_size, device):
  """Sample a random batch of contexts and their next-token targets.

  Returns:
    x, y: both (batch_size, block_size) long tensors on `device`

  randint's high bound is exclusive, so len(data) - block_size gives a
  maximum offset whose y slice ends exactly at len(data).

  The +1 shift lives HERE. model.forward deliberately does not shift, so
  doing it in both places would train the model to predict two ahead.
  """

  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i: i + block_size] for i in ix])
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

  return x.to(device), y.to(device)


# ----------------------------------------------------------- step 5: eval

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters):
  """Average loss over several batches for each split.

  Returns:
    dict like {"train": float, "val": float}

  TODO:
    - model.eval() first, model.train() at the end. dropout is on during
      training and would make these numbers noisy and pessimistic.
    - average over eval_iters batches per split. a single batch is far
      too noisy to read a curve from.
  """

  model.eval()
  out = {}

  for split, data in [("train", train_data), ("val", val_data)]:
    temp = []
    for i in range(eval_iters):
      x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
      _, loss = model(x, y)
      temp.append(loss)

    avg_loss = sum(temp)/len(temp)
    out[split] = avg_loss

  model.train()

  return out


# ------------------------------------------------------- step 6: schedule

def lr_lambda(step):
  """Multiplier on LEARNING_RATE for a given step, for LambdaLR.

  Linear warmup then cosine decay to zero — GPT-1's pretraining schedule
  (paper 4.1), not the (1-x) linear decay in the released fine-tuning code.

  TODO:
    - warmup_steps = WARMUP_FRAC * MAX_STEPS
    - during warmup: ramp 0 -> 1 linearly
    - after: 0.5 * (1 + cos(pi * progress)) where progress runs 0 -> 1
      across the post-warmup steps
    - return a MULTIPLIER, not a learning rate. LambdaLR multiplies.
  """

  warmup = int(WARMUP_FRAC * MAX_STEPS)

  if step < warmup: return step / warmup 

  progress = (step - warmup) / (MAX_STEPS - warmup)

  return 0.5 * (1 + math.cos(math.pi * progress))
  
# ------------------------------------------------------------ checkpoints

def save_checkpoint(path, model, optimizer, scheduler, step, config):
  """Save everything needed to resume, not just the weights.

  TODO: torch.save a dict with model/optimizer/scheduler state_dicts,
  the step, and the config. Dropping the optimizer state loses Adam's
  moment estimates, and the loss visibly jumps on resume.
  """
  raise NotImplementedError


def load_checkpoint(path, model, optimizer, scheduler):
  """Restore from save_checkpoint. Returns the step to resume at."""
  raise NotImplementedError


# ------------------------------------------------------ steps 3-4: train

def train():
  """Run the smoke test.

  TODO, in order:
    3. build the model, get one batch, check loss ~ ln(vocab_size)
        before writing any loop. if that number is wrong, training will
        not fix it.
    4. the dumb loop: get_batch -> logits, loss = model(x, y) ->
        zero_grad(set_to_none=True) -> backward ->
        clip_grad_norm_(GRAD_CLIP) -> optimizer.step() -> scheduler.step()
    5. call estimate_loss every EVAL_INTERVAL steps and print both splits
    6. save a checkpoint at the end

  Notes:
    - max_len on the model must be >= BLOCK_SIZE
    - torch.manual_seed(SEED) so runs are comparable
    - AdamW, weight_decay=WEIGHT_DECAY
    - if loss spikes or goes NaN in the first few hundred steps, the
      warmup is too short. that is post-LN behaving as documented, not a
      bug in the model.
  """

  