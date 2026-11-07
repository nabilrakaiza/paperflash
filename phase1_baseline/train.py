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
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "phase1.pt")

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
MAX_STEPS     = 5000  # ~62 epochs at ~81 steps/epoch. 500 under-trained
                      # (val 5.04, above bigram 4.81). Expect val to bottom
                      # out and rise late: the checkpoint is saved at the end
WARMUP_FRAC   = 0.05  # 0.05-0.10. post-LN needs this; too short and loss spikes

EVAL_INTERVAL = 250   # 20 eval points over the run
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

  Args:
    path:   where to write, e.g. "checkpoints/phase1.pt". The folder is
            created if needed. checkpoints/ is gitignored.
    config: dict of the arguments used to build the model, such as
            {"vocab": ..., "N": ..., "d_model": ...}. A state_dict holds
            weights but not the architecture, so this is the only record
            of which GPT1 to rebuild before loading.

  The optimizer state matters for resuming: it holds Adam's running
  averages of past gradients. Without it the loss visibly jumps on resume.

  The checkpoint is only meaningful together with data/tokenizer.json,
  because the weights are indexed by token id.
  """
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(folder, exist_ok=True)

  torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "config": config,
  }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
  """Restore from save_checkpoint. Returns the step to resume at.

  `model` must already be built with the same config the checkpoint was
  saved with; loading into a different shape raises an error.

  Pass optimizer and scheduler to resume training. Leave them out to load
  only the weights, e.g. to generate text from a trained model.

  map_location=DEVICE lets a checkpoint saved on Colab's GPU load on a
  machine without one, and the other way round.
  """
  checkpoint = torch.load(path, map_location=DEVICE)

  model.load_state_dict(checkpoint["model"])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
  if scheduler is not None:
    scheduler.load_state_dict(checkpoint["scheduler"])

  return checkpoint["step"]


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

  torch.manual_seed(SEED)

  train, val, tokenizer = load_data()
  x, y = get_batch(train, BLOCK_SIZE, BATCH_SIZE, DEVICE)

  model = GPT1(tokenizer.vocab_size, N, D_MODEL, H, D_FF, DROPOUT, BLOCK_SIZE)
  model.to(DEVICE)
  logits, loss = model(x, y)

  print(loss.item())
  print(math.log(tokenizer.vocab_size))

  optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

  for step in range(MAX_STEPS):
    x, y = get_batch(train, BLOCK_SIZE, BATCH_SIZE, DEVICE)
    logits, loss = model(x, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()

    if step % EVAL_INTERVAL == 0:
      result = estimate_loss(model, train, val, EVAL_ITERS)
      print(f"step {step:4d} | train {result['train']:.4f} | val {result['val']:.4f}")

  result = estimate_loss(model, train, val, EVAL_ITERS)
  print(f"step {MAX_STEPS:4d} | train {result['train']:.4f} | val {result['val']:.4f}  (final)")

  # keys match GPT1's argument names, so GPT1(**config) rebuilds the model
  config = dict(vocab=tokenizer.vocab_size, N=N, d_model=D_MODEL, h=H,
                d_ff=D_FF, dropout=DROPOUT, max_len=BLOCK_SIZE)
  save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, MAX_STEPS, config)