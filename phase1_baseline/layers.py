"""Transformer building blocks for the phase 1 GPT-1 baseline.

Ported by hand from an earlier encoder-decoder implementation
(github.com/nabilrakaiza/transformer), keeping only what a decoder-only
model needs. Assembly lives in model.py; this file is components only.

Deliberately pre-GPT-2 throughout — sinusoidal positional encoding, ReLU,
post-LN sublayers, and the sqrt(d_model) embedding scale. Each of those is
phase 5's material, so don't "improve" them here.
"""

import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math
import copy


# ---------------------------------------------------------------- helpers

def clones(module, N):
  """Produce N deep copies of a module as a ModuleList.

  Note the copies share their *initial values* — deepcopy copies weights,
  not just structure. Whatever assembles these must run an init pass
  afterwards, or every layer starts identical.
  """
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ------------------------------------------------------------- attention

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
  """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

  Leading dimensions are treated as batch dimensions by matmul, so any
  rank works. In practice MultiHeadAttention calls this with 4-D tensors,
  after the heads have been split out.

  Input shape:
    query: (..., seq_len_q, d_k)   in practice (batch, h, seq_len, d_k)
    key:   (..., seq_len_k, d_k)
    value: (..., seq_len_k, d_v)
    mask:  broadcastable to (..., seq_len_q, seq_len_k)
           True / 1 = attend, 0 = block. Note this is the OPPOSITE of
           PyTorch's own boolean attn_mask, where True means masked out.

  Returns:
    output:            (..., seq_len_q, d_v)
    attention_weights: (..., seq_len_q, seq_len_k), post-dropout
  """
  d_k = query.size(-1)

  # scores[..., i, j] = dot(query_i, key_j): every query against every key.
  # Rows are queries, columns are keys — which is why the softmax and the
  # causal mask both act on the last dimension.
  transposed_K = torch.transpose(key, -1, -2)
  scores = torch.matmul(query, transposed_K)

  # sqrt(d_k) keeps the dot products from growing with dimension, which
  # would otherwise push softmax into its saturated, low-gradient regime.
  scaled_scores = scores / math.sqrt(d_k)

  if mask is not None:
    # finfo(dtype).min rather than -1e9: -1e9 overflows to -inf in fp16,
    # and a fully-masked row would then softmax to NaN.
    scaled_scores = scaled_scores.masked_fill(mask == 0, torch.finfo(scaled_scores.dtype).min)

  attention_weights = softmax(scaled_scores, dim=-1)
  if dropout is not None:
    attention_weights = dropout(attention_weights)

  output = torch.matmul(attention_weights, value)

  return output, attention_weights


class MultiHeadAttention(nn.Module):
  """h attention heads run in parallel over disjoint slices of d_model.

  Splitting d_model into h heads of d_k = d_model/h is what makes multi-head
  free: h heads cost exactly the same parameters and FLOPs as one full-width
  head. What you buy is h independent attention patterns per token instead
  of one averaged compromise.
  """

  def __init__(self, d_model, h, dropout=0.1):
    super(MultiHeadAttention, self).__init__()
    assert d_model % h == 0, "d_model must be divisible by h"

    self.d_model = d_model
    self.h = h
    self.d_k = d_model // h  # d_v = d_k

    # linear layers
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_out = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    """
    Input shape:
        query: tensor of shape [batch_size, seq_len, d_model]
        key:   tensor of shape [batch_size, seq_len, d_model]
        value: tensor of shape [batch_size, seq_len, d_model]
        mask:  broadcastable to [batch_size, h, seq_len, seq_len].
               [seq_len, seq_len] is enough for a causal mask, since it
               is the same for every batch element and every head.
               True / 1 = attend, 0 = block.

    Returns:
        tensor of shape [batch_size, seq_len, d_model]
    """
    nbatches = query.size(0)

    # projected tensor, output shape of each: [batch_size, seq_len, d_model]
    q_projected = self.w_q(query)
    k_projected = self.w_k(key)
    v_projected = self.w_v(value)

    # divide model dimension to h heads, then move h next to the batch
    # dimension so attention treats each head as an independent batch item:
    # [B, T, d_model] -> [B, T, h, d_k] -> [B, h, T, d_k]
    q_heads = q_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    k_heads = k_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    v_heads = v_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

    attn_output, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask, self.dropout)

    # combine it back. .contiguous() is required because transpose returns a
    # strided view, and view() cannot flatten across non-contiguous strides.
    concat_output = attn_output.transpose(1, 2).contiguous().view(nbatches, -1, self.d_model)

    # w_out lets the model mix the heads; without it each head would write
    # to a fixed, non-overlapping slice of the residual stream forever.
    return self.w_out(concat_output)


# ----------------------------------------------------------- feed-forward

class PositionWiseFeedForward(nn.Module):
  """Two linear layers applied independently at every position.

  "Position-wise" is the whole point: this layer cannot move information
  between tokens. All cross-token mixing happens in attention. At the
  conventional d_ff = 4 * d_model, this holds two thirds of a block's
  parameters.

  ReLU, not GELU — GELU is phase 5's material.
  """

  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionWiseFeedForward, self).__init__()

    # two linear layers
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    """
    Input shape:
        x: tensor of shape [batch_size, seq_len, d_model]

    Returns:
        tensor of the same shape
    """
    return self.w_2(self.dropout(self.w_1(x).relu()))


# ------------------------------------------------------------- embeddings

class Embeddings(nn.Module):
  """Token id -> dense vector, scaled by sqrt(d_model).

  The scaling comes from Vaswani section 3.4. It only lands in a sensible
  range if the embedding matrix is initialized small — with PyTorch's
  default N(0,1) it would put token embeddings ~40x above the positional
  signal. model.py's init pass (std=0.02) is what makes it work.

  GPT-2 drops the scaling entirely, so this is phase 5's material too.
  """

  def __init__(self, d_model, vocab):
    super(Embeddings, self).__init__()
    self.lut = nn.Embedding(vocab, d_model)  # lut is lookup table
    self.d_model = d_model

  def forward(self, x):
    """[batch_size, seq_len] of token ids -> [batch_size, seq_len, d_model]"""
    return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
  """Fixed sinusoidal position signal, added to the token embeddings.

  Each dimension is a sinusoid at a different frequency, which makes
  PE(pos + k) a fixed linear transform of PE(pos) for any offset k. That
  is what lets attention learn relative-position behaviour from absolute
  encodings.

  Nothing here is learned. GPT-1 uses learned positional embeddings
  instead — phase 5's material.
  """

  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    # sin fills the even columns, cos the odd ones, so they must be equal in number
    assert d_model % 2 == 0, "d_model must be even"
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    # changing the term a bit for numerical stability: exp(-2i*log(10000)/d)
    # instead of 1 / 10000**(2i/d), which loses precision at high frequencies
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)

    # Register pe as a buffer so it follows the model to the GPU.
    # persistent=False: pe is a fixed function of position, so recomputing it on
    # load is free and keeps it out of state_dict (10MB at d_model=512, max_len=5000)
    self.register_buffer('pe', pe, persistent=False)

  def forward(self, x):
    """
    Input shape:
        x: Input token embedding tensor of shape [batch_size, seq_len, d_model]

    Returns:
        tensor of the same shape, with position added and dropout applied
    """
    x = x + self.pe[:, : x.size(1)]

    return self.dropout(x)


# ------------------------------------------------ normalization / residual

class LayerNorm(nn.Module):
  """Layer normalization with a learnable gain (a_2) and bias (b_2).

  Uses the biased variance with eps inside the sqrt, matching nn.LayerNorm.
  (torch.std would apply Bessel's n-1 correction, and eps outside the sqrt
  under-damps near-constant activations by ~900x.)
  """

  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    """Normalizes over the last dimension. Shape is unchanged."""
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2


class SublayerConnection(nn.Module):
  """Residual connection around a sublayer, post-LN.

      out = LayerNorm(x + dropout(sublayer(x)))

  This is GPT-1's ordering: normalize *after* the residual add. GPT-2 moved
  the norm to the sublayer input (pre-LN), which is phase 5's headline
  delta. Two consequences of post-LN: no final LayerNorm is needed after
  the block stack, and training likely needs learning-rate warmup.

  Note dropout wraps only the sublayer output, never the residual — the
  identity path has to stay clean for gradients to reach early layers.
  """

  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer_fn):
    """
    Args:
        x: Input tensor from the previous layer [batch_size, seq_len, d_model]
        sublayer_fn: A lambda function or module (like your Attention or FFN)

    Returns:
        tensor of the same shape as x
    """
    return self.norm(x + self.dropout(sublayer_fn(x)))
