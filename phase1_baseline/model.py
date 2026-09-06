"""Decoder-only Transformer assembly for the phase 1 GPT-1 baseline.

Components live in layers.py; this file stacks them into a language model.

Architecture notes:
  - post-LN, matching GPT-1's reference block. No final LayerNorm after the
    stack (each block already ends normalized), and training will likely
    need learning-rate warmup.
  - weights initialized at normal(std=0.02), biases at zero, as GPT-1 does.
    This pass is mandatory, not cosmetic: Decoder uses clones(), so without
    it every layer would start with identical weights.
  - the LM head is tied to the input embedding, per P(u) = softmax(h_n W_e^T)
    in the paper.
"""

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from .layers import *


def subsequent_mask(size):
  """Lower-triangular causal mask, True = may attend.

  Standalone utility. GPT1 does not call this - it caches an equivalent
  mask as a buffer so the tensor isn't rebuilt on every forward pass.
  """
  return torch.ones((size, size), dtype=torch.bool).tril()


class DecoderLayer(nn.Module):
  """One post-LN decoder block: masked self-attention, then feed-forward.

  Two sublayers, not three - there is no encoder to cross-attend to, which
  is the whole difference between this and the original encoder-decoder
  DecoderLayer.
  """

  def __init__(self, size, self_attn, feed_forward, dropout=0.1):
    super(DecoderLayer, self).__init__()
    self.size = size
    self.self_attn = self_attn
    self.feed_forward = feed_forward

    # 2 sublayer, masked self-attention and feed-forward
    self.sublayer = clones(SublayerConnection(size, dropout), 2)

  def forward(self, x, mask):
    """[batch_size, seq_len, d_model] in and out."""
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
    x = self.sublayer[1](x, lambda x: self.feed_forward(x))

    return x


class Decoder(nn.Module):
  """A stack of N identical DecoderLayers.

  No final LayerNorm - that belongs to pre-LN designs, where nothing else
  ever normalizes the residual stream. GPT-1 has no ln_f; GPT-2 added one
  exactly when it moved to pre-LN.
  """

  def __init__(self, N, d_model, h, d_ff, dropout=0.1):
    super(Decoder, self).__init__()
    self.layers = clones(DecoderLayer(d_model,
                                      MultiHeadAttention(d_model, h, dropout),
                                      PositionWiseFeedForward(d_model, d_ff, dropout), dropout),
                                      N)

  def forward(self, x, mask):
    """[batch_size, seq_len, d_model] in and out."""
    for layer in self.layers:
      x = layer(x, mask)

    return x


class GPT1(nn.Module):
  """Decoder-only Transformer language model.

  token ids -> embeddings + positional encoding -> N decoder blocks
            -> linear head -> logits over the vocabulary

  Args:
      vocab:   vocabulary size
      N:       number of decoder blocks
      d_model: residual stream width
      h:       number of attention heads (must divide d_model)
      d_ff:    feed-forward hidden width, conventionally 4 * d_model
      dropout: applied to attention weights, the FFN hidden layer, sublayer
               outputs, and the embedding sum
      max_len: maximum sequence length. Sizes the positional encoding and
               the causal mask, so mask memory grows quadratically in it.
  """

  def __init__(self, vocab, N, d_model, h, d_ff, dropout=0.1, max_len=5000):
    super(GPT1, self).__init__()
    self.embeddings = Embeddings(d_model, vocab)
    self.positional_embeddings = PositionalEncoding(d_model, dropout, max_len)
    self.decoder = Decoder(N, d_model, h, d_ff, dropout)

    # bias=False because the head is tied to the embedding matrix, and
    # GPT-1's output is a bare matmul against it
    self.head = nn.Linear(d_model, vocab, bias=False)

    # order matters: initialize everything first, then tie. Tying is a
    # rebind, so it must point at the already-initialized embedding.
    self.apply(self._init_weights)

    self.head.weight = self.embeddings.lut.weight

    # cached causal mask. A buffer so it follows the model to the GPU;
    # persistent=False so it stays out of checkpoints. Slicing works
    # because a causal mask is nested: the top-left T x T corner of a
    # bigger one is exactly the T x T mask.
    i = torch.arange(max_len)
    self.register_buffer("causal_mask", i[:, None] >= i[None, :], persistent=False)

  def _init_weights(self, module):
    """GPT-1's initialization: normal(std=0.02) weights, zero biases.

    Called once per submodule by nn.Module.apply, which handles the
    recursion. LayerNorm is deliberately skipped - its gain and bias are
    already correct at ones and zeros.
    """
    if isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0, std=0.02)

    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0, std=0.02)
      if module.bias is not None:
        nn.init.constant_(module.bias, 0)

  def forward(self, idx, targets=None):
    """
    Args:
        idx:     [batch_size, seq_len] token ids
        targets: [batch_size, seq_len] next-token ids, already shifted by
                 the dataloader. Do NOT shift again here.

    Returns:
        logits: [batch_size, seq_len, vocab]
        loss:   scalar, or None when targets is None
    """
    B, T = idx.shape

    assert T <= self.causal_mask.size(0), f"sequence length {T} exceeds max_len {self.causal_mask.size(0)}"

    embed = self.embeddings(idx)
    x = self.positional_embeddings(embed)

    mask = self.causal_mask[:T, :T]

    x = self.decoder(x, mask)

    logits = self.head(x)

    if targets is not None:
      # flatten batch and time together: every position is one independent
      # next-token classification. cross_entropy wants raw logits - it
      # applies log_softmax itself.
      return logits, cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, None
