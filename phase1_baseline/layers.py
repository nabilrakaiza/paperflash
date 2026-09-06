import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math
import copy

## Helper Functions
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

## scaled dot product attention
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
  """
  Computing the scaled dot-product attention.

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
  
  transposed_K = torch.transpose(key, -1, -2)
  scores = torch.matmul(query, transposed_K)
  scaled_scores =  scores / math.sqrt(d_k)

  if mask is not None:
    scaled_scores = scaled_scores.masked_fill(mask==0, torch.finfo(scaled_scores.dtype).min)

  attention_weights = softmax(scaled_scores, dim = -1)
  if dropout is not None:
    attention_weights = dropout(attention_weights)

  output = torch.matmul(attention_weights, value)

  return output, attention_weights


## Multi-Head Attention
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, h, dropout=0.1):
    super(MultiHeadAttention, self).__init__()
    assert d_model % h == 0, "d_model must be divisible by h"
    
    self.d_model = d_model
    self.h = h
    self.d_k = d_model // h # d_v = d_k
    
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
    """
    nbatches = query.size(0)
    
    # projected tensor, output shape of each: [batch_size, seq_len, d_model]
    q_projected = self.w_q(query)
    k_projected = self.w_k(key)
    v_projected = self.w_v(value)

    # divide model dimension to h heads
    q_heads = q_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    k_heads = k_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    v_heads = v_projected.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

    attn_output, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask, self.dropout)

    # combine it back
    concat_output = attn_output.transpose(1, 2).contiguous().view(nbatches, -1, self.d_model)
    
    return self.w_out(concat_output)

# Feed-Forward Network
class PositionWiseFeedForward(nn.Module):
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
    """
    
    return self.w_2(self.dropout(self.w_1(x).relu()))

# Embeddings
class Embeddings(nn.Module):
  def __init__(self, d_model, vocab):
    super(Embeddings, self).__init__()
    self.lut = nn.Embedding(vocab, d_model) # lut is lookup table
    self.d_model = d_model

  def forward(self, x):
    return self.lut(x) * math.sqrt(self.d_model)

# Positional Encoding
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    # sin fills the even columns, cos the odd ones, so they must be equal in number
    assert d_model % 2 == 0, "d_model must be even"
    self.dropout = nn.Dropout(p=dropout)
    
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    # changing the term a bit for numerical stability
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0)
    
    # Register pe as a buffer
    # persistent=False: pe is a fixed function of position, so recomputing it on
    # load is free and keeps it out of state_dict (10MB at d_model=512, max_len=5000)
    self.register_buffer('pe', pe, persistent=False)

  def forward(self, x):
    """
    Input shape:
        x: Input token embedding tensor of shape [batch_size, seq_len, d_model]
    """
    x = x + self.pe[:, : x.size(1)]
    
    return self.dropout(x)

# Sublayer Connection
class LayerNorm(nn.Module):
  "Construct a layernorm module (See citation for details)."

  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim = True)
    return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2

class SublayerConnection(nn.Module):
  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer_fn):
    """
    Args:
        x: Input tensor from the previous layer [batch_size, seq_len, d_model]
        sublayer_fn: A lambda function or module (like your Attention or FFN)
    """
    return self.norm(x + self.dropout(sublayer_fn(x)))