import torch 

def make_dataset(p, train_frac, seed):
  X = torch.zeros(p**2, 3, dtype=torch.long)
  y = torch.zeros(p**2, dtype=torch.long)

  for a in range(p):
    for b in range(p):
      X[p*a + b][0] = a
      X[p*a + b][1] = b
      X[p*a + b][2] = p
      y[p*a + b] = (a + b) % p

  g = torch.Generator().manual_seed(seed)
  shuffled_indices = torch.randperm(X.size(0), generator=g)

  split_idx = int(train_frac * len(shuffled_indices))

  train_indices = shuffled_indices[:split_idx]
  test_indices = shuffled_indices[split_idx:]

  train_x, test_x = X[train_indices], X[test_indices]
  train_y, test_y = y[train_indices], y[test_indices]

  return train_x, train_y, test_x, test_y