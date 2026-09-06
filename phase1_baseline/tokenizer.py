## improved version
import json
import regex as re

class RegexTokenizer:
    def __init__(self):
        self.is_trained = False
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.GPT4_SPLIT_PATTERN = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

    def register_special_tokens(self, special_tokens: dict[str, int]):
        """Registers tokens like {'[PAD]': 16000, '[SOS]': 16001}"""
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    @property
    def vocab_size(self):
        """Number of embedding rows a model needs to cover every token id.

        This is max(id) + 1, not len(vocab): special tokens can be registered
        at ids well past the end of the merge vocabulary (e.g. '[PAD]': 16000),
        and the embedding table is indexed by id, so any gap still has to be
        allocated for.
        """
        highest = max(self.vocab) if self.vocab else -1
        if self.special_tokens:
            highest = max(highest, max(self.special_tokens.values()))
        return highest + 1

    def _get_stats(self, tokens_list: list[list[int]]):
        stats = {}
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                temp = tokens[i], tokens[i + 1]
                if temp not in stats:
                    stats[temp] = 0

                stats[temp] += 1

        return stats

    def _merge(self, tokens_list, pair_bytes, new_byte):
        new_tokens_list = []

        for tokens in tokens_list:
            n = len(tokens)
            i = 0

            new_tokens = []
            while i < n - 1:
                temp = tokens[i], tokens[i + 1]
                if temp == pair_bytes:
                    new_tokens.append(new_byte)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            if i == n-1:
                new_tokens.append(tokens[i])

            new_tokens_list.append(new_tokens)

        return new_tokens_list
    
    def _add_new_bytes(self, tokens_list, new_byte):
        pair_bytes_count = self._get_stats(tokens_list)
        
        if not pair_bytes_count:
            return None, None
        
        pair_bytes = max(pair_bytes_count, key=pair_bytes_count.get)

        if pair_bytes_count[pair_bytes] == 1:
            return None, None

        new_tokens_list = self._merge(tokens_list, pair_bytes, new_byte)
        return pair_bytes, new_tokens_list

    def _bpe(self, tokens_list: list[list[int]], current_size: int, vocab_size: int, verbose = False):
        merges = {}

        while current_size < vocab_size:
            pair_bytes, temp = self._add_new_bytes(tokens_list, current_size)

            if not temp:
                break

            if verbose:
                print(f"merging {pair_bytes} into a new token {current_size}")

            tokens_list = temp 
            merges[pair_bytes] = current_size
            current_size += 1

        vocab = {idx:bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        return tokens_list, merges, vocab

    def train(self, corpus, vocab_size, verbose=False):
        """Learn BPE merges from a corpus.

        Args:
            corpus: a LIST of strings (documents, lines, sentences) - not one
                    big string. Only the first 5000 entries are used.
            vocab_size: target size. Training stops early if no byte pair
                    repeats, so the result may be smaller.
        """
        # Iterating a str yields single characters, which produces a tokenizer
        # with zero merges and no error. Catch it here rather than let it pass.
        if isinstance(corpus, (str, bytes)):
            raise TypeError(
                "corpus must be a list of strings, not a single string - "
                "iterating a string yields characters. Try corpus.split('\\n')."
            )

        self.is_trained = True
        tokens_splits = []
        for i, text in enumerate(corpus):
            if i >= 5000:  # only use 5k sentences for BPE training
                break
            splits = re.findall(self.GPT4_SPLIT_PATTERN, text)
            tokens_splits.extend([list(map(int, s.encode("utf-8"))) for s in splits])

        new_tokens_list, merges, vocab = self._bpe(tokens_splits, 256, vocab_size, verbose)
        self.merges = merges
        self.vocab = vocab

    def encode(self, text, allowed_special="all"):
        """Text -> list of token ids. Requires a trained tokenizer."""
        if not self.is_trained:
            raise RuntimeError("tokenizer is not trained - call train() or load() first")
        
        if allowed_special == "all" and text in self.special_tokens:
            return [self.special_tokens[text]]
        
        splits = re.findall(self.GPT4_SPLIT_PATTERN, text)
        tokens_splits = [list(map(int, split.encode("utf-8"))) for split in splits]

        # Process merges on a per-chunk basis (Vastly faster inference)
        for i, tokens in enumerate(tokens_splits):
            while len(tokens) >= 2:
                stats = { (tokens[j], tokens[j+1]): j for j in range(len(tokens) - 1) }
                # Find pair that was merged earliest during training
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                # In-place chunk substitution optimization loop
                new_tokens = []
                k = 0
                while k < len(tokens) - 1:
                    if (tokens[k], tokens[k+1]) == pair:
                        new_tokens.append(idx)
                        k += 2
                    else:
                        new_tokens.append(tokens[k])
                        k += 1
                if k == len(tokens) - 1:
                    new_tokens.append(tokens[k])
                tokens = new_tokens
            tokens_splits[i] = tokens

        return [t for chunk in tokens_splits for t in chunk]

    def decode(self, ids):
        """List of token ids -> text. Requires a trained tokenizer."""
        if not self.is_trained:
            raise RuntimeError("tokenizer is not trained - call train() or load() first")
        
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                # Convert the special index back to its string format block representation safely
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {idx}")
                
        return b"".join(part_bytes).decode(encoding="utf-8", errors="replace")

    def save(self, path):
        """Write the trained tokenizer to a JSON file.

        Saves the merges, the special tokens, and the split pattern. The
        vocab is not saved - it is rebuilt from the merges on load, since
        it is fully determined by them.

        The split pattern is stored because encoding depends on it: the same
        merges applied after a different pre-split produce different token
        ids, which would silently invalidate any model checkpoint trained
        against this tokenizer.
        """
        if not self.is_trained:
            raise RuntimeError("cannot save an untrained tokenizer")

        payload = {
            "pattern": self.GPT4_SPLIT_PATTERN.pattern,
            # JSON has no tuple keys, so merges go out as [p0, p1, idx] triples
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, path):
        """Load a tokenizer written by save(). Replaces any current state."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        if payload["pattern"] != self.GPT4_SPLIT_PATTERN.pattern:
            raise ValueError(
                "saved split pattern does not match this tokenizer's pattern; "
                "encoding would not match the ids the model was trained on"
            )

        # dict order is insertion order, and encode() breaks ties by merge
        # index, so merges must come back in the order they were learned
        self.merges = {(p0, p1): idx for p0, p1, idx in payload["merges"]}

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        self.register_special_tokens(payload["special_tokens"])
        self.is_trained = True
        return self
