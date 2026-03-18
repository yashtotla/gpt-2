import tiktoken
import torch

class DataLoaderLite:
    # "lite" - minimal CPU-only dataloader, good enough for early debugging
    # - tiny shakespeare: ~1M chars, ~338k tokens (gpt2 BPE ~3:1 compression)
    # - all ASCII, no exotic unicode
    # - tokens stay on CPU; caller ships each batch to GPU to avoid wasting VRAM
    # - walks the document in non-overlapping B*T chunks, wraps at end
    # - will be replaced with something more serious later (shuffling, sharding, etc.)
    def __init__(self, B, T):
        self.B = B
        self.T = T

        text = open("./dataset/input.txt", "r").read()
        # encode entire dataset into a flat 1D token tensor
        tokens = tiktoken.get_encoding("gpt2").encode(text)

        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # load B*T + 1 tokens: the +1 gives us the target for the very last position
        # without it, the final token in each row has no ground truth label
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        # x: inputs  - all tokens except the last, viewed as (B, T)
        # y: targets - all tokens except the first, viewed as (B, T)
        # at every position (b, t): x[b,t] predicts y[b,t] = x[b, t+1]
        # .view() is a zero-copy reshape: 1D sequence -> 2D batch of rows
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance by B*T so next call gets the next non-overlapping chunk
        self.current_position += B*T
        # wrap around to start of dataset when we run out of tokens
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y
