import tiktoken
import torch
import torch.nn.functional as F
from .gpt import GPT
from .device import get_device


def main():
    model = GPT.from_pretrained("gpt2")
    print("didn't crash yay!")

    num_return_sequences = 5
    max_length = 30

    device = get_device()
    print(f"using device: {device}")

    # eval mode: good practice when not training
    # - layers like dropout/batchnorm behave differently at train vs eval time
    # - our model has none of those, so this may be a no-op here
    # - pytorch internals might still do something clever in eval mode, so keep it
    model.eval()
    # move all tensors/params to GPU - GPU is a separate computer optimised for
    # parallel workloads, running the net here is much faster than CPU
    model.to(device)

    # tiktoken: openai BPE tokenizer for gpt2
    # encode prefix string -> list of integer token ids
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    # unsqueeze(0): (T,) -> (1, T), then repeat to get (num_return_sequences, T)
    # all five rows start with the same prefix tokens
    x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.mps.manual_seed(42)

    # sampling loop: each iteration appends one new token column to x
    # x grows from (B, T_prefix) -> (B, max_length) one column at a time
    while x.size(1) < max_length:
        with torch.no_grad():
            # no_grad: not calling backward, so pytorch skips caching intermediate
            # tensors for the backward pass - saves memory and time
            logits = model(x)
            # only need logits at the last position - predicts token t+1
            # (wasteful to recompute all positions each step, but correct)
            logits = logits[:, -1, :]           # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # top-k sampling (k=50): HuggingFace pipeline default
            # - keep only top 50 most likely tokens, clamp rest to 0, renormalize
            # - prevents sampling very rare tokens that send the model off the rails
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)  # sample 1 token per row
            xcol = torch.gather(topk_indices, -1, ix)           # map back to vocab indices
            x = torch.cat((x, xcol), dim=1)                     # append new column

    # decode each completed sequence and print
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print("> ", decoded)

if __name__ == "__main__":
    main()
