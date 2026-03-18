import argparse
import tiktoken
import torch
import torch.nn.functional as F
from .gpt import GPT, GPTConfig
from .device import get_device
from .data_loader import DataLoaderLite


def run_pretrained():
    # to train from scratch instead: model = GPT(GPTConfig())
    # - pytorch default-initialises all layers (e.g. xavier/kaiming for Linear)
    #   so no extra work needed - outputs will be random garbage until trained
    # - GPTConfig() defaults give the 124M param gpt2 (small) architecture
    model = GPT.from_pretrained("gpt2")
    print("didn't crash yay!")

    num_return_sequences = 5
    max_length = 30

    # auto-detect best available device (cuda > mps > cpu)
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
            ix = torch.multinomial(topk_probs, num_samples=1)   # sample 1 token per row
            xcol = torch.gather(topk_indices, -1, ix)           # map back to vocab indices
            x = torch.cat((x, xcol), dim=1)                     # append new column

    # decode each completed sequence and print
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print("> ", decoded)


def run_train():

    device = get_device()
    print(f"using device: {device}")

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.mps.manual_seed(1337)

    # small B=4, T=32 for debugging - just want a single cheap batch to verify the loop
    # production runs use much larger B and T (e.g. B=16, T=1024)
    train_loader = DataLoaderLite(B=4, T=32)

    # random init - GPTConfig() defaults to 124M param gpt2 (small) architecture
    model = GPT(GPTConfig())
    model.to(device)

    # AdamW: adam with decoupled weight decay (bugfix of adam, imo)
    # - keeps two buffers per param: m (first moment, like momentum) and v (second moment, like RMSProp)
    # - per-element gradient normalization -> much faster convergence than SGD for LLMs
    # - lr=3e-4: safe default for early debugging; will be scheduled properly later
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for iter in range(50):
        x, y = train_loader.next_batch()
        # data loader keeps tokens on CPU (avoids wasting GPU memory)
        # must ship each batch to device before forward pass
        # note: tensor.to(device) is NOT in-place, returns a new tensor on the target device
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()       # must zero grads: .backward() does += on gradients, not =
        logits, loss = model(x, y)
        loss.backward()             # backprop: deposit gradients into all .grad tensors
        optimizer.step()            # update params using AdamW rule
        # loss.item(): extracts single-element tensor to a python float
        # - behind the scenes: ships the 1-element tensor from GPU -> CPU, converts to float
        print(f"iteration {iter}, loss {loss.item()}")
    # overfitting a single batch: if we do NOT call next_batch (reuse same x, y), loss
    # should drop to ~0 - the transformer memorises that one batch perfectly
    # with fresh batches: loss drops but not to 0 in 50 steps
    # - easy early gains: driving logits of tokens that never occur in the dataset to -inf
    #   (e.g. exotic unicode, other languages) lowers loss without learning real patterns
    # - with B=4, T=32 on tiny shakespeare (~338k tokens): 1 epoch = ~2600 batches
    #   so 50 steps is not even close to 1 epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrained", "train"], required=True)
    args = parser.parse_args()

    if args.mode == "pretrained":
        run_pretrained()
    elif args.mode == "train":
        run_train()


if __name__ == "__main__":
    main()
