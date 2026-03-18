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

    # B=16, T=1024: realistic workload for benchmarking throughput
    # - max out batch size that fits in GPU; decrease if OOM
    # - use power-of-2 friendly sizes (8, 16, 24, 32, 48); NOT e.g. 17 (inefficient on GPU)
    # - tokens/sec is the objective metric, not ms/iter (batch size may change over time)
    train_loader = DataLoaderLite(B=16, T=1024)

    # random init - GPTConfig() defaults to 124M param gpt2 (small) architecture
    model = GPT(GPTConfig())
    model.to(device)

    # --- precision and tensor cores ---
    # - pytorch default dtype: fp32 for all params, activations, grads
    # - fp32 is overkill for DL: training tolerates much lower precision
    #
    # A100 80GB SXM peak throughput (without sparsity):
    # - fp32:       19.5 TFLOPS
    # - tf32:      156   TFLOPS  (8x)
    # - bf16/fp16: 312   TFLOPS  (16x)
    # - int8:      624   TFLOPS  (inference only; uniform spacing is a poor
    #              fit for normal-distributed weights/activations)
    #
    # tensor cores: hardware 4x4 matmul instruction on A100
    # - configurable input/accumulator/output precision
    # - all large matmuls (linear layers) decomposed into these 4x4 ops
    # - ref: A100 architecture white paper, figure 9
    #
    # memory bandwidth often the real bottleneck:
    # - most DL training is memory-bound, not compute-bound
    # - tensor cores idle waiting for data most of the time
    # - 60% utilization = excellent for a well-tuned workload
    # - lower precision = fewer bits per number = less memory + faster transfers
    #
    # TF32 (tensor float 32):
    # - mantissa truncated: 23 bits -> 10 bits (32 -> 19 total bits)
    # - internal to tensor core instruction; pytorch only sees fp32 in/out
    # - accumulator still fp32
    # - 8x matmul speedup, empirically indistinguishable from full fp32
    # - zero code changes, slightly more approximate: very good tradeoff
    # - observed on A100: 1000ms -> 333ms (~3x, not the theoretical 8x)
    #   still memory-bound: numbers in memory are still fp32, only the matmul op itself is faster
    torch.set_float32_matmul_precision('high')

    # --- bf16 mixed precision via autocast ---
    # bf16 vs fp16:
    # - bf16: same exponent (8 bits) as fp32, truncates mantissa only
    #   -> same range, less precision, NO gradient scaler needed
    # - fp16: reduced exponent -> reduced range -> NEEDS gradient scalers
    #   (extra state/complexity; fp16 came first on Volta, bf16 on Ampere simplified everything)
    #
    # torch.autocast usage:
    # - wrap forward pass + loss calculation ONLY
    # - do NOT wrap backward() or optimizer.step()
    # - do NOT manually cast tensors to bf16; let autocast decide
    #
    # mixed precision: params stay fp32, activations selectively become bf16
    # - matmuls -> bf16 (robust to precision changes)
    # - normalizations, softmax, layernorm, loss -> stay fp32 (more susceptible)
    #
    # observed on A100: 333ms -> 300ms, ~55k tok/s
    # - modest gain on top of TF32 because many other bottlenecks remain
    # - worth the tradeoff: slightly less accurate, can train longer to compensate

    # AdamW: adam with decoupled weight decay (bugfix of adam, imo)
    # - keeps two buffers per param: m (first moment, like momentum) and v (second moment, like RMSProp)
    # - per-element gradient normalization -> much faster convergence than SGD for LLMs
    # - lr=3e-4: safe default for early debugging; will be scheduled properly later
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for iter in range(50):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        # GPU timing: CPU schedules work on GPU asynchronously (queues kernels)
        # - without synchronize(), time.time() fires while GPU is still working
        # - must drain GPU work queue before measuring wall time
        torch.cuda.synchronize()
        # first iteration often slower: pytorch lazily inits gradient buffers etc.
        print(f"iteration {iter} | loss {loss.item():.4f}")
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
