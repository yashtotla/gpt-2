import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # gpt2 (small) defaults: 12 layers, 12 heads, 768 embd dim
    block_size: int = 1024   # max sequence length
    vocab_size: int = 50257  # BPE vocab: 50000 merges + 256 byte tokens + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    # - two linear projections sandwiched around GELU nonlinearity
    # - expands to 4x embd dim then projects back down
    # - "map" operation: each token processed independently, no cross-token communication
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU (approximate tanh version) used by gpt2 for historical reasons:
        # - erf (exact GELU) was slow in TensorFlow at the time, so tanh approx was adopted
        # - picked up by BERT, gpt2, etc. - now a historical quirk
        # - advantage over ReLU: no "dead neuron" problem - always contributes local gradient
        # - modern nets (e.g. LLaMA 3) use SwiGLU and other variants instead
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.APPLY_SCALING = True

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    # - "reduce" operation: tokens communicate and exchange information
    # - multi-head attention collapsed into one module via tensor gymnastics (efficient vs. separate head modules)
    # - NH treated as a batch dim alongside B so pytorch applies ops to all heads in parallel
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # single linear projects q, k, v for all heads at once
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.APPLY_SCALING = True
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal/autoregressive mask: tokens only attend to positions before them, never future
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim (n_embd)

        # compute q, k, v for all heads and split
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape: (B, T, C) -> (B, NH, T, HS) where HS = head size = C / NH
        # NH becomes a batch dim - pytorch will apply all ops across B and NH in parallel
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention scores: queries and keys interact multiplicatively ("how interesting do they find each other")
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # apply causal mask
        att = F.softmax(att, dim=-1)  # normalize so weights sum to 1

        # weighted sum of values: aggregate information from tokens found interesting
        y = att @ v
        # reassemble heads: transpose + view performs the concatenation of all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)  # output projection
        return y


class Block(nn.Module):
    # gpt2 modification vs. original transformer: "pre-norm" layout
    # - original: layernorm AFTER attention/MLP (inside residual stream)
    # - gpt2: layernorm BEFORE attention/MLP (pre-normalization)
    # - benefit: clean residual pathway from tokens to supervision
    #   - gradients from top flow straight back to inputs unchanged via residual
    #   - addition distributes gradients equally to both branches (see micrograd)
    #   - blocks still contribute their own gradient signal, but residual is clean
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)  # "reduce": cross-token communication
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)                   # "map": per-token computation / "thinking"

    def forward(self, x):
        # pre-norm: LN applied before each sub-layer, output added back to residual stream
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # transformer = repeated map-reduce: tokens communicate (attn), then think (mlp)
        # each block iteratively refines representations in the residual stream
        return x


class GPT(nn.Module):
    # naming mirrors HuggingFace GPT2LMHeadModel schema intentionally -
    # makes weight loading from HF state_dict trivial (keys match directly)
    def __init__(self, config):
        super().__init__()
        self.config = config

        # nn.ModuleDict: index submodules by string keys (like a dict)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings ("output embedding" in paper)
            wpe = nn.Embedding(config.block_size, config.n_embd),   # positional encodings
            # nn.ModuleList: index by integer (h.0, h.1, ... h.11 in HF schema)
            # nn.Embedding = glorified wrapper around a tensor, lookup by row index
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),  # extra final LN added by gpt2 (not in original transformer)
        ))

        # lm_head: projects 768 -> vocab_size (50257), no bias (gpt2 paper)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.init_weights)

    def init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if module.APPLY_SCALING:
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx: (B, T) token indices
        # - B: batch of independent sequences packed together for efficiency
        # - T: sequence length (up to block_size); each row is one sequence of T tokens
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # device=idx.device: pos must live on the same device as idx
        # - pytorch cannot combine tensors across devices (e.g. one on CPU, one on GPU)
        # - tying pos to idx.device means forward works regardless of which device the
        #   model and input were moved to - no manual device tracking needed at call sites
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emd = self.transformer.wpe(pos)   # (T, n_embd)
        tok_emd = self.transformer.wte(idx)   # (B, T, n_embd)

        # pos_emd is (T, n_embd) - same positions apply to every sequence in the batch
        # broadcasting adds an implicit B dim: (T, n_embd) -> (1, T, n_embd) -> (B, T, n_embd)
        x = tok_emd + pos_emd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # output: (B, T, vocab_size) logits
        # - at every (b, t) position: distribution over what token comes next (t+1)
        # - one softmax away from probabilities
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        # classmethod = alternative constructor: GPT.from_pretrained("gpt2") returns a GPT object
        # with weights copied from HuggingFace - no need to train from scratch
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        # hyperparams for all four gpt2 variants
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=24, n_embd=1280),  # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=32, n_embd=1600),  # 1558M params
        }[model_type]
        # vocab_size and block_size fixed across all gpt2 variants
        config_args['vocab_size'] = 50257  # 50000 BPE merges + 256 byte tokens + 1 <|endoftext|>
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)

        # init our model and grab its state dict
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # .attn.bias is our causal mask buffer, not a trainable param - skip it
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # load HF model and grab its state dict
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        # HF has two buffer keys to ignore: attn.masked_bias and attn.bias (both causal masks)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        # HF gpt2 comes from a TensorFlow checkpoint and uses Conv1D instead of nn.Linear
        # Conv1D stores weights transposed relative to what pytorch nn.Linear expects
        # must manually .t() these before copying
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # sanity check: our keys and HF keys should match 1:1 after filtering buffers
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # copy weights from HF into our model, transposing Conv1D weights where needed
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
