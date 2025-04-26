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

    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.mps.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print("> ", decoded)

if __name__ == "__main__":
    main()
