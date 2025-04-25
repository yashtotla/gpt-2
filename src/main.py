from .gpt import GPT

def main():
    model = GPT.from_pretrained("gpt2")
    print("didn't crash yay!")

if __name__ == "__main__":
    main()
