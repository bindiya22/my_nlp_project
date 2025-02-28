from model import GPT, GPTConfig

# Define the configuration for a small GPT model
config = GPTConfig(
    vocab_size=50257,  # Vocabulary size
    block_size=128,    # Maximum sequence length
    n_layer=6,         # Number of transformer layers
    n_head=6,          # Number of attention heads
    n_embd=384         # Embedding size
)

# Initialize the GPT model
model = GPT(config)

print("Model initialized successfully!")
