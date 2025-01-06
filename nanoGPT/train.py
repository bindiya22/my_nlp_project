import os
import math
import time
import numpy as np
import torch
from torch.nn import functional as F

# Configurations
config = {
    'batch_size': 16,
    'block_size': 128,
    'learning_rate': 1e-4,
    'eval_interval': 100,
    'eval_iters': 200,
    'max_iters': 3000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype': 'float16',  # Change to 'bfloat16' or 'float16' for mixed precision
    'compile': False,
    'out_dir': 'out',
    'data_dir': 'data/my_dataset',  # Path to the dataset
}

# Ensure output directory exists
os.makedirs(config['out_dir'], exist_ok=True)

# Define meta_vocab_size
meta_vocab_size = 50257  # Example value, adjust for your tokenizer or dataset

# Load train and validation data
train_data = np.memmap(os.path.join(config['data_dir'], 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(config['data_dir'], 'val.bin'), dtype=np.uint16, mode='r')

# Function to get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - config['block_size'], size=config['batch_size'])
    X = np.stack([data[i:i + config['block_size']] for i in ix])
    Y = np.stack([data[i + 1:i + 1 + config['block_size']] for i in ix])
    X = torch.tensor(X, dtype=torch.long, device=config['device'])
    Y = torch.tensor(Y, dtype=torch.long, device=config['device'])
    return X, Y

# Function to estimate loss
def estimate_loss():
    """Estimate the loss for training and validation splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            with torch.no_grad():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Simple GPT Model (example placeholder)
class SimpleGPT(torch.nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed = torch.nn.Embedding(vocab_size, block_size)
        self.lm_head = torch.nn.Linear(block_size, vocab_size)

    def forward(self, x, targets=None):
        x = self.embed(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

# Initialize model
model = SimpleGPT(vocab_size=meta_vocab_size, block_size=config['block_size'])
model.to(config['device'])
if config['compile']:
    model = torch.compile(model)  # For PyTorch 2.0+

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

# Training Loop
for iter in range(config['max_iters']):
    # Evaluate periodically
    if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    # Get batch and compute loss
    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss periodically
    if iter % config['eval_interval'] == 0:
        print(f"Iter {iter}: Loss {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join(config['out_dir'], 'model.pt'))
print("Model saved.")
