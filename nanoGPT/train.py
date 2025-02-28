import torch
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device to CPU since you don't have a GPU
device = 'cpu'

# Hyperparameters for CPU training
batch_size = 4  # Further reduce batch size for CPU stability
block_size = 256  # Further reduce context length for memory efficiency
initial_learning_rate = 1e-3  # Initial learning rate
adjusted_learning_rate = 1e-4  # Adjusted learning rate
max_iters = 1000  # Adjust based on your needs

# Dataset path
DATASET_PATH = "C:\Users\Bindiya\Documents\NLProject\nanoGPT\data\my_dataset"

# Load data
def load_data():
    try:
        train_data = np.memmap(os.path.join(DATASET_PATH, 'train_new.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(DATASET_PATH, 'val_new.bin'), dtype=np.uint16, mode='r')
        return train_data, val_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

train_data, val_data = load_data()

# Function to get training and validation batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# Load pre-trained GPT-2 model and tokenizer
model_name = "distilgpt2"  # Use a smaller model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.to(device)

# Training loop
optimizer = AdamW(model.parameters(), lr=initial_learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-4)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    outputs = model(xb, labels=yb)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Step the scheduler with the current loss
    scheduler.step(loss)
    
    # Manually adjust the learning rate at specific iterations
    if iter == 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_learning_rate  # New learning rate
    
    if iter % 50 == 0:
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Iteration {iter}: Loss {loss.item():.4f}, Learning Rate: {current_lr:.6f}")

print("Training Complete!")