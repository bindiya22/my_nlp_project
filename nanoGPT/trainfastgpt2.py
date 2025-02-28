import torch
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Set device to CPU (since no GPU is available)
device = 'cpu'

# Hyperparameters
batch_size = 8  # Increased batch size for faster training on CPU
block_size = 256  # Reduced sequence length for lower memory usage
initial_lr = 1e-3  # Start with 1e-3
mid_lr = 5e-4  # Reduce to 5e-4 after 500 iterations
final_lr = 1e-4  # Reduce to 1e-4 after 1000 iterations
max_iters = 1500
save_interval = 100
val_interval = 50  # Print validation loss every 50 iterations
warmup_steps = 100  # Gradual warmup to prevent instability

# Dataset path
DATASET_PATH = "C:\\Users\\Bindiya\\Documents\\NLProject\\nanoGPT\\data\\my_dataset"

def load_data():
    train_data = np.memmap(os.path.join(DATASET_PATH, 'train_new.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATASET_PATH, 'val_new.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data

train_data, val_data = load_data()

# Function to get training and validation batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=initial_lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_iters)

# Checkpoint directory
CHECKPOINT_DIR = "C:\\Users\\Bindiya\\Documents\\NLProject\\nanoGPT\\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

loss_values = []
val_loss_values = []

for iter in range(max_iters):
    xb, yb = get_batch('train')

    # Forward pass
    outputs = model(xb, labels=yb)
    loss = outputs.loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Adjust learning rate at specific iterations
    if iter == 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = mid_lr  # Reduce to 5e-4
    elif iter == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = final_lr  # Reduce to 1e-4

    # Print training loss
    if iter % 50 == 0:
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Iteration {iter}: Train Loss {loss.item():.4f}, Learning Rate: {current_lr:.6f}")

    loss_values.append(loss.item())

    # Validation every 50 iterations
    if iter % val_interval == 0:
        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch('val')
            val_outputs = model(xb_val, labels=yb_val)
            val_loss = val_outputs.loss.item()
            val_loss_values.append(val_loss)
            print(f"Iteration {iter}: Validation Loss {val_loss:.4f}")
        model.train()

    # Save model checkpoint every 100 iterations
    if iter % save_interval == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_checkpoint_{iter}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at iteration {iter}")

# Save final model
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
torch.save(model.state_dict(), final_model_path)
print("Final model saved!")

# Save loss values
np.save(os.path.join(CHECKPOINT_DIR, "loss_values.npy"), np.array(loss_values))
np.save(os.path.join(CHECKPOINT_DIR, "val_loss_values.npy"), np.array(val_loss_values))
print("Loss values saved!")
