import torch
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set device to CPU
device = 'cpu'

# Paths
DATASET_PATH = "C:\\Users\\Bindiya\\Documents\\NLProject\\nanoGPT\\data\\my_dataset"
CHECKPOINT_DIR = "C:\\Users\\Bindiya\\Documents\\NLProject\\nanoGPT\\checkpoints"
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_model.pt")

# Load validation dataset
val_data = np.memmap(os.path.join(DATASET_PATH, 'val_new.bin'), dtype=np.uint16, mode='r')

# Define batch parameters
batch_size = 8
block_size = 512

# Function to get validation batches
def get_val_batch():
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(val_data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(val_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# Load trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load model weights
model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Define loss function (same as in training)
loss_function = torch.nn.CrossEntropyLoss()

# Evaluate on validation set
def evaluate(model, val_data, num_batches=10):
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for _ in range(num_batches):
            xb, yb = get_val_batch()
            outputs = model(xb, labels=yb)
            loss = outputs.loss
            total_loss += loss.item()

    avg_val_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")

# Run evaluation
evaluate(model, val_data)
