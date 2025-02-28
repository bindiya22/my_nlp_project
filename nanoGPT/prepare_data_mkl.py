import os
import pickle
import numpy as np
from collections import Counter

# Load the training and validation text datasets
with open('data/my_dataset/train_data.txt', 'r', encoding='utf-8') as f:
    train_text = f.read()

with open('data/my_dataset/val_data.txt', 'r', encoding='utf-8') as f:
    val_text = f.read()

# Combine both datasets (train and val) for tokenization
text = train_text + " " + val_text

# Tokenization - you can use a simple whitespace split or any other tokenizer
tokens = text.split()  # This is a basic tokenization; adjust as needed

# Build the vocabulary
vocab = Counter(tokens)
vocab_size = len(vocab)

# Create a mapping from tokens to IDs (and vice versa)
token_to_id = {token: idx for idx, (token, _) in enumerate(vocab.most_common())}
id_to_token = {idx: token for token, idx in token_to_id.items()}

# Encode the text into token IDs
encoded_text = [token_to_id[token] for token in tokens]

# Save the metadata as a pickle file (meta.pkl)
meta = {
    'vocab_size': vocab_size,
    'token_to_id': token_to_id,
    'id_to_token': id_to_token,
}

# Save the meta information in meta.pkl
os.makedirs('data/my_dataset', exist_ok=True)
with open('data/my_dataset/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# Encode the train and validation data into token IDs
train_tokens = [token_to_id[token] for token in train_text.split()]
val_tokens = [token_to_id[token] for token in val_text.split()]

# Convert to numpy arrays and save
train_data = np.array(train_tokens, dtype=np.uint16)
val_data = np.array(val_tokens, dtype=np.uint16)

# Save the tokenized data
train_data.tofile('data/my_dataset/train_new.bin')
val_data.tofile('data/my_dataset/val_new.bin')
