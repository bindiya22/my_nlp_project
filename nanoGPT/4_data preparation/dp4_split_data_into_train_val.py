import json
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Path to the combined tokenized data
combined_tokenized_data_file = r"C:\Users\Bindiya\Documents\NLProject\combined_tokenized_data.json"

# Output directory for new train and val files
output_dir = r"C:\Users\Bindiya\Documents\NLProject\new_split"
os.makedirs(output_dir, exist_ok=True)

# File paths for new train, val, and vocab files
train_bin_path = os.path.join(output_dir, "train_new.bin")
val_bin_path = os.path.join(output_dir, "val_new.bin")
train_txt_path = os.path.join(output_dir, "train_new.txt")
val_txt_path = os.path.join(output_dir, "val_new.txt")
vocab_path = os.path.join(output_dir, "vocab_new.json")

# Load the combined tokenized data
with open(combined_tokenized_data_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
    corpus = data['words']  # Or use data['sentences'] if sentence-level data is preferred

# Extract the unique vocabulary
vocab = sorted(set(corpus))  # Sorted to ensure consistent indexing
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Create mappings for tokenization
stoi = {char: i for i, char in enumerate(vocab)}  # String to integer
itos = {i: char for char, i in stoi.items()}      # Integer to string

# Tokenize the corpus
encoded_data = np.array([stoi[word] for word in corpus], dtype=np.uint16)

# Split the data into training and validation sets
train_data, val_data = train_test_split(encoded_data, test_size=0.2, random_state=42)

# Save the split data to binary files
train_data.tofile(train_bin_path)
val_data.tofile(val_bin_path)

# Save the split data to text files
with open(train_txt_path, 'w', encoding='utf-8') as train_file:
    train_file.write(" ".join([itos[token] for token in train_data]))

with open(val_txt_path, 'w', encoding='utf-8') as val_file:
    val_file.write(" ".join([itos[token] for token in val_data]))

# Save the vocabulary to a JSON file
with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
    json.dump({'vocab': vocab, 'stoi': stoi, 'itos': itos}, vocab_file, ensure_ascii=False, indent=4)

# Summary of file paths
print(f"New training binary file saved to: {train_bin_path}")
print(f"New validation binary file saved to: {val_bin_path}")
print(f"New training text file saved to: {train_txt_path}")
print(f"New validation text file saved to: {val_txt_path}")
print(f"Vocabulary file saved to: {vocab_path}")
