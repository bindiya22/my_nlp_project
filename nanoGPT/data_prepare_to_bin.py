# data_prepare.py
# This script processes the raw text data (train_data.txt and val_data.txt) from the specified 
# directory, tokenizes the content, and saves the tokenized data as binary files (train.bin and val.bin) 
# for use in training the NanoGPT model.
#
# This script is designed to work with Windows file paths.

import os
import pickle

# Set up the paths to your data (Windows-style paths with backslashes)
train_file_path = 'data\\my_dataset\\train_data.txt'
val_file_path = 'data\\my_dataset\\val_data.txt'

# Example tokenization function (you can replace this with your own tokenizer)
def tokenize(text):
    # Simple tokenization by splitting text into words (adjust to your needs)
    return text.split()

# Function to read the text file and tokenize it
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    tokens = tokenize(text)
    return tokens

# Process the train and validation files
train_tokens = process_file(train_file_path)
val_tokens = process_file(val_file_path)

# Save the tokenized data as binary files
train_bin_path = 'data\\my_dataset\\train.bin'
val_bin_path = 'data\\my_dataset\\val.bin'

with open(train_bin_path, 'wb') as train_bin_file:
    pickle.dump(train_tokens, train_bin_file)

with open(val_bin_path, 'wb') as val_bin_file:
    pickle.dump(val_tokens, val_bin_file)

print("Data preparation complete: train.bin and val.bin saved.")
