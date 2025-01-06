import json
from sklearn.model_selection import train_test_split

# Path to the combined tokenized data
combined_tokenized_data_file = r"C:\Users\Bindiya\Downloads\combined_tokenized_data.json"

# Load the combined tokenized data
with open(combined_tokenized_data_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
    corpus = data['words']  # Or use data['sentences'] if you prefer sentence-level data

# Split the data into training and validation sets
train_data, val_data = train_test_split(corpus, test_size=0.2, random_state=42)

# Save the split data to files
train_file_path = r"C:\Users\Bindiya\Downloads\train_data.txt"
val_file_path = r"C:\Users\Bindiya\Downloads\val_data.txt"

with open(train_file_path, 'w', encoding='utf-8') as train_file:
    train_file.write(" ".join(train_data))  # Joining words into a single string

with open(val_file_path, 'w', encoding='utf-8') as val_file:
    val_file.write(" ".join(val_data))  # Joining words into a single string

print(f"Training data saved to {train_file_path}")
print(f"Validation data saved to {val_file_path}")
