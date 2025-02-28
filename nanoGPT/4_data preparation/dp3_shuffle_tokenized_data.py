import os
import json
import random

# Directories
input_file = r"C:\Users\Bindiya\Documents\NLProject\combined_tokenized_data.json"  # Combined tokenized data
output_file = r"C:\Users\Bindiya\Documents\NLProject\shuffled_tokenized_data.json"  # Output file for shuffled data

# Load the combined tokenized data
with open(input_file, 'r', encoding='utf-8') as file:
    combined_data = json.load(file)

# Combine all words into a corpus
corpus = combined_data['words']

# Shuffle the corpus
random.shuffle(corpus)

# Save the shuffled corpus back into a new file
shuffled_data = {'words': corpus, 'sentences': combined_data['sentences']}

with open(output_file, 'w', encoding='utf-8') as output:
    json.dump(shuffled_data, output)

print(f"Shuffled tokenized data saved to {output_file}")
