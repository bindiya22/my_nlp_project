import os
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize
import json

# Make sure to download necessary NLTK resources
nltk.download('punkt')

# Directories
input_directory = r"C:\Users\Bindiya\Downloads\cleaned_text"  # Folder containing cleaned text files
output_directory = r"C:\Users\Bindiya\Downloads\tokenized_text"  # Folder to save tokenized text files

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Function to tokenize text
def tokenize_text(text):
    # Word Tokenization
    words = word_tokenize(text)
    
    # Sentence Tokenization (Optional)
    sentences = sent_tokenize(text)
    
    return words, sentences

# Process each text file in the cleaned text directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_directory, file_name)

        # Read the cleaned file
        with open(file_path, 'r', encoding='utf-8') as file:
            cleaned_text = file.read()

        # Tokenize the text
        words, sentences = tokenize_text(cleaned_text)

        # Save the tokenized words and sentences to the output directory
        output_file_path = os.path.join(output_directory, file_name)

        # Save the tokenized words (you can also save sentences if needed)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # Write the tokenized words as a JSON (or simply a plain text list)
            json.dump({'words': words, 'sentences': sentences}, output_file)

        print(f"Tokenized and saved {file_name}")
