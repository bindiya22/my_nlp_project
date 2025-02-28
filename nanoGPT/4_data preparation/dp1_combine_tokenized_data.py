import os

# Directories
tokenized_directory = r"C:\Users\Bindiya\Documents\NLProject\tokenized_text"  # Directory containing tokenized text files
corpus = []  # This will hold all the tokenized words

# Load all tokenized files and combine the data
for file_name in os.listdir(tokenized_directory):
    if file_name.endswith('.txt'):
        file_path = os.path.join(tokenized_directory, file_name)
        
        # Read the tokenized data from each file
        with open(file_path, 'r', encoding='utf-8') as file:
            tokenized_data = file.read()
            words = tokenized_data.split()  # Tokenized words are space-separated
            corpus.extend(words)  # Add the words to the corpus

print(f"Total words in corpus: {len(corpus)}")
