import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# List of stopwords in English
stop_words = set(stopwords.words('english'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Directories
input_directory = r"C:\Users\Bindiya\Documents\NLProject\extracted_text"  # Folder containing extracted text files
output_directory = r"C:\Users\Bindiya\Documents\NLProject\cleaned_text"  # Folder to save cleaned text files

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Function to clean the text
def clean_text(text):
    # Step 1: Remove unwanted characters (non-alphabetic)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Step 2: Lowercase the text
    text = text.lower()
    
    # Step 3: Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    
    # Step 4: Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    text = ' '.join(lemmatized_words)
    
    # Step 5: Remove excessive white spaces
    text = ' '.join(text.split())
    
    return text

# Process each text file in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_directory, file_name)

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()

        # Clean the text
        cleaned_text = clean_text(raw_text)

        # Save the cleaned text to the output directory
        output_file_path = os.path.join(output_directory, file_name)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(cleaned_text)

        print(f"Processed and cleaned {file_name}")
