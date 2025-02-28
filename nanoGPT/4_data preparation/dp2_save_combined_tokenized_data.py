import os
import json

# Directories
input_directory = r"C:\Users\Bindiya\Documents\NLProject\tokenized_text"  # Folder containing tokenized text files
output_file = r"C:\Users\Bindiya\Documents\NLProject\combined_tokenized_data.json"  # Output file to store combined data

# Function to combine tokenized data
def combine_tokenized_data():
    combined_data = {'words': [], 'sentences': []}

    # Process each file in the tokenized text directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                combined_data['words'].extend(data['words'])
                combined_data['sentences'].extend(data['sentences'])

    # Save the combined data to a file
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(combined_data, output)

    print(f"Combined tokenized data saved to {output_file}")

# Run the function
combine_tokenized_data()
