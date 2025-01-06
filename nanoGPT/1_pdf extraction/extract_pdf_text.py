import os
import pdfplumber
import concurrent.futures

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""  # Add text of the page
            return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

# Function to process multiple PDFs concurrently
def process_pdfs(pdf_folder, output_folder):
    # Get all PDF files from the specified folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    # Path of PDFs
    pdf_paths = [os.path.join(pdf_folder, pdf_file) for pdf_file in pdf_files]
    
    # Start the concurrent execution of PDF text extraction
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_text_from_pdf, pdf_path): pdf_path for pdf_path in pdf_paths}
        
        # Track progress and results
        for future in concurrent.futures.as_completed(futures):
            pdf_path = futures[future]
            text = future.result()
            if text:
                print(f"Text extracted from {pdf_path}")
                
                # Save the extracted text to a text file in the output folder
                output_file = os.path.join(output_folder, os.path.basename(pdf_path) + ".txt")
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(text)
            else:
                print(f"Failed to extract text from {pdf_path}")

# Example usage
pdf_folder = r"C:\Users\Bindiya\Downloads\research_papers"  # Path of your PDFs
output_folder = r"C:\Users\Bindiya\Downloads\extracted_text"  # Folder to save text files

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

process_pdfs(pdf_folder, output_folder)
