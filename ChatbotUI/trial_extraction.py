import pdfplumber
import docx
import os
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text if text.strip() else "No readable text found in PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text.strip() else "No text found in the document."
    except Exception as e:
        return f"Error reading Word document: {e}"


# Test with your file paths
pdf_path = r"C:\Users\Bindiya\Downloads\btest.pdf"  # Change this to your PDF file path
docx_path = r"C:\Users\Bindiya\Downloads\btest.docx"  # Change this to your DOCX file path

print("PDF Extraction Result:\n", extract_text_from_pdf(pdf_path))
print("DOCX Extraction Result:\n", extract_text_from_docx(docx_path))
