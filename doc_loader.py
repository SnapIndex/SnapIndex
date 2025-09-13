import os
from pypdf import PdfReader

def find_pdfs(folder_path):
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def extract_pdf_text(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    reader = PdfReader(pdf_path)
    all_text = []
    for i, page in enumerate(reader.pages, start = 1):
        text = page.extract_text() or ""
        all_text.append({
            "pdf_name": pdf_name,
            "page_number": i,
            "text": text.strip()
        })
    return all_text