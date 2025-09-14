import os
import mimetypes
from pypdf import PdfReader
from docx import Document
import openpyxl
from pptx import Presentation
import csv

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    # PDF files
    '.pdf': 'pdf',
    
    # Microsoft Word documents
    '.docx': 'docx',
    '.doc': 'doc',
    
    # Microsoft Excel files
    '.xlsx': 'xlsx',
    '.xls': 'xls',
    '.csv': 'csv',
    
    # Microsoft PowerPoint presentations
    '.pptx': 'pptx',
    '.ppt': 'ppt',
    
    # Text files
    '.txt': 'txt',
    '.rtf': 'rtf',
    '.md': 'markdown',
    
    # Other text formats
    '.html': 'html',
    '.htm': 'html',
    '.xml': 'xml',
    '.json': 'json',
    '.log': 'log',
    '.ini': 'ini',
    '.cfg': 'cfg',
    '.conf': 'conf'
}

def find_documents(folder_path):
    """Find all supported document files in the folder"""
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                documents.append({
                    'path': os.path.join(root, file),
                    'name': file,
                    'extension': file_ext,
                    'type': SUPPORTED_EXTENSIONS[file_ext]
                })
    return documents

def extract_text_from_file(file_info):
    """Extract text from a single file based on its type"""
    file_path = file_info['path']
    file_name = file_info['name']
    file_type = file_info['type']
    
    all_text_chunks = []
    
    try:
        # First, add the file name as a searchable chunk
        file_name_chunk = {
            "file_name": file_name,
            "page_number": 0,  # Special page number for file name
            "text": f"Filename: {file_name}",
            "file_type": file_type,
            "chunk_type": "filename"
        }
        all_text_chunks.append(file_name_chunk)
        
        # Then extract content based on file type
        content_chunks = []
        if file_type == 'pdf':
            content_chunks = extract_pdf_text(file_path, file_name)
        elif file_type in ['docx', 'doc']:
            content_chunks = extract_docx_text(file_path, file_name)
        elif file_type in ['xlsx', 'xls']:
            content_chunks = extract_excel_text(file_path, file_name)
        elif file_type == 'csv':
            content_chunks = extract_csv_text(file_path, file_name)
        elif file_type in ['pptx', 'ppt']:
            content_chunks = extract_pptx_text(file_path, file_name)
        elif file_type in ['txt', 'rtf', 'markdown', 'html', 'xml', 'json', 'log', 'ini', 'cfg', 'conf']:
            content_chunks = extract_text_file(file_path, file_name)
        
        # Add chunk type to content chunks
        for chunk in content_chunks:
            chunk["chunk_type"] = "content"
            all_text_chunks.append(chunk)
            
        return all_text_chunks
        
    except Exception as e:
        print(f"Error extracting text from {file_name}: {e}")
        return all_text_chunks  # Return at least the filename chunk

def extract_pdf_text(pdf_path, file_name):
    """Extract text from PDF files"""
    reader = PdfReader(pdf_path)
    all_text = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        all_text.append({
            "file_name": file_name,
            "page_number": i,
            "text": text.strip(),
            "file_type": "pdf"
        })
    return all_text

def extract_docx_text(docx_path, file_name):
    """Extract text from Word documents"""
    doc = Document(docx_path)
    all_text = []
    
    # Extract text from paragraphs
    full_text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            full_text.append(paragraph.text.strip())
    
    # Extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                if cell.text.strip():
                    row_text.append(cell.text.strip())
            if row_text:
                full_text.append(" | ".join(row_text))
    
    # Combine all text
    combined_text = "\n".join(full_text)
    if combined_text.strip():
        all_text.append({
            "file_name": file_name,
            "page_number": 1,  # Word docs don't have pages like PDFs
            "text": combined_text,
            "file_type": "docx"
        })
    
    return all_text

def extract_excel_text(excel_path, file_name):
    """Extract text from Excel files"""
    workbook = openpyxl.load_workbook(excel_path, data_only=True)
    all_text = []
    
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        sheet_text = []
        
        for row in sheet.iter_rows():
            row_text = []
            for cell in row:
                if cell.value is not None:
                    row_text.append(str(cell.value).strip())
            if row_text:
                sheet_text.append(" | ".join(row_text))
        
        if sheet_text:
            combined_text = "\n".join(sheet_text)
            all_text.append({
                "file_name": f"{file_name} - {sheet_name}",
                "page_number": 1,
                "text": combined_text,
                "file_type": "excel"
            })
    
    return all_text

def extract_csv_text(csv_path, file_name):
    """Extract text from CSV files"""
    all_text = []
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as file:
        csv_reader = csv.reader(file)
        rows = []
        
        for row in csv_reader:
            if any(cell.strip() for cell in row):  # Skip empty rows
                rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
        
        if rows:
            combined_text = "\n".join(rows)
            all_text.append({
                "file_name": file_name,
                "page_number": 1,
                "text": combined_text,
                "file_type": "csv"
            })
    
    return all_text

def extract_pptx_text(pptx_path, file_name):
    """Extract text from PowerPoint presentations"""
    prs = Presentation(pptx_path)
    all_text = []
    
    for i, slide in enumerate(prs.slides, start=1):
        slide_text = []
        
        # Extract text from shapes
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())
        
        # Extract text from tables
        for shape in slide.shapes:
            if shape.has_table:
                table = shape.table
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        slide_text.append(" | ".join(row_text))
        
        if slide_text:
            combined_text = "\n".join(slide_text)
            all_text.append({
                "file_name": f"{file_name} - Slide {i}",
                "page_number": i,
                "text": combined_text,
                "file_type": "pptx"
            })
    
    return all_text

def extract_text_file(file_path, file_name):
    """Extract text from plain text files"""
    all_text = []
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read().strip()
                    if content:
                        all_text.append({
                            "file_name": file_name,
                            "page_number": 1,
                            "text": content,
                            "file_type": "text"
                        })
                        break
            except UnicodeDecodeError:
                continue
                
    except Exception as e:
        print(f"Error reading text file {file_name}: {e}")
    
    return all_text

def extract_all_documents(folder_path):
    """Extract text from all supported documents in a folder"""
    documents = find_documents(folder_path)
    all_extracted_text = []
    
    print(f"Found {len(documents)} supported documents")
    
    for doc in documents:
        print(f"Processing: {doc['name']} ({doc['type']})")
        extracted_text = extract_text_from_file(doc)
        all_extracted_text.extend(extracted_text)
    
    return all_extracted_text

# Backward compatibility functions
def find_pdfs(folder_path):
    """Backward compatibility - find only PDF files"""
    documents = find_documents(folder_path)
    return [doc['path'] for doc in documents if doc['type'] == 'pdf']

def extract_pdf_text_legacy(pdf_path):
    """Backward compatibility - extract text from PDF"""
    file_name = os.path.basename(pdf_path)
    return extract_pdf_text(pdf_path, file_name)