from fastembed import TextEmbedding
import tempfile
import shutil
import os
import json
import hashlib
import platform
import ctypes
from datetime import datetime
import numpy as np
import faiss
import pickle
from file_loader import find_documents, extract_text_from_file, SUPPORTED_EXTENSIONS

def embed_documents(documents, progress_callback=None, parallel=True):
    """Create embeddings for documents with optional parallel processing"""
    if not documents:
        return []
    
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    if parallel and len(documents) > 5:
        # Use parallel processing for larger document sets
        return embed_documents_parallel(documents, embedding_model, progress_callback)
    else:
        # Use standard batch processing for smaller sets
        return list(embedding_model.embed(documents, batch_size=64))

def embed_documents_parallel(documents, embedding_model, progress_callback=None):
    """Create embeddings using optimized parallel processing"""
    import concurrent.futures
    import threading
    
    # For embedding, we'll use larger chunks but with more workers for better efficiency
    num_workers = min(6, len(documents))  # Use up to 6 workers for embeddings
    chunk_size = max(32, len(documents) // num_workers)  # Larger chunks for embeddings (32+ docs per chunk)
    document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
    
    # Progress tracking
    completed_chunks = 0
    total_chunks = len(document_chunks)
    progress_lock = threading.Lock()
    all_embeddings = []
    result_lock = threading.Lock()
    
    def process_chunk(chunk_index, chunk_docs):
        """Process a chunk of documents"""
        nonlocal completed_chunks, all_embeddings
        
        try:
            # Create embeddings for this chunk with larger batch size for efficiency
            embeddings = list(embedding_model.embed(chunk_docs, batch_size=128))
            
            # Update progress
            if progress_callback:
                with progress_lock:
                    completed_chunks += 1
                    progress_value = 0.7 + (0.15 * completed_chunks / total_chunks)  # 70% to 85%
                    progress_callback(progress_value, f"Creating embeddings... ({completed_chunks}/{total_chunks} chunks, {len(chunk_docs)} docs)")
            
            # Merge results thread-safely
            with result_lock:
                all_embeddings.extend(embeddings)
            
            return embeddings
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {e}")
            return []
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_chunk, i, chunk): i 
            for i, chunk in enumerate(document_chunks)
        }
        
        # Wait for all chunks to complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                future.result()  # Results are collected in the function
            except Exception as e:
                print(f"Chunk {chunk_index} generated an exception: {e}")
    
    return all_embeddings

def extract_text_parallel(documents, progress_callback=None):
    """Extract text from documents using true parallel processing (one document per thread)"""
    import concurrent.futures
    import threading
    
    # Progress tracking
    completed_docs = 0
    total_docs = len(documents)
    progress_lock = threading.Lock()
    all_texts = []
    all_metadata = []
    result_lock = threading.Lock()
    
    def process_single_document(doc):
        """Process a single document"""
        nonlocal completed_docs, all_texts, all_metadata
        
        doc_texts = []
        doc_metadata = []
        
        try:
            print(f"Processing: {doc['name']} ({doc['type']})")
            doc_data = extract_text_from_file(doc)
            
            for page_data in doc_data:
                if page_data["text"].strip():  # Only non-empty text
                    doc_texts.append(page_data["text"])
                    doc_metadata.append({
                        "pdf_name": page_data["file_name"],  # Keep same key for compatibility
                        "page_number": page_data["page_number"],
                        "text": page_data["text"],
                        "chunk_type": page_data.get("chunk_type", "content")
                    })
            
            # Update progress
            if progress_callback:
                with progress_lock:
                    completed_docs += 1
                    progress_value = 0.2 + (0.4 * completed_docs / total_docs)  # 20% to 60%
                    progress_callback(progress_value, f"Processing: {doc['name']} ({completed_docs}/{total_docs})")
            
        except Exception as e:
            print(f"Error processing document {doc['name']}: {e}")
        
        # Merge results thread-safely
        with result_lock:
            all_texts.extend(doc_texts)
            all_metadata.extend(doc_metadata)
        
        return doc_texts, doc_metadata
    
    # Process each document in parallel
    max_workers = min(8, len(documents))  # Use up to 8 workers for better parallelization
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each document individually for processing
        future_to_doc = {
            executor.submit(process_single_document, doc): doc 
            for doc in documents
        }
        
        # Wait for all documents to complete
        for future in concurrent.futures.as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                future.result()  # Just wait for completion, results are collected in the function
            except Exception as e:
                print(f"Document {doc['name']} generated an exception: {e}")
    
    return all_texts, all_metadata

def _processed_state_path(folder_path: str) -> str:
    return os.path.join(folder_path, ".snapindex_processed.json")

def _folder_hash(folder_path: str) -> str:
    return hashlib.sha1(os.path.abspath(folder_path).encode("utf-8")).hexdigest()[:16]

def _fallback_state_path(folder_path: str) -> str:
    base = os.path.join("./faiss_databases", "_state")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"state_{_folder_hash(folder_path)}.json")

def _load_processed_paths(folder_path: str) -> set:
    state_file = _processed_state_path(folder_path)
    # Try primary location first
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed_files", []))
        except Exception:
            pass
    # Fallback location
    fb = _fallback_state_path(folder_path)
    if os.path.exists(fb):
        try:
            with open(fb, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed_files", []))
        except Exception:
            pass
    return set()

def _save_processed_paths(folder_path: str, paths: set):
    state_file = _processed_state_path(folder_path)
    payload = {
        "processed_files": sorted(list(paths)),
        "updated_at": datetime.now().isoformat()
    }
    # First try writing to the selected folder (hidden file)
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if platform.system() == "Windows":
            try:
                ctypes.windll.kernel32.SetFileAttributesW(state_file, 0x02)  # FILE_ATTRIBUTE_HIDDEN
            except Exception:
                pass
        return
    except Exception as e:
        print(f"Warning: failed to write processed state file in folder: {e}")
    
    # Fallback to app-managed state directory inside project
    fb = _fallback_state_path(folder_path)
    try:
        with open(fb, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # No need to set hidden here; it's under our app state directory
    except Exception as e:
        print(f"Warning: failed to write fallback processed state file: {e}")

def create_faiss_db(folder_path, save_path=None, incremental=True, progress_callback=None):
    """Create or update FAISS database from documents in folder.

    Uses a hidden state file `.snapindex_processed.json` within the selected folder
    to keep track of already embedded files. On subsequent runs, only new files
    will be embedded and appended to the existing FAISS index.
    
    Args:
        folder_path: Path to folder containing documents
        save_path: Path to save the FAISS database
        incremental: Whether to do incremental update or full rebuild
        progress_callback: Function to call with progress updates (value, message)
    """

    # Discover all supported documents
    if progress_callback:
        progress_callback(0.1, "Scanning for documents...")
    
    all_documents = find_documents(folder_path)
    if not all_documents:
        print("No supported documents found!")
        if progress_callback:
            progress_callback(1.0, "No documents found!")
        return None

    processed_paths = _load_processed_paths(folder_path) if incremental else set()

    if incremental:
        # Only process files whose full path is not yet recorded
        documents = [d for d in all_documents if d['path'] not in processed_paths]
        if not documents:
            print("✅ No new files to process. Database is up to date!")
            if progress_callback:
                progress_callback(1.0, "Database is up to date!")
            return save_path
        print(f"📄 Found {len(documents)} new files to process")
        if progress_callback:
            progress_callback(0.2, f"Found {len(documents)} new files to process")
    else:
        # Full rebuild
        print("🔄 Full rebuild mode - processing all files")
        documents = all_documents
        processed_paths = set()
        if progress_callback:
            progress_callback(0.2, f"Processing {len(documents)} files")
    
    # Extract text from documents with parallel processing
    all_texts = []
    all_metadata = []
    
    if len(documents) > 2:  # Use parallel processing for 3+ documents
        all_texts, all_metadata = extract_text_parallel(documents, progress_callback)
    else:
        # Use sequential processing for smaller sets
        for i, doc in enumerate(documents):
            print(f"Processing: {doc['name']} ({doc['type']})")
            if progress_callback:
                progress_value = 0.2 + (0.4 * i / len(documents))  # 20% to 60%
                progress_callback(progress_value, f"Processing: {doc['name']}")
            
            doc_data = extract_text_from_file(doc)
            
            for page_data in doc_data:
                if page_data["text"].strip():  # Only non-empty text
                    all_texts.append(page_data["text"])
                    all_metadata.append({
                        "pdf_name": page_data["file_name"],  # Keep same key for compatibility
                        "page_number": page_data["page_number"],
                        "text": page_data["text"],
                        "chunk_type": page_data.get("chunk_type", "content")
                    })
    
    print(f"Extracted {len(all_texts)} text chunks from {len(documents)} files")
    if progress_callback:
        progress_callback(0.6, f"Extracted {len(all_texts)} text chunks")
    
    if not all_texts:
        print("No text extracted from files")
        return save_path
    
    # Create embeddings with parallel processing
    print("Creating embeddings with parallel processing...")
    if progress_callback:
        progress_callback(0.7, "Creating embeddings with parallel processing...")
    
    embeddings = embed_documents(all_texts, progress_callback=progress_callback, parallel=True)
    embeddings_array = np.array(embeddings, dtype='float32')
    
    if progress_callback:
        progress_callback(0.85, f"Created {len(embeddings)} embeddings")
    
    # Handle database creation/update
    if save_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        save_path = tmp_dir.name
    else:
        os.makedirs(save_path, exist_ok=True)
    
    if progress_callback:
        progress_callback(0.9, "Saving database...")
    
    # Check if database already exists
    index_path = os.path.join(save_path, "index.faiss")
    metadata_path = os.path.join(save_path, "metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path) and incremental:
        # Load existing database
        print("📚 Loading existing database...")
        if progress_callback:
            progress_callback(0.92, "Loading existing database...")
        
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            existing_metadata = pickle.load(f)
        
        # Add new embeddings to existing index
        print(f"➕ Adding {len(embeddings_array)} new vectors to existing database")
        if progress_callback:
            progress_callback(0.95, "Adding new vectors to database...")
        
        index.add(embeddings_array)
        
        # Combine metadata
        existing_metadata.extend(all_metadata)
        all_metadata = existing_metadata
        
        print(f"📊 Updated database now contains {index.ntotal} vectors")
    else:
        # Create new database
        print("🆕 Creating new database...")
        if progress_callback:
            progress_callback(0.92, "Creating new database...")
        
        vector_size = len(embeddings[0])
        index = faiss.IndexFlatL2(vector_size)
        index.add(embeddings_array)
        print(f"📊 New database contains {index.ntotal} vectors")
    
    # Save updated database
    if progress_callback:
        progress_callback(0.95, "Saving database files...")
    
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)
    
    # Update processed state file with newly processed document paths
    if progress_callback:
        progress_callback(0.98, "Updating file tracking...")
    
    newly_processed = {d['path'] for d in documents}
    processed_paths.update(newly_processed)
    _save_processed_paths(folder_path, processed_paths)

    print(f"💾 Database saved to: {save_path}")
    if progress_callback:
        progress_callback(1.0, "Database ready!")
    
    return save_path

def search_faiss_db(query_string, db_path, k=3):
    """Search FAISS database"""
    # Load index and metadata
    index = faiss.read_index(os.path.join(db_path, "index.faiss"))
    with open(os.path.join(db_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Create query embedding
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    query_vector = list(embedding_model.embed([query_string]))[0]
    
    # Search
    distances, indices = index.search(np.array([query_vector], dtype='float32'), k)
    
    # Print results
    print(f"\nSearch results for: '{query_string}'")
    print("=" * 50)
    
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        
        meta = metadata[idx]
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        
        print(f"\n{i+1}. Similarity: {similarity:.3f}")
        print(f"   PDF: {meta['pdf_name']}")
        print(f"   Page: {meta['page_number']}")
        print(f"   Text: {meta['text'][:200]}...")

def delete_faiss_db(db_path):
    """Delete FAISS database directory"""
    if os.path.exists(db_path) and os.path.isdir(db_path):
        shutil.rmtree(db_path)
        print(f"Deleted database: {db_path}")
    else:
        print(f"Database does not exist: {db_path}")

# Example usage
if __name__ == "__main__":
    # Create database from PDFs
    folder_path = "C:\\Users\\akarsh\\Downloads"  # Change this to your PDF folder
    db_path = create_faiss_db(folder_path, "./my_faiss_db")
    
    if db_path:
        # Search the database
        search_faiss_db("ipsum", db_path, k=3)
        search_faiss_db("Lorem ipsum dolor sit amet", db_path, k=3)
        
        # Clean up (optional)
        # delete_faiss_db(db_path)
