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

def embed_documents(documents):
    """Create embeddings for documents"""
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return list(embedding_model.embed(documents, batch_size=64))

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

def create_faiss_db(folder_path, save_path=None, incremental=True):
    """Create or update FAISS database from documents in folder.

    Uses a hidden state file `.snapindex_processed.json` within the selected folder
    to keep track of already embedded files. On subsequent runs, only new files
    will be embedded and appended to the existing FAISS index.
    """

    # Discover all supported documents
    all_documents = find_documents(folder_path)
    if not all_documents:
        print("No supported documents found!")
        return None

    processed_paths = _load_processed_paths(folder_path) if incremental else set()

    if incremental:
        # Only process files whose full path is not yet recorded
        documents = [d for d in all_documents if d['path'] not in processed_paths]
        if not documents:
            print("✅ No new files to process. Database is up to date!")
            return save_path
        print(f"📄 Found {len(documents)} new files to process")
    else:
        # Full rebuild
        print("🔄 Full rebuild mode - processing all files")
        documents = all_documents
        processed_paths = set()
    
    # Extract text from documents
    all_texts = []
    all_metadata = []
    
    for doc in documents:
        print(f"Processing: {doc['name']} ({doc['type']})")
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
    
    if not all_texts:
        print("No text extracted from files")
        return save_path
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = embed_documents(all_texts)
    embeddings_array = np.array(embeddings, dtype='float32')
    
    # Handle database creation/update
    if save_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        save_path = tmp_dir.name
    else:
        os.makedirs(save_path, exist_ok=True)
    
    # Check if database already exists
    index_path = os.path.join(save_path, "index.faiss")
    metadata_path = os.path.join(save_path, "metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path) and incremental:
        # Load existing database
        print("📚 Loading existing database...")
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            existing_metadata = pickle.load(f)
        
        # Add new embeddings to existing index
        print(f"➕ Adding {len(embeddings_array)} new vectors to existing database")
        index.add(embeddings_array)
        
        # Combine metadata
        existing_metadata.extend(all_metadata)
        all_metadata = existing_metadata
        
        print(f"📊 Updated database now contains {index.ntotal} vectors")
    else:
        # Create new database
        print("🆕 Creating new database...")
        vector_size = len(embeddings[0])
        index = faiss.IndexFlatL2(vector_size)
        index.add(embeddings_array)
        print(f"📊 New database contains {index.ntotal} vectors")
    
    # Save updated database
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)
    
    # Update processed state file with newly processed document paths
    newly_processed = {d['path'] for d in documents}
    processed_paths.update(newly_processed)
    _save_processed_paths(folder_path, processed_paths)

    print(f"💾 Database saved to: {save_path}")
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
