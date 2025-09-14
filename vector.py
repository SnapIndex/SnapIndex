from fastembed import TextEmbedding
import tempfile
import shutil
import os
import numpy as np
import faiss
import pickle

def embed(documents):
    embedding_model = TextEmbedding(model_name = "BAAI/bge-small-en-v1.5")
    return list(embedding_model.embed(documents, batch_size = 64))

def create_db(text_dicts, embeddings):
    tmp_dir = tempfile.TemporaryDirectory()
    path = tmp_dir.name
    vector_size = len(embeddings[0])
    index = faiss.IndexFlatL2(vector_size)  # L2 distance
    index.add(np.array(embeddings, dtype='float32'))
    if not os.path.exists(path):
        os.makedirs(path)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "metadata.pkl"), "wb") as f:
        pickle.dump(text_dicts, f)
    return path, tmp_dir

def query_db(query_string, path):
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    embedding_model = TextEmbedding(model_name = "BAAI/bge-small-en-v1.5")
    query_vector = list(embedding_model.embed([query_string]))[0]
    distances, indices = index.search(query_vector, 5)
    for score, idx in zip(distances[0], indices[0]):
        meta = metadata[idx]
        print(f"Score: {score}, PDF: {meta['pdf_name']}, Page: {meta['page_number']}")
        print(meta["text"][:200], "...")

def delete_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    else:
        print(f"Directory does not exist: {path}")