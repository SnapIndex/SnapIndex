import flet as ft
import sys
import os
import subprocess
import platform
import hashlib
from simple_faiss import create_faiss_db

"""This variant mirrors standalone_app.py behavior but uses a hardcoded folder path.
It builds a per-folder FAISS DB, performs semantic search across all documents,
deduplicates results per file, and allows clicking a result to open the file.
"""

# Hard-coded folder location (simulate context-opened folder)
selected_folder = "C:\\Users\\akarsh\\Downloads"

def _compute_folder_db_path(folder_path: str) -> str:
    abs_folder = os.path.abspath(folder_path)
    folder_hash = hashlib.sha1(abs_folder.encode("utf-8")).hexdigest()[:12]
    safe_name = os.path.basename(abs_folder) or "root"
    return os.path.join("./faiss_databases", f"{safe_name}_{folder_hash}")

def search_in_documents(query, folder_path):
    """Search using FAISS semantic search across all document types for this folder."""
    try:
        folder_db_path = _compute_folder_db_path(folder_path)
        # Create or update the DB
        if not os.path.exists(folder_db_path):
            print(f"Creating new FAISS database for folder: {folder_path}")
            create_faiss_db(folder_path, folder_db_path, incremental=False)
        else:
            print(f"Updating FAISS database incrementally for folder: {folder_path}")
            create_faiss_db(folder_path, folder_db_path, incremental=True)
        # Search and format
        return _get_faiss_results(query, folder_db_path, k=10, folder_path=folder_path)
    except Exception as e:
        print(f"Error in FAISS search: {e}")
        return []

def _get_faiss_results(query_string, db_path, k=10, folder_path=None):
    import faiss
    import numpy as np
    import pickle
    from fastembed import TextEmbedding

    index = faiss.read_index(os.path.join(db_path, "index.faiss"))
    with open(os.path.join(db_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    query_vector = list(embedding_model.embed([query_string]))[0]
    distances, indices = index.search(np.array([query_vector], dtype='float32'), k)

    results = []
    threshold = 0.3  # mirror standalone_app threshold
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        similarity = 1 / (1 + distance)
        if similarity < threshold:
            continue
        file_name = meta["pdf_name"]
        file_ext = os.path.splitext(file_name)[1].lower()
        file_type = file_ext[1:] if file_ext else "unknown"
        chunk_type = meta.get("chunk_type", "content")
        page_display = "Filename" if chunk_type == "filename" else f"Page {meta['page_number']}"
        results.append({
            "file": file_name,
            "page": page_display,
            "snippet": meta["text"][:300] + "..." if len(meta["text"]) > 300 else meta["text"],
            "full_path": folder_path or selected_folder,
            "similarity": similarity,
            "file_type": file_type,
            "chunk_type": chunk_type
        })
    # Dedup best per file
    best_by_file = {}
    for item in results:
        key = item["file"]
        if key not in best_by_file or item["similarity"] > best_by_file[key]["similarity"]:
            best_by_file[key] = item
    deduped = list(best_by_file.values())
    deduped.sort(key=lambda r: r.get("similarity", 0), reverse=True)
    return deduped

def open_file_location(file_path):
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", file_path], check=True)
        else:
            subprocess.run(["xdg-open", file_path], check=True)
    except Exception as e:
        print(f"Error opening file: {e}")

def main(page: ft.Page):
    page.title = "SnapIndex - Semantic Document Search"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1000
    page.window_height = 700
    page.padding = 20

    if not os.path.exists(selected_folder):
        page.add(ft.Text(f"Folder does not exist: {selected_folder}", color="red"))
        page.add(ft.Text("Please check the folder path in the code.", color="gray"))
        return

    # Create search components
    search_field = ft.TextField(
        label="Search All Documents",
        hint_text="Search across PDFs, Word docs, Excel, PowerPoint, and text files...",
        expand=True,
        on_submit=lambda e: perform_search(e, search_field, results_container, selected_folder)
    )
    
    search_button = ft.ElevatedButton(
        "Search",
        on_click=lambda e: perform_search(e, search_field, results_container, selected_folder)
    )
    
    results_container = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        expand=True
    )

    def perform_search(e, search_field, results_container, folder_path):
        query = search_field.value.strip()
        if not query:
            results_container.controls.clear()
            results_container.controls.append(
                ft.Text("Please enter a search query.", color="gray", italic=True)
            )
            page.update()
            return
        
        # Clear previous results
        results_container.controls.clear()
        results_container.controls.append(
            ft.Text(f"Searching for: '{query}'", weight=ft.FontWeight.BOLD, size=16)
        )
        results_container.controls.append(ft.Divider())
        page.update()
        
        # Perform search (all document types)
        results = search_in_documents(query, folder_path)
        
        # Clear and show results
        results_container.controls.clear()
        results_container.controls.append(
            ft.Text(f"Search Results for: '{query}'", weight=ft.FontWeight.BOLD, size=16)
        )
        results_container.controls.append(
            ft.Text(f"Found {len(results)} matches", color="blue", italic=True)
        )
        results_container.controls.append(ft.Divider())
        
        if results:
            for i, result in enumerate(results, 1):
                # Create result card
                similarity_score = result.get('similarity', 0)
                def on_card_click(res=result):
                    file_name = res['file']
                    folder = res['full_path']
                    full_file_path = os.path.join(folder, file_name)
                    open_file_location(full_file_path)

                result_card = ft.Card(
                    content=ft.GestureDetector(
                        content=ft.Container(
                            content=ft.Column([
                                ft.Row([
                                    ft.Icon(
                                        ft.Icons.TITLE if result.get('chunk_type') == 'filename' else ft.Icons.DESCRIPTION,
                                        color="orange" if result.get('chunk_type') == 'filename' else "blue"
                                    ),
                                    ft.Text(
                                        f"{result['file']} ({result.get('file_type','UNKNOWN').upper()}) - {result['page']}",
                                        weight=ft.FontWeight.BOLD,
                                        color="blue"
                                    ),
                                    ft.Text(
                                        f"Similarity: {similarity_score:.3f}",
                                        size=12,
                                        color="green",
                                        weight=ft.FontWeight.BOLD
                                    )
                                ]),
                                ft.Text(
                                    result['snippet'],
                                    size=12,
                                    color="gray"
                                ),
                                ft.Text(
                                    f"Folder: {result['full_path']}",
                                    size=10,
                                    color="gray",
                                    italic=True
                                )
                            ], tight=True),
                            padding=15
                        ),
                        on_tap=lambda e, r=result: on_card_click(r)
                    ),
                    margin=ft.margin.only(bottom=10)
                )
                results_container.controls.append(result_card)
        else:
            results_container.controls.append(
                ft.Text("No matches found.", color="gray", italic=True)
            )
        
        page.update()

    # Create the main layout
    page.add(
        ft.Column([
            # Header
            ft.Row([
                ft.Text("SnapIndex Semantic Document Search", size=24, weight=ft.FontWeight.BOLD)
            ]),
            ft.Text(f"Searching in: {selected_folder} (Per-folder FAISS, filename+content)", color="gray"),
            ft.Divider(),
            
            # Search section
            ft.Row([
                search_field,
                search_button
            ]),
            ft.Divider(),
            
            # Results section
            ft.Container(
                content=results_container,
                expand=True,
                padding=10
            )
        ], expand=True)
    )

ft.app(target=main)
