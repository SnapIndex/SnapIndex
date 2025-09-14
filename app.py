import flet as ft
import sys
import os
from simple_faiss import create_faiss_db, search_faiss_db, delete_faiss_db

# Hard-coded folder location
selected_folder = "C:\\Users\\akarsh\\Downloads"
db_path = "./faiss_database"

def search_in_pdfs(query, folder_path):
    """Search for query text using FAISS semantic search"""
    try:
        # Check if database exists, if not create it
        if not os.path.exists(db_path):
            print("Creating FAISS database...")
            create_faiss_db(folder_path, db_path)
        
        # Use the search_faiss_db function from simple_faiss.py
        # and capture the results in the expected format
        return _get_faiss_results(query, db_path, k=10)
        
    except Exception as e:
        print(f"Error in FAISS search: {e}")
        return []

def _get_faiss_results(query_string, db_path, k=10):
    """Get FAISS search results using functions from simple_faiss.py"""
    import faiss
    import numpy as np
    import pickle
    from fastembed import TextEmbedding
    
    # Load index and metadata (same logic as in simple_faiss.py)
    index = faiss.read_index(os.path.join(db_path, "index.faiss"))
    with open(os.path.join(db_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Create query embedding (same logic as in simple_faiss.py)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    query_vector = list(embedding_model.embed([query_string]))[0]
    
    # Search (same logic as in simple_faiss.py)
    distances, indices = index.search(np.array([query_vector], dtype='float32'), k)
    
    # Format results to match the original app.py structure
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        
        meta = metadata[idx]
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        
        results.append({
            "file": meta["pdf_name"],
            "page": meta["page_number"],
            "snippet": meta["text"][:300] + "..." if len(meta["text"]) > 300 else meta["text"],
            "full_path": selected_folder,  # Use the global selected_folder
            "similarity": similarity
        })
    
    return results

def main(page: ft.Page):
    page.title = "SnapIndex - PDF Search"
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
        label="Search in PDFs",
        hint_text="Enter your search query...",
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
        
        # Perform search
        results = search_in_pdfs(query, folder_path)
        
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
                result_card = ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text(
                                    f"{result['file']} - Page {result['page']}",
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
                                f"File: {result['full_path']}",
                                size=10,
                                color="gray",
                                italic=True
                            )
                        ], tight=True),
                        padding=15
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
                ft.Text("SnapIndex Semantic Search", size=24, weight=ft.FontWeight.BOLD)
            ]),
            ft.Text(f"Searching in: {selected_folder} (FAISS-powered semantic search)", color="gray"),
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
