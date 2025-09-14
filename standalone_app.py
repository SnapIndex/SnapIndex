import flet as ft
import os
import subprocess
import hashlib
import platform
from simple_faiss import create_faiss_db, search_faiss_db, delete_faiss_db
from file_loader import find_documents, extract_text_from_file

# Global variables
selected_folder = "C:\\Users\\akarsh\\Downloads"
current_tab = "search"
db_path = "./faiss_database"

def search_in_documents(query, folder_path):
    """Search for query text using FAISS semantic search across all document types"""
    try:
        # Create a unique database path for each folder (avoid overlaps across same-named folders)
        abs_folder = os.path.abspath(folder_path)
        folder_hash = hashlib.sha1(abs_folder.encode("utf-8")).hexdigest()[:12]
        safe_name = os.path.basename(abs_folder) or "root"
        folder_db_path = os.path.join("./faiss_databases", f"{safe_name}_{folder_hash}")
        
        # Check if database exists, if not create it, otherwise update incrementally
        if not os.path.exists(folder_db_path):
            print(f"Creating new FAISS database for folder: {folder_path}")
            create_faiss_db(folder_path, folder_db_path, incremental=False)
        else:
            print(f"Updating FAISS database incrementally for folder: {folder_path}")
            create_faiss_db(folder_path, folder_db_path, incremental=True)
        
        # Use FAISS search and return results in the expected format
        return _get_faiss_results(query, folder_db_path, k=10, folder_path=folder_path)
        
    except Exception as e:
        print(f"Error in FAISS search: {e}")
        return []

# Backward compatibility
def search_in_pdfs(query, folder_path):
    """Backward compatibility - same as search_in_documents"""
    return search_in_documents(query, folder_path)

def _get_faiss_results(query_string, db_path, k=10, folder_path=None):
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
    
    # Format results to match the original structure
    results = []
    threshold = 0.6  # Only show results with similarity >= 0.75
    
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        
        meta = metadata[idx]
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        
        # Only include results above the threshold
        if similarity >= threshold:
            # Determine file type from the file name
            file_name = meta["pdf_name"]
            file_ext = os.path.splitext(file_name)[1].lower()
            file_type = file_ext[1:] if file_ext else "unknown"
            
            # Determine if this is a filename match or content match
            chunk_type = meta.get("chunk_type", "content")
            page_display = "Filename" if chunk_type == "filename" else f"Page {meta['page_number']}"
            
            results.append({
                "file": file_name,
                "page": page_display,
                "snippet": meta["text"][:300] + "..." if len(meta["text"]) > 300 else meta["text"],
                "full_path": folder_path or selected_folder,  # Use the actual folder path
                "similarity": similarity,
                "file_type": file_type,
                "chunk_type": chunk_type
            })
    
    # Deduplicate: keep only the best match per file to avoid repeated results
    best_by_file = {}
    for item in results:
        key = item["file"]  # collapse to one result per file
        if key not in best_by_file or item["similarity"] > best_by_file[key]["similarity"]:
            best_by_file[key] = item
    deduped = list(best_by_file.values())
    # Sort by similarity descending
    deduped.sort(key=lambda r: r.get("similarity", 0), reverse=True)
    
    return deduped

def open_file_location(file_path):
    """Open the PDF file directly"""
    try:
        if platform.system() == "Windows":
            # Use Windows default program to open the PDF file
            os.startfile(file_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['open', file_path], check=True)
        else:  # Linux
            subprocess.run(['xdg-open', file_path], check=True)
    except Exception as e:
        print(f"Error opening file: {e}")

def create_search_content(file_picker):
    """Create search tab content"""
    def perform_search(e, search_field, results_container):
        query = search_field.value.strip()
        if not query:
            results_container.controls.clear()
            results_container.controls.append(
                ft.Text("Please enter a search query.", color="gray", italic=True)
            )
            return
        
        if not selected_folder or not os.path.exists(selected_folder):
            results_container.controls.clear()
            results_container.controls.append(
                ft.Text("Please select a valid folder first.", color="red", italic=True)
            )
            return
        
        # Clear previous results
        results_container.controls.clear()
        results_container.controls.append(
            ft.Text(f"Searching for: '{query}'", weight=ft.FontWeight.BOLD, size=16)
        )
        results_container.controls.append(ft.Divider())
        
        # Perform search
        results = search_in_pdfs(query, selected_folder)
        
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
            for result in results:
                # Create clickable result card
                def create_click_handler(result_data):
                    def on_card_click(e):
                        # Construct the full file path
                        file_name = result_data['file']
                        folder_path = result_data['full_path']
                        full_file_path = os.path.join(folder_path, file_name)
                        
                        # Open the PDF file directly
                        open_file_location(full_file_path)
                    return on_card_click
                
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
                                    f"{result['file']} ({result.get('file_type', 'unknown').upper()}) - {result['page']}",
                                    weight=ft.FontWeight.BOLD,
                                    color="blue"
                                ),
                                    ft.Text(
                                        f"Similarity: {result.get('similarity', 0):.3f}",
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
                                ft.Row([
                                    ft.Text(
                                        f"File: {result['full_path']}",
                                        size=10,
                                        color="gray",
                                        italic=True
                                    ),
                                ft.Text(
                                    "Click to open file",
                                    size=10,
                                    color="blue",
                                    italic=True,
                                    weight=ft.FontWeight.BOLD
                                )
                                ])
                            ], tight=True),
                            padding=15
                        ),
                        on_tap=create_click_handler(result)
                    ),
                    margin=ft.margin.only(bottom=10)
                )
                results_container.controls.append(result_card)
        else:
            results_container.controls.append(
                ft.Text("No matches found.", color="gray", italic=True)
            )
        
        results_container.update()
    
    # File picker will be added to page overlay in main function
    
    # Header with banner image
    header = ft.Container(
        content=ft.Image(
            src="banner.png",
        ),
        border_radius=ft.border_radius.all(15),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        margin=ft.margin.only(left=-20, right=-20, top=-20),
        padding=ft.padding.only(bottom=20),
    )
    
    # Folder selection section
    folder_display = ft.Text(f"Selected: {selected_folder}", size=12, color=ft.Colors.GREY_600)
    
    folder_section = ft.Container(
        content=ft.Column([
            ft.Text("Select Search Folder", size=16, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.ElevatedButton(
                    "Choose Folder",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=lambda _: file_picker.get_directory_path(),
                    bgcolor=ft.Colors.GREEN_600,
                    color=ft.Colors.WHITE,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.padding.symmetric(horizontal=20, vertical=12),
                        elevation=3,
                        shadow_color=ft.Colors.GREEN_800
                    )
                ),
                folder_display
            ], alignment=ft.MainAxisAlignment.START, spacing=10)
        ], spacing=8),
        padding=ft.padding.only(bottom=20)
    )
    
    # Search section
    search_field = ft.TextField(
        label="Search All Documents",
        hint_text="Search across PDFs, Word docs, Excel, PowerPoint, and text files...",
        expand=True,
        prefix_icon=ft.Icons.SEARCH,
        on_submit=lambda e: perform_search(e, search_field, results_container)
    )
    
    search_button = ft.ElevatedButton(
        "Search Documents",
        icon=ft.Icons.SEARCH,
        on_click=lambda e: perform_search(e, search_field, results_container),
        bgcolor=ft.Colors.GREEN,
        color=ft.Colors.WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8)
        )
    )
    
    results_container = ft.Column(
        expand=True
    )

    return ft.Column([
        header,
        folder_section,
        ft.Row([
            search_field,
            search_button
        ], spacing=15),
        ft.Divider(),
        ft.Container(
            content=results_container,
            expand=True,
            padding=10
        )
    ], expand=True, scroll=ft.ScrollMode.AUTO)

def create_organize_content():
    """Create organize tab content"""
    return ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.FOLDER_OPEN, color="green", size=30),
            ft.Text("Organize PDFs", size=24, weight=ft.FontWeight.BOLD)
        ]),
        ft.Text("Organize and categorize your PDF documents", color="gray"),
        ft.Divider(),
        
        ft.ElevatedButton(
            "Create Categories",
            icon=ft.Icons.ADD,
            bgcolor="green",
            color="white",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        ),
        ft.ElevatedButton(
            "Auto-categorize",
            icon=ft.Icons.AUTO_AWESOME,
            bgcolor="teal",
            color="white",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        ),
        ft.ElevatedButton(
            "Import Categories",
            icon=ft.Icons.UPLOAD,
            bgcolor="blue",
            color="white",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        )
    ], expand=True, scroll=ft.ScrollMode.AUTO)

def create_rename_content():
    """Create rename tab content"""
    return ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.EDIT, color="orange", size=30),
            ft.Text("Rename PDFs", size=24, weight=ft.FontWeight.BOLD)
        ]),
        ft.Text("Rename and organize your PDF files", color="gray"),
        ft.Divider(),
        
        ft.ElevatedButton(
            "Batch Rename",
            icon=ft.Icons.DRIVE_FILE_RENAME_OUTLINE,
            bgcolor="orange",
            color="white",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        ),
        ft.ElevatedButton(
            "Auto-rename by Content",
            icon=ft.Icons.SMART_TOY,
            bgcolor="purple",
            color="white",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        ),
        ft.ElevatedButton(
            "Rename by Date",
            icon=ft.Icons.CALENDAR_MONTH,
            bgcolor="indigo",
            color="white",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        )
    ], expand=True, scroll=ft.ScrollMode.AUTO)

def create_settings_content():
    """Create settings tab content"""
    folder_field = ft.TextField(
        label="Default Search Folder",
        value=selected_folder,
        expand=True,
        read_only=True
    )
    
    def browse_folder(e):
        # Note: Flet doesn't have native file dialog yet, so we'll use a simple input
        folder_field.value = "C:\\Users\\akarsh\\Downloads"  # Placeholder
        folder_field.update()
    
    return ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.SETTINGS, color="gray", size=30),
            ft.Text("Settings", size=24, weight=ft.FontWeight.BOLD)
        ]),
        ft.Text("Configure SnapIndex preferences", color="gray"),
        ft.Divider(),
        
        ft.Text("Default Search Folder:", weight=ft.FontWeight.BOLD),
        ft.Row([
            folder_field,
            ft.ElevatedButton(
                "Browse",
                icon=ft.Icons.FOLDER_OPEN,
                on_click=browse_folder
            )
        ]),
        
        ft.Divider(),
        
        ft.Text("Search Options:", weight=ft.FontWeight.BOLD),
        ft.Switch(label="Case sensitive search", value=False),
        ft.Switch(label="Search in subfolders", value=True),
        ft.Switch(label="Show file paths", value=True),
        
        ft.Divider(),
        
        ft.ElevatedButton(
            "Save Settings",
            icon=ft.Icons.SAVE,
            bgcolor="gray",
            color="white",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        )
    ], expand=True, scroll=ft.ScrollMode.AUTO)

def main(page: ft.Page):
    page.title = "SnapIndex"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1000
    page.window_height = 700
    page.padding = 0
    
    # Set custom window icon to replace default Flet icon
    try:
        page.window_icon = "logo-dark.svg"
    except:
        # Fallback if icon file not found
        pass

    # Create file picker
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)

    # Create sidebar
    def create_sidebar():
        def on_tab_change(tab_name):
            global current_tab
            current_tab = tab_name
            
            # Update button styles
            for btn in sidebar_buttons:
                btn.bgcolor = "white" if btn.data != tab_name else "#f5f7fa"
            
            # Update content
            content_container.content = get_content_for_tab(tab_name)
            content_container.update()
            page.update()
        
        sidebar_buttons = []
        
        # Search button
        search_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.SEARCH, color="blue"),
                ft.Text("Search", weight=ft.FontWeight.BOLD if current_tab == "search" else ft.FontWeight.NORMAL)
            ]),
            bgcolor="#f5f7fa" if current_tab == "search" else "white",
            padding=8,
            on_click=lambda e: on_tab_change("search"),
            data="search"
        )
        sidebar_buttons.append(search_btn)
        
        # Organize button
        organize_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.FOLDER_OPEN, color="green"),
                ft.Text("Organize", weight=ft.FontWeight.BOLD if current_tab == "organize" else ft.FontWeight.NORMAL)
            ]),
            bgcolor="#f5f7fa" if current_tab == "organize" else "white",
            padding=8,
            on_click=lambda e: on_tab_change("organize"),
            data="organize"
        )
        sidebar_buttons.append(organize_btn)
        
        # Rename button
        rename_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.EDIT, color="orange"),
                ft.Text("Rename", weight=ft.FontWeight.BOLD if current_tab == "rename" else ft.FontWeight.NORMAL)
            ]),
            bgcolor="#f5f7fa" if current_tab == "rename" else "white",
            padding=8,
            on_click=lambda e: on_tab_change("rename"),
            data="rename"
        )
        sidebar_buttons.append(rename_btn)
        
        # Settings button
        settings_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.SETTINGS, color="gray"),
                ft.Text("Settings", weight=ft.FontWeight.BOLD if current_tab == "settings" else ft.FontWeight.NORMAL)
            ]),
            bgcolor="#f5f7fa" if current_tab == "settings" else "white",
            padding=8,
            on_click=lambda e: on_tab_change("settings"),
            data="settings"
        )
        sidebar_buttons.append(settings_btn)
        
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([
                        ft.Image(
                            src="logo-dark.svg",
                            width=24,
                            height=24,
                            fit=ft.ImageFit.CONTAIN
                        ),
                        ft.Text("SnapIndex", size=20, weight=ft.FontWeight.BOLD)
                    ]),
                    padding=10
                ),
                ft.Divider(),
                *sidebar_buttons
            ]),
            width=200,
            bgcolor="white",
            border=ft.border.only(right=ft.BorderSide(1, "gray"))
        )
    
    def get_content_for_tab(tab_name):
        if tab_name == "search":
            return create_search_content(file_picker)
        elif tab_name == "organize":
            return create_organize_content()
        elif tab_name == "rename":
            return create_rename_content()
        elif tab_name == "settings":
            return create_settings_content()
        return create_search_content(file_picker)
    
    # Create main layout
    sidebar = create_sidebar()
    content_container = ft.Container(
        content=get_content_for_tab(current_tab),
        expand=True,
        padding=20
    )
    
    # Set up folder picker callback
    def on_folder_picked(e: ft.FilePickerResultEvent):
        if e.path:
            global selected_folder
            selected_folder = e.path
            # Update the content to refresh folder display
            content_container.content = get_content_for_tab(current_tab)
            content_container.update()
    
    file_picker.on_result = on_folder_picked
    
    page.add(
        ft.Row([
            sidebar,
            content_container
        ], expand=True)
    )

ft.app(target=main)
