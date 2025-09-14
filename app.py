import flet as ft
import os
import subprocess
import hashlib
import platform
import sys
from simple_faiss import create_faiss_db, search_faiss_db, delete_faiss_db
from file_loader import find_documents, extract_text_from_file

# Function to get resource path for bundled files
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Default folder path
DEFAULT_FOLDER = "C:\\Users\\akarsh\\Downloads"

# Global variables
selected_folder = DEFAULT_FOLDER
current_tab = "search"
db_path = "./faiss_database"

def show_usage():
    """Display usage information"""
    print("SnapIndex - Semantic Document Search")
    print("Usage: python app.py [folder_path]")
    print(f"  folder_path: Optional path to search folder (default: {DEFAULT_FOLDER})")
    print("Examples:")
    print(f"  python app.py                    # Use default folder: {DEFAULT_FOLDER}")
    print("  python app.py C:\\Documents       # Use C:\\Documents as search folder")
    print("  python app.py /home/user/docs     # Use /home/user/docs as search folder")

def get_folder_path():
    """Get folder path from command line arguments or use default"""
    if len(sys.argv) > 1:
        # Check for help flags
        if sys.argv[1] in ['-h', '--help', '/?']:
            show_usage()
            sys.exit(0)
        
        provided_path = sys.argv[1]
        if os.path.exists(provided_path) and os.path.isdir(provided_path):
            abs_path = os.path.abspath(provided_path)
            print(f"Using provided folder: {abs_path}")
            return abs_path
        else:
            print(f"Error: Provided path '{provided_path}' does not exist or is not a directory.")
            print(f"Using default folder: {DEFAULT_FOLDER}")
            return DEFAULT_FOLDER
    else:
        print(f"Using default folder: {DEFAULT_FOLDER}")
        print("Tip: You can specify a folder path as a command line argument.")
        print("Run 'python app.py --help' for usage information.")
        return DEFAULT_FOLDER

# Initialize selected folder
selected_folder = get_folder_path()

def search_in_documents(query, folder_path, progress_callback=None):
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
            create_faiss_db(folder_path, folder_db_path, incremental=False, progress_callback=progress_callback)
        else:
            print(f"Updating FAISS database incrementally for folder: {folder_path}")
            create_faiss_db(folder_path, folder_db_path, incremental=True, progress_callback=progress_callback)
        
        # Use FAISS search and return results in the expected format
        return _get_faiss_results(query, folder_db_path, k=10, folder_path=folder_path)
        
    except Exception as e:
        print(f"Error in FAISS search: {e}")
        return []

# Backward compatibility
def search_in_pdfs(query, folder_path, progress_callback=None):
    """Backward compatibility - same as search_in_documents"""
    return search_in_documents(query, folder_path, progress_callback)

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
    # Progress bar components (will be updated from main function)
    progress_bar = ft.ProgressBar(
        value=0, 
        visible=False, 
        width=500,
        color=ft.Colors.ORANGE,
        bgcolor=ft.Colors.GREY_300,
        bar_height=12
    )
    progress_text = ft.Text(
        "Processing documents...", 
        visible=False, 
        color=ft.Colors.ORANGE,
        size=16,
        weight=ft.FontWeight.BOLD
    )
    progress_percentage = ft.Text(
        "0%", 
        visible=False, 
        color=ft.Colors.ORANGE,
        size=18,
        weight=ft.FontWeight.BOLD
    )
    progress_time_remaining = ft.Text(
        "Calculating time remaining...", 
        visible=False, 
        color=ft.Colors.GREY_700,
        size=13
    )
    progress_description = ft.Text(
        "Please wait while we process your documents...",
        visible=False,
        color=ft.Colors.GREY_600,
        size=12,
        italic=True
    )
    progress_container = ft.Container(
        content=ft.Column([
            progress_text,
            ft.Row([
                progress_bar,
                progress_percentage
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
            progress_time_remaining,
            progress_description
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
        visible=False,
        padding=25,
        bgcolor=ft.Colors.WHITE,
        border_radius=12,
        border=ft.border.all(2, ft.Colors.ORANGE),
        shadow=ft.BoxShadow(
            spread_radius=2,
            blur_radius=8,
            color=ft.Colors.GREY_400,
            offset=ft.Offset(0, 2)
        )
    )
    
    print(f"DEBUG: Created progress components - Container: {progress_container}, Text: {progress_text}, Bar: {progress_bar}")
    
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
    banner_path = resource_path("banner.png")
    header = ft.Container(
        content=ft.Image(
            src=banner_path if os.path.exists(banner_path) else None,
            fit=ft.ImageFit.COVER if os.path.exists(banner_path) else None,
        ) if os.path.exists(banner_path) else ft.Container(
            content=ft.Column([
                ft.Text("SnapIndex", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE),
                ft.Text("Semantic Document Search", size=16, color=ft.Colors.GREY_600)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20
        ),
        border_radius=ft.border_radius.all(15),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        margin=ft.margin.only(left=-20, right=-20, top=-160),
        padding=ft.padding.only(bottom=10),
    )
    
    # Folder selection section
    folder_display = ft.Text(f"Selected: {selected_folder}", size=12, color=ft.Colors.GREY_600, ref=ft.Ref[ft.Text]())
    
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

    search_content = ft.Column([
        header,
        folder_section,
        progress_container,
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
    
    # Return both the content and progress components for external access
    return search_content, progress_bar, progress_text, progress_container, progress_percentage, progress_time_remaining, progress_description

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
        logo_path = resource_path("logo-dark.svg")
        page.window_icon = logo_path
    except:
        # Fallback if icon file not found
        pass
    
    # Print current folder information
    print(f"SnapIndex started with search folder: {selected_folder}")
    if not os.path.exists(selected_folder):
        print(f"Warning: Selected folder does not exist: {selected_folder}")
        print("Please select a valid folder using the 'Choose Folder' button.")
    else:
        print(f"Initializing search database for: {selected_folder}")

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
                            src=resource_path("logo-dark.svg"),
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
    
    # Global variables to store progress components
    global_progress_bar = None
    global_progress_text = None
    global_progress_container = None
    global_progress_percentage = None
    global_progress_time_remaining = None
    global_progress_description = None
    
    # Time tracking for progress estimation
    progress_start_time = None
    last_progress_time = None
    last_progress_value = 0
    
    def get_content_for_tab(tab_name):
        nonlocal global_progress_bar, global_progress_text, global_progress_container, global_progress_percentage, global_progress_time_remaining, global_progress_description
        
        if tab_name == "search":
            content, progress_bar, progress_text, progress_container, progress_percentage, progress_time_remaining, progress_description = create_search_content(file_picker)
            # Store progress components globally
            global_progress_bar = progress_bar
            global_progress_text = progress_text
            global_progress_container = progress_container
            global_progress_percentage = progress_percentage
            global_progress_time_remaining = progress_time_remaining
            global_progress_description = progress_description
            print(f"DEBUG: Progress components stored - Container: {global_progress_container}, Text: {global_progress_text}, Bar: {global_progress_bar}")
            return content
        elif tab_name == "organize":
            return create_organize_content()
        elif tab_name == "rename":
            return create_rename_content()
        elif tab_name == "settings":
            return create_settings_content()
        else:
            content, progress_bar, progress_text, progress_container, progress_percentage, progress_time_remaining, progress_description = create_search_content(file_picker)
            # Store progress components globally
            global_progress_bar = progress_bar
            global_progress_text = progress_text
            global_progress_container = progress_container
            global_progress_percentage = progress_percentage
            global_progress_time_remaining = progress_time_remaining
            global_progress_description = progress_description
            return content
    
    def show_progress(message="Processing documents..."):
        """Show progress bar with message"""
        nonlocal progress_start_time, last_progress_time, last_progress_value
        import time
        
        print(f"DEBUG: show_progress called with message: {message}")
        if global_progress_container and global_progress_text and global_progress_bar:
            print("DEBUG: Progress components found, updating...")
            global_progress_text.value = message
            global_progress_bar.value = 0
            global_progress_percentage.value = "0%"
            global_progress_time_remaining.value = "Calculating time remaining..."
            
            # Make all components visible
            global_progress_container.visible = True
            global_progress_text.visible = True
            global_progress_bar.visible = True
            global_progress_percentage.visible = True
            global_progress_time_remaining.visible = True
            if global_progress_description:
                global_progress_description.visible = True
            
            # Initialize time tracking
            progress_start_time = time.time()
            last_progress_time = progress_start_time
            last_progress_value = 0
            
            print(f"DEBUG: Progress container visible set to: {global_progress_container.visible}")
            page.update()
        else:
            print("DEBUG: Progress components not found!")
            print(f"DEBUG: Container: {global_progress_container}, Text: {global_progress_text}, Bar: {global_progress_bar}")
    
    def update_progress(value, message=None):
        """Update progress bar value and optionally message"""
        nonlocal progress_start_time, last_progress_time, last_progress_value
        import time
        
        print(f"DEBUG: update_progress called with value: {value}, message: {message}")
        if global_progress_bar and global_progress_percentage and global_progress_time_remaining:
            global_progress_bar.value = value
            
            # Update percentage
            percentage = int(value * 100)
            global_progress_percentage.value = f"{percentage}%"
            
            # Update message
            if message and global_progress_text:
                global_progress_text.value = message
            
            # Calculate time remaining
            current_time = time.time()
            if progress_start_time and value > 0:
                elapsed_time = current_time - progress_start_time
                if value > last_progress_value and elapsed_time > 0:
                    # Estimate total time based on current progress
                    estimated_total_time = elapsed_time / value
                    remaining_time = estimated_total_time - elapsed_time
                    
                    if remaining_time > 60:
                        minutes = int(remaining_time // 60)
                        seconds = int(remaining_time % 60)
                        global_progress_time_remaining.value = f"Estimated time remaining: {minutes}m {seconds}s"
                    elif remaining_time > 10:
                        global_progress_time_remaining.value = f"Estimated time remaining: {int(remaining_time)}s"
                    elif remaining_time > 0:
                        global_progress_time_remaining.value = "Almost done..."
                    else:
                        global_progress_time_remaining.value = "Finishing up..."
                else:
                    global_progress_time_remaining.value = "Calculating time remaining..."
            
            last_progress_time = current_time
            last_progress_value = value
            page.update()
            print(f"DEBUG: Progress updated to {value} ({percentage}%)")
        else:
            print("DEBUG: global_progress_bar not found in update_progress!")
    
    def hide_progress():
        """Hide progress bar"""
        nonlocal progress_start_time, last_progress_time, last_progress_value
        print("DEBUG: hide_progress called")
        if global_progress_container:
            # Hide all progress components
            global_progress_container.visible = False
            if global_progress_text:
                global_progress_text.visible = False
            if global_progress_bar:
                global_progress_bar.visible = False
            if global_progress_percentage:
                global_progress_percentage.visible = False
            if global_progress_time_remaining:
                global_progress_time_remaining.visible = False
            if global_progress_description:
                global_progress_description.visible = False
            
            # Reset time tracking
            progress_start_time = None
            last_progress_time = None
            last_progress_value = 0
            
            page.update()
            print("DEBUG: Progress hidden")
        else:
            print("DEBUG: global_progress_container not found in hide_progress!")
    
    # Create main layout
    sidebar = create_sidebar()
    content_container = ft.Container(
        content=get_content_for_tab(current_tab),
        expand=True,
        padding=ft.padding.only(left=20, right=20, bottom=20)
    )
    
    # Initialize database on startup if folder exists
    def initialize_database_on_startup():
        """Initialize the search database for the selected folder on app startup"""
        if os.path.exists(selected_folder):
            print("Starting automatic database initialization...")
            
            # Show progress immediately
            show_progress("Initializing search database...")
            
            # Process folder with real progress tracking
            import threading
            
            def process_startup_folder():
                try:
                    # Create a unique database path for the folder
                    abs_folder = os.path.abspath(selected_folder)
                    folder_hash = hashlib.sha1(abs_folder.encode("utf-8")).hexdigest()[:12]
                    safe_name = os.path.basename(abs_folder) or "root"
                    folder_db_path = os.path.join("./faiss_databases", f"{safe_name}_{folder_hash}")
                    
                    # Check if database exists and create/update it with real progress
                    if not os.path.exists(folder_db_path):
                        print(f"Creating new FAISS database for folder: {selected_folder}")
                        create_faiss_db(selected_folder, folder_db_path, incremental=False, progress_callback=update_progress)
                    else:
                        print(f"Updating FAISS database incrementally for folder: {selected_folder}")
                        create_faiss_db(selected_folder, folder_db_path, incremental=True, progress_callback=update_progress)
                    
                    # Wait a moment to show completion, then hide progress bar
                    import time
                    time.sleep(1)
                    hide_progress()
                    
                    print(f"Successfully initialized search database for: {selected_folder}")
                    
                except Exception as ex:
                    hide_progress()
                    print(f"Error initializing database: {ex}")
            
            # Run in background thread to avoid blocking UI
            thread = threading.Thread(target=process_startup_folder)
            thread.daemon = True
            thread.start()
    
    # Start database initialization after a short delay to ensure UI is ready
    import threading
    import time
    
    def delayed_startup():
        time.sleep(0.5)  # Small delay to ensure UI is fully loaded
        initialize_database_on_startup()
    
    startup_thread = threading.Thread(target=delayed_startup)
    startup_thread.daemon = True
    startup_thread.start()
    
    # Set up folder picker callback
    def on_folder_picked(e: ft.FilePickerResultEvent):
        if e.path:
            global selected_folder
            selected_folder = e.path
            
            # Update folder display text immediately
            if current_tab == "search":
                # Find and update the folder display text
                search_content = content_container.content
                if hasattr(search_content, 'controls') and len(search_content.controls) > 1:
                    folder_section = search_content.controls[1]  # folder_section is the second control
                    if hasattr(folder_section, 'content') and hasattr(folder_section.content, 'controls'):
                        folder_row = folder_section.content.controls[1]  # folder_row is the second control in folder_section
                        if hasattr(folder_row, 'controls') and len(folder_row.controls) > 1:
                            folder_display_text = folder_row.controls[1]  # folder_display is the second control in folder_row
                            folder_display_text.value = f"Selected: {selected_folder}"
                            folder_display_text.update()
            
            # Show progress bar immediately
            show_progress("Setting up search database...")
            
            # Process folder with real progress tracking
            import threading
            
            def process_folder():
                try:
                    # Create a unique database path for the folder
                    abs_folder = os.path.abspath(selected_folder)
                    folder_hash = hashlib.sha1(abs_folder.encode("utf-8")).hexdigest()[:12]
                    safe_name = os.path.basename(abs_folder) or "root"
                    folder_db_path = os.path.join("./faiss_databases", f"{safe_name}_{folder_hash}")
                    
                    # Check if database exists and create/update it with real progress
                    if not os.path.exists(folder_db_path):
                        print(f"Creating new FAISS database for folder: {selected_folder}")
                        create_faiss_db(selected_folder, folder_db_path, incremental=False, progress_callback=update_progress)
                    else:
                        print(f"Updating FAISS database incrementally for folder: {selected_folder}")
                        create_faiss_db(selected_folder, folder_db_path, incremental=True, progress_callback=update_progress)
                    
                    # Wait a moment to show completion, then hide progress bar
                    import time
                    time.sleep(1)
                    hide_progress()
                    
                    print(f"Successfully set up search database for: {selected_folder}")
                    
                except Exception as ex:
                    hide_progress()
                    print(f"Error processing folder: {ex}")
            
            # Run in background thread to avoid blocking UI
            thread = threading.Thread(target=process_folder)
            thread.daemon = True
            thread.start()
    
    file_picker.on_result = on_folder_picked
    
    page.add(
        ft.Row([
            sidebar,
            content_container
        ], expand=True)
    )

ft.app(target=main)
