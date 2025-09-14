import flet as ft
import os
import subprocess
import hashlib
import platform
import shutil
import json
from datetime import datetime
from simple_faiss import create_faiss_db, search_faiss_db, delete_faiss_db
from file_loader import find_documents, extract_text_from_file

# Global variables
selected_folder = "C:\\Users\\akarsh\\Downloads"
current_tab = "search"
db_path = "./faiss_database"

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
    header = ft.Container(
        content=ft.Image(
            src="banner.png",
        ),
        border_radius=ft.border_radius.all(20),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        margin=ft.margin.only(left=0, right=0, top=-50),
        padding=ft.padding.only(bottom=0),
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

def create_organize_content(file_picker):
    """Create organize tab content"""
    # Global variables for source and destination folders
    source_folder = ""
    destination_folder = ""
    
    # Create separate file pickers for source and destination
    source_picker = ft.FilePicker()
    destination_picker = ft.FilePicker()
    
    # Progress bar components
    progress_bar = ft.ProgressBar(
        value=0, 
        visible=False, 
        width=500,
        color=ft.Colors.ORANGE,
        bgcolor=ft.Colors.GREY_300,
        bar_height=12
    )
    progress_text = ft.Text(
        "Organizing files...", 
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
    
    # Tree view for showing file moves
    tree_view = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        spacing=2,
        height=300,
        controls=[ft.Text("📁 Select source and destination folders, then click 'Organize Files' to begin", 
                          size=12, color=ft.Colors.GREY_600, italic=True)]
    )
    
    tree_container = ft.Container(
        content=ft.Column([
            ft.Text("File Organization Tree:", size=14, weight=ft.FontWeight.BOLD),
            ft.Container(
                content=tree_view,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=8,
                padding=10,
                bgcolor=ft.Colors.GREY_50,
                height=300
            )
        ], spacing=8),
        visible=False,  # Hidden initially, shown after organization
        padding=ft.padding.only(bottom=10)
    )
    
    progress_container = ft.Container(
        content=ft.Column([
            progress_text,
            ft.Row([
                progress_bar,
                progress_percentage
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
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
    
    # File extension categories
    file_categories = {
        'Documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.md'],
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp', '.ico', '.psd'],
        'Videos': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.3gp', '.ogv'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.amr'],
        'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.iso'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php', '.rb', '.go', '.rs', '.sql'],
        'Executables': ['.exe', '.msi', '.deb', '.rpm', '.dmg', '.app', '.bat', '.sh']
    }
    
    def get_file_category(filename):
        """Determine file category based on extension"""
        ext = os.path.splitext(filename)[1].lower()
        for category, extensions in file_categories.items():
            if ext in extensions:
                return category
        return 'Other'
    
    def create_tree_node(text, icon=None, color=None, indent=0):
        """Create a tree node with proper indentation and styling"""
        node_content = ft.Row([
            ft.Text("  " * indent, size=12),  # Indentation
            ft.Icon(icon, size=16, color=color) if icon else ft.Text("", width=16),
            ft.Text(text, size=12, color=color or ft.Colors.BLACK)
        ], spacing=5)
        return ft.Container(
            content=node_content,
            padding=ft.padding.only(left=5, right=5, top=2, bottom=2),
            bgcolor=ft.Colors.WHITE if indent == 0 else ft.Colors.GREY_50,
            border_radius=4
        )
    
    def clear_tree_view():
        """Clear the tree view"""
        tree_view.controls.clear()
        try:
            tree_view.update()
        except AssertionError:
            # Tree view not added to page yet, skip update
            pass
    
    def add_to_tree_view(node):
        """Add a node to the tree view"""
        tree_view.controls.append(node)
        try:
            tree_view.update()
        except AssertionError:
            # Tree view not added to page yet, skip update
            pass
    
    def update_tree_view_with_move(source_path, dest_path, category):
        """Add a file move to the tree view"""
        filename = os.path.basename(source_path)
        category_name = category if category != "Directories" else "📁 Directories"
        
        # Add category folder if not already shown
        category_exists = any(
            isinstance(control.content, ft.Row) and 
            len(control.content.controls) > 1 and
            isinstance(control.content.controls[1], ft.Icon) and
            category_name in str(control.content.controls[2].value)
            for control in tree_view.controls
        )
        
        if not category_exists:
            category_icon = ft.Icons.FOLDER
            if category == "Documents":
                category_icon = ft.Icons.DESCRIPTION
            elif category == "Images":
                category_icon = ft.Icons.IMAGE
            elif category == "Videos":
                category_icon = ft.Icons.VIDEO_FILE
            elif category == "Audio":
                category_icon = ft.Icons.AUDIO_FILE
            elif category == "Archives":
                category_icon = ft.Icons.ARCHIVE
            elif category == "Code":
                category_icon = ft.Icons.CODE
            elif category == "Executables":
                category_icon = ft.Icons.APPS
            elif category == "Directories":
                category_icon = ft.Icons.FOLDER_OPEN
            else:
                category_icon = ft.Icons.INSERT_DRIVE_FILE
            
            category_node = create_tree_node(
                f"📁 {category_name}",
                icon=category_icon,
                color=ft.Colors.BLUE_600,
                indent=0
            )
            add_to_tree_view(category_node)
        
        # Add file to category
        file_icon = ft.Icons.INSERT_DRIVE_FILE
        if category == "Images":
            file_icon = ft.Icons.IMAGE
        elif category == "Videos":
            file_icon = ft.Icons.VIDEO_FILE
        elif category == "Audio":
            file_icon = ft.Icons.AUDIO_FILE
        elif category == "Documents":
            file_icon = ft.Icons.DESCRIPTION
        
        file_node = create_tree_node(
            f"📄 {filename}",
            icon=file_icon,
            color=ft.Colors.GREY_700,
            indent=1
        )
        add_to_tree_view(file_node)
    
    
    def organize_files():
        """Organize files from source to destination with path tracking"""
        if not source_folder or not destination_folder:
            return
        
        try:
            # Show progress immediately
            progress_container.visible = True
            progress_text.visible = True
            progress_bar.visible = True
            progress_percentage.visible = True
            progress_bar.value = 0
            progress_text.value = "Starting file organization..."
            progress_percentage.value = "0%"
            
            # Clear and initialize tree view
            clear_tree_view()
            
            # Add initial message to tree view
            add_to_tree_view(create_tree_node("📋 Preparing to organize files...", color=ft.Colors.BLUE_600))
            
            # Force update the tree container to show the initial message
            tree_container.update()
            
            # Create rollback tracking file
            rollback_file = os.path.join(destination_folder, "rollback_info.json")
            rollback_data = {
                "timestamp": datetime.now().isoformat(),
                "source_folder": source_folder,
                "destination_folder": destination_folder,
                "moves": [],
                "created_directories": []  # Track directories created during organization
            }
            
            # Get all files and directories from source
            all_items = []
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    all_items.append(os.path.join(root, file))
                for dir_name in dirs:
                    all_items.append(os.path.join(root, dir_name))
            
            total_items = len(all_items)
            if total_items == 0:
                progress_text.value = "No files found to organize."
                # Clear the initial message and show no files message
                clear_tree_view()
                add_to_tree_view(create_tree_node("❌ No files found to organize.", color=ft.Colors.RED_600))
                return
            
            # Update tree view to show scanning complete
            clear_tree_view()
            add_to_tree_view(create_tree_node(f"📊 Found {total_items} items to organize", color=ft.Colors.GREEN_600))
            add_to_tree_view(create_tree_node("🔄 Starting organization...", color=ft.Colors.ORANGE_600))
            
            processed = 0
            
            for item_path in all_items:
                try:
                    item_name = os.path.basename(item_path)
                    original_path = item_path  # Store original path
                    
                    if os.path.isfile(item_path):
                        # It's a file - categorize by extension
                        category = get_file_category(item_name)
                        category_dir = os.path.join(destination_folder, category)
                        
                        # Create category directory if it doesn't exist
                        if not os.path.exists(category_dir):
                            os.makedirs(category_dir, exist_ok=True)
                            rollback_data["created_directories"].append(category_dir)
                        
                        # Move file
                        dest_path = os.path.join(category_dir, item_name)
                        if os.path.exists(dest_path):
                            # Handle duplicate names
                            base, ext = os.path.splitext(item_name)
                            counter = 1
                            while os.path.exists(dest_path):
                                new_name = f"{base}_{counter}{ext}"
                                dest_path = os.path.join(category_dir, new_name)
                                counter += 1
                        
                        # Move the file and track the move
                        shutil.move(item_path, dest_path)
                        
                        # Update tree view
                        update_tree_view_with_move(original_path, dest_path, category)
                        
                        # Record the move for rollback
                        rollback_data["moves"].append({
                            "original_path": original_path,
                            "new_path": dest_path,
                            "type": "file",
                            "category": category
                        })
                        
                    elif os.path.isdir(item_path):
                        # It's a directory - move to "Directories" folder
                        dirs_dir = os.path.join(destination_folder, "Directories")
                        if not os.path.exists(dirs_dir):
                            os.makedirs(dirs_dir, exist_ok=True)
                            rollback_data["created_directories"].append(dirs_dir)
                        
                        dest_path = os.path.join(dirs_dir, item_name)
                        if os.path.exists(dest_path):
                            # Handle duplicate directory names
                            counter = 1
                            while os.path.exists(dest_path):
                                new_name = f"{item_name}_{counter}"
                                dest_path = os.path.join(dirs_dir, new_name)
                                counter += 1
                        
                        # Move the directory and track the move
                        shutil.move(item_path, dest_path)
                        
                        # Update tree view
                        update_tree_view_with_move(original_path, dest_path, "Directories")
                        
                        # Record the move for rollback
                        rollback_data["moves"].append({
                            "original_path": original_path,
                            "new_path": dest_path,
                            "type": "directory",
                            "category": "Directories"
                        })
                    
                    processed += 1
                    progress = processed / total_items
                    progress_bar.value = progress
                    progress_percentage.value = f"{int(progress * 100)}%"
                    progress_text.value = f"Organizing files... ({processed}/{total_items})"
                    
                except Exception as e:
                    print(f"Error processing {item_path}: {e}")
                    continue
            
            # Save rollback information
            with open(rollback_file, 'w') as f:
                json.dump(rollback_data, f, indent=2)
            
            progress_text.value = f"Organization complete! Processed {processed} items. Rollback info saved."
            
            # Add completion message to tree view
            add_to_tree_view(create_tree_node(f"✅ Organization complete! Processed {processed} items.", color=ft.Colors.GREEN_600))
            
            # Show tree view after organization is complete
            tree_container.visible = True
            tree_container.update()
            
            # Update rollback button visibility
            rollback_button.visible = True
            rollback_button.update()
            
        except Exception as e:
            progress_text.value = f"Error during organization: {str(e)}"
            print(f"Error in organize_files: {e}")
    
    def rollback_organization():
        """Rollback file organization to original paths"""
        if not destination_folder:
            return
        
        rollback_file = os.path.join(destination_folder, "rollback_info.json")
        
        if not os.path.exists(rollback_file):
            progress_text.value = "No rollback information found!"
            return
        
        try:
            # Show progress
            progress_container.visible = True
            progress_text.visible = True
            progress_bar.visible = True
            progress_percentage.visible = True
            progress_bar.value = 0
            progress_text.value = "Starting rollback..."
            progress_percentage.value = "0%"
            
            # Load rollback data
            with open(rollback_file, 'r') as f:
                rollback_data = json.load(f)
            
            moves = rollback_data.get("moves", [])
            total_moves = len(moves)
            
            if total_moves == 0:
                progress_text.value = "No moves to rollback."
                return
            
            processed = 0
            successful_rollbacks = 0
            
            for move_info in moves:
                try:
                    original_path = move_info["original_path"]
                    new_path = move_info["new_path"]
                    
                    # Check if the moved file/directory still exists
                    if os.path.exists(new_path):
                        # Create parent directory if it doesn't exist
                        os.makedirs(os.path.dirname(original_path), exist_ok=True)
                        
                        # Move back to original location
                        shutil.move(new_path, original_path)
                        successful_rollbacks += 1
                    
                    processed += 1
                    progress = processed / total_moves
                    progress_bar.value = progress
                    progress_percentage.value = f"{int(progress * 100)}%"
                    progress_text.value = f"Rolling back... ({processed}/{total_moves})"
                    
                except Exception as e:
                    print(f"Error rolling back {move_info}: {e}")
                    continue
            
            # Clean up created directories after rollback
            created_dirs = rollback_data.get("created_directories", [])
            deleted_dirs = 0
            
            for dir_path in created_dirs:
                try:
                    # Check if directory is empty before deleting
                    if os.path.exists(dir_path) and not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        deleted_dirs += 1
                    elif os.path.exists(dir_path):
                        # Directory not empty, try to remove if it only contains empty subdirectories
                        try:
                            os.removedirs(dir_path)  # Removes directory and empty parent directories
                            deleted_dirs += 1
                        except:
                            pass  # If we can't remove, it's not empty or has files
                except Exception as e:
                    print(f"Could not delete directory {dir_path}: {e}")
                    continue
            
            # Remove rollback file after successful rollback
            if successful_rollbacks > 0:
                try:
                    os.remove(rollback_file)
                except:
                    pass  # Don't fail if we can't remove the rollback file
            
            progress_text.value = f"Rollback complete! Restored {successful_rollbacks} items and deleted {deleted_dirs} empty directories."
            
            # Hide rollback button
            rollback_button.visible = False
            rollback_button.update()
            
        except Exception as e:
            progress_text.value = f"Error during rollback: {str(e)}"
            print(f"Error in rollback_organization: {e}")
    
    # Header with banner image
    header = ft.Container(
        content=ft.Image(
            src="reorg.png",
        ),
        border_radius=ft.border_radius.all(20),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        margin=ft.margin.only(left=0, right=0, top=0),
        padding=ft.padding.only(bottom=0),
    )
    
    
    # Source folder selection
    source_display = ft.Text("No source folder selected", size=12, color=ft.Colors.GREY_600)
    
    def on_source_folder_picked(e):
        nonlocal source_folder
        if e.path:
            source_folder = e.path
            source_display.value = f"Source: {source_folder}"
            source_display.update()
    
    source_picker.on_result = on_source_folder_picked
    
    source_section = ft.Container(
        content=ft.Column([
            ft.ElevatedButton(
                "Choose Source",
                icon=ft.Icons.FOLDER_OPEN,
                on_click=lambda _: source_picker.get_directory_path(),
                bgcolor=ft.Colors.GREEN_600,
                color=ft.Colors.WHITE,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=8),
                    padding=ft.padding.symmetric(horizontal=20, vertical=12),
                    elevation=3,
                    shadow_color=ft.Colors.GREEN_800
                )
            ),
            source_display
        ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.padding.only(bottom=20)
    )
    
    # Destination folder selection
    destination_display = ft.Text("No destination folder selected", size=12, color=ft.Colors.GREY_600)
    
    def on_destination_folder_picked(e):
        nonlocal destination_folder
        if e.path:
            destination_folder = e.path
            destination_display.value = f"Destination: {destination_folder}"
            destination_display.update()
    
    destination_picker.on_result = on_destination_folder_picked
    
    destination_section = ft.Container(
        content=ft.Column([
            ft.ElevatedButton(
                "Choose Destination",
                icon=ft.Icons.FOLDER_OPEN,
                on_click=lambda _: destination_picker.get_directory_path(),
                bgcolor=ft.Colors.GREEN_600,
                color=ft.Colors.WHITE,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=8),
                    padding=ft.padding.symmetric(horizontal=20, vertical=12),
                    elevation=3,
                    shadow_color=ft.Colors.GREEN_800
                )
            ),
            destination_display
        ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.padding.only(bottom=20)
    )
    
    # Organize button
    organize_button = ft.ElevatedButton(
        "Organize Files",
        icon=ft.Icons.SORT,
        on_click=lambda e: organize_files(),
        bgcolor=ft.Colors.BLACK,
        color=ft.Colors.WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=ft.padding.symmetric(horizontal=30, vertical=15)
        )
    )
    
    # Rollback button (initially hidden)
    rollback_button = ft.ElevatedButton(
        "Rollback Organization",
        icon=ft.Icons.UNDO,
        on_click=lambda e: rollback_organization(),
        bgcolor=ft.Colors.RED_600,
        color=ft.Colors.WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=ft.padding.symmetric(horizontal=30, vertical=15)
        ),
        visible=False  # Hidden by default, shown after organization
    )
    
    # File categories info
    categories_info = ft.Container(
        content=ft.Column([
            ft.Text("File Categories:", size=14, weight=ft.FontWeight.BOLD),
            ft.Text("• Documents: PDF, Word, Excel, PowerPoint, Text files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Images: JPG, PNG, GIF, SVG, and other image formats", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Videos: MP4, AVI, MOV, and other video formats", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Audio: MP3, WAV, FLAC, and other audio formats", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Archives: ZIP, RAR, 7Z, and other compressed files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Code: Python, JavaScript, HTML, CSS, and other code files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Executables: EXE, MSI, and other executable files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Directories: All subdirectories will be moved to 'Directories' folder", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Other: Files with unrecognized extensions", size=12, color=ft.Colors.GREY_600),
        ], spacing=2),
        padding=ft.padding.all(15),
        bgcolor=ft.Colors.GREY_50,
        border_radius=8,
        border=ft.border.all(1, ft.Colors.GREY_300),
        height=220
    )
    
    # Rollback info section
    rollback_info = ft.Container(
        content=ft.Column([
            ft.Text("Rollback Information:", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_600),
            ft.Text("• Original file paths are preserved during organization", size=12, color=ft.Colors.GREY_600),
            ft.Text("• A rollback_info.json file is created in the destination folder", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Use the 'Rollback Organization' button to restore files to original locations", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Rollback will also delete empty classification directories created during organization", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Rollback button appears after successful organization", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Rollback information is automatically deleted after successful rollback", size=12, color=ft.Colors.GREY_600),
        ], spacing=2),
        padding=ft.padding.all(15),
        bgcolor=ft.Colors.RED_50,
        border_radius=8,
        border=ft.border.all(1, ft.Colors.RED_200),
        height=220
    )
    
    
    # Tree view will be populated when organize button is clicked
    
    organize_content = ft.Column([
        header,
        # All three sections in one line: source, destination, and organize button
        ft.Row([
            source_section,
            destination_section,
            ft.Container(
                content=ft.Row([
                    organize_button,
                    rollback_button
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                padding=ft.padding.only(bottom=20)
            )
        ], alignment=ft.MainAxisAlignment.SPACE_EVENLY, vertical_alignment=ft.CrossAxisAlignment.START, spacing=15),
        progress_container,  # Progress bar only when organizing
        tree_container,  # Tree view appears after organization
        ft.Divider(),
        ft.Row([
            categories_info,
            rollback_info
        ], alignment=ft.MainAxisAlignment.SPACE_EVENLY, spacing=20),
    ], expand=True, scroll=ft.ScrollMode.AUTO)
    
    # Return content and file pickers for overlay
    return organize_content, source_picker, destination_picker

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
                ft.Icon(ft.Icons.SEARCH, color="black"),
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
                ft.Icon(ft.Icons.FOLDER_OPEN, color="black"),
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
                ft.Icon(ft.Icons.EDIT, color="black"),
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
                ft.Icon(ft.Icons.SETTINGS, color="black"),
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
                    padding=ft.padding.only(left=10, right=10, top=10, bottom=0)
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
            content, source_picker, destination_picker = create_organize_content(file_picker)
            # Add file pickers to page overlay
            page.overlay.append(source_picker)
            page.overlay.append(destination_picker)
            return content
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
        padding=20
    )
    
    # Set up folder picker callback
    def on_folder_picked(e: ft.FilePickerResultEvent):
        if e.path:
            global selected_folder
            selected_folder = e.path
            
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
                    
                    # Update the content to refresh folder display (but don't regenerate progress components)
                    if current_tab == "search":
                        # Just update the folder display text, don't recreate the whole content
                        content_container.content = get_content_for_tab(current_tab)
                        content_container.update()
                    
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
