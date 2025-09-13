import flet as ft
import os
from doc_loader import find_pdfs, extract_pdf_text

# Global variables
selected_folder = "C:\\Users\\akarsh\\Downloads"
current_tab = "search"

def search_in_pdfs(query, folder_path):
    """Search for query text in all PDFs in the folder"""
    results = []
    pdf_files = find_pdfs(folder_path)
    
    for pdf_path in pdf_files:
        try:
            for page_data in extract_pdf_text(pdf_path):
                text = page_data["text"].lower()
                if query.lower() in text:
                    # Find the position of the query in the text
                    query_pos = text.find(query.lower())
                    start = max(0, query_pos - 100)
                    end = min(len(page_data["text"]), query_pos + len(query) + 100)
                    snippet = page_data["text"][start:end]
                    
                    results.append({
                        "file": page_data["pdf_name"],
                        "page": page_data["page_number"],
                        "snippet": snippet,
                        "full_path": pdf_path
                    })
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    return results

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
                # Create result card
                result_card = ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.PICTURE_AS_PDF, color="red"),
                                ft.Text(
                                    f"{result['file']} - Page {result['page']}",
                                    weight=ft.FontWeight.BOLD,
                                    color="blue"
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
        
        results_container.update()
    
    # File picker will be added to page overlay in main function
    
    # Header
    header = ft.Container(
        content=ft.Column([
            ft.Text("Semantic File Search", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800),
            ft.Text("Search through your documents with intelligent semantic matching", size=14, color=ft.Colors.GREY_600),
            ft.Divider(height=1, color=ft.Colors.BLUE_200)
        ], spacing=8),
        padding=ft.padding.only(bottom=20)
    )
    
    # Folder selection section
    folder_display = ft.Text(f"Selected: {selected_folder}", size=12, color=ft.Colors.GREY_600)
    
    folder_section = ft.Container(
        content=ft.Column([
            ft.Text("Select Folder to Search", size=16, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.ElevatedButton(
                    "Choose Folder",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=lambda _: file_picker.get_directory_path(),
                    bgcolor=ft.Colors.BLUE,
                    color=ft.Colors.WHITE
                ),
                folder_display
            ], alignment=ft.MainAxisAlignment.START, spacing=10)
        ], spacing=8),
        padding=ft.padding.only(bottom=20)
    )
    
    # Search section
    search_field = ft.TextField(
        label="Search Query",
        hint_text="Enter your search query...",
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
        scroll=ft.ScrollMode.AUTO,
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
    ], expand=True)

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
