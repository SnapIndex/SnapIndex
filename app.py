import flet as ft
import sys
import os
from doc_loader import find_pdfs, extract_pdf_text

# Hard-coded folder location
selected_folder = "C:\\Users\\akarsh\\Downloads"

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
                result_card = ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
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
        
        page.update()

    # Create the main layout
    page.add(
        ft.Column([
            # Header
            ft.Row([
                ft.Text("SnapIndex PDF Search", size=24, weight=ft.FontWeight.BOLD)
            ]),
            ft.Text(f"Searching in: {selected_folder}", color="gray"),
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
