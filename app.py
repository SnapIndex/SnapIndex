import flet as ft
import sys
import os

# Get folder from command line argument
if len(sys.argv) > 1:
    selected_folder = sys.argv[1]
else:
    selected_folder = None

def main(page: ft.Page):
    page.title = "SnapIndex"

    if not selected_folder or not os.path.exists(selected_folder):
        page.add(ft.Text("No folder selected or folder does not exist.", color="red"))
        return

    # List all .pdf files
    pdf_files = [f for f in os.listdir(selected_folder) if f.lower().endswith(".pdf")]

    page.add(ft.Text("Selected Folder:", weight=ft.FontWeight.BOLD))
    page.add(ft.Text(selected_folder, color="blue"))
    page.add(ft.Divider())

    if pdf_files:
        # Display PDFs as a list
        for pdf in pdf_files:
            page.add(ft.Text(pdf))
    else:
        page.add(ft.Text("No PDF files found in this folder.", italic=True, color="gray"))

ft.app(target=main)
