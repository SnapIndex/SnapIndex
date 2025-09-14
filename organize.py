import flet as ft
import os
import sys
import shutil
import json
from datetime import datetime
import re
import traceback
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from file_loader import extract_text_from_file, SUPPORTED_EXTENSIONS

# -------------------------
# LLM for genre classification (Qwen)
# -------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Fixed set of document genres
GENRE_LABELS: List[str] = [
    "History", "Geography", "Statistics", "Mathematics", "Physics", "Chemistry", "Biology",
    "Computer Science", "Programming", "Data Science", "Machine Learning", "Artificial Intelligence",
    "Economics", "Finance", "Business", "Marketing", "Management",
    "Law", "Politics", "Government",
    "Literature", "Philosophy", "Psychology", "Sociology",
    "Art", "Music",
    "Engineering", "Technology",
    "Medicine", "Health",
    "Education",
    "Environmental Science", "Earth Science", "Astronomy",
    "Travel", "Sports",
    "Research Paper", "Report", "Manual", "How-To", "Tutorial",
    "Presentation", "Resume", "Invoice", "Contract", "Legal",
    "Cooking", "Recipe",
    "Other",
]


class GenreLLM:
    """Lazy-loaded LLM wrapper to classify document context into a fixed genre set."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._load_failed: bool = False

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self._load_failed:
            return
        try:
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="auto",
            )
            if device == "cpu":
                model = model.to("cpu")

            self._tokenizer = tokenizer
            self._model = model
            try:
                print(f"[GenreLLM] Loaded {self.model_id} on {device}", flush=True)
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"[GenreLLM][error] Load failed: {e}", flush=True)
                traceback.print_exc()
            except Exception:
                pass
            self._load_failed = True

    def classify_genre(self, context: str) -> str:
        """Return a single genre label from GENRE_LABELS for the given context."""
        self._ensure_loaded()
        if self._load_failed or self._tokenizer is None or self._model is None:
            return self._fallback_classification(context)

        genres_str = ", ".join(GENRE_LABELS)
        system_msg = (
            "You are a strict document classifier. Choose EXACTLY ONE genre label from the provided list. "
            "Respond with only the label text, no punctuation, no explanation."
        )
        user_msg = (
            f"Genres: {genres_str}\n\n"
            f"Document excerpt (short):\n{(context or '').strip()[:1600]}\n\n"
            "Return exactly one label from the list above."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            tokenizer = self._tokenizer
            model = self._model
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            else:
                inputs = tokenizer(user_msg, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=16,
                    temperature=0.0,
                    do_sample=False,
                )

            text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            candidate = (text.splitlines()[0] if text.strip() else "").strip()
            # Normalize and validate against allowed labels
            normalized = candidate
            # Exact match first
            if normalized in GENRE_LABELS:
                return normalized
            # Try loose matching (case-insensitive)
            for g in GENRE_LABELS:
                if normalized.lower() == g.lower():
                    return g
            return "Other"
        except Exception as e:
            try:
                print(f"[GenreLLM][error] Classification failed: {e}", flush=True)
                traceback.print_exc()
            except Exception:
                pass
            self._load_failed = True
            return self._fallback_classification(context)

    def _fallback_classification(self, context: str) -> str:
        # Quick heuristics as a last resort
        txt = (context or "").lower()
        if any(k in txt for k in ["algorithm", "code", "python", "java", "programming"]):
            return "Programming"
        if any(k in txt for k in ["statistic", "regression", "probability", "mean", "variance"]):
            return "Statistics"
        if any(k in txt for k in ["history", "ancient", "medieval", "revolution", "war"]):
            return "History"
        if any(k in txt for k in ["economics", "market", "finance", "inflation", "gdp"]):
            return "Economics"
        return "Other"


def extract_document_context(file_path: str, file_name: str, max_chars: int = 4000) -> str:
    """Extract a concise context snippet (about 1-2 pages) for LLM classification."""
    try:
        ext = os.path.splitext(file_name)[1].lower()
        file_type = SUPPORTED_EXTENSIONS.get(ext)
        file_info = {
            "path": file_path,
            "name": file_name,
            "extension": ext,
            "type": file_type,
        }
        chunks = extract_text_from_file(file_info)
        # Prefer 'content' chunks and take the first few until max_chars
        content_texts: List[str] = []
        for ch in chunks:
            if ch.get("chunk_type") == "content" and ch.get("text"):
                content_texts.append(ch["text"].strip())
        # Fallbacks: any non-filename chunk
        if not content_texts:
            for ch in chunks:
                if ch.get("chunk_type") != "filename" and ch.get("text"):
                    content_texts.append(ch["text"].strip())
        combined = "\n\n".join(content_texts)
        return combined[:max_chars] if combined else os.path.splitext(file_name)[0]
    except Exception:
        return os.path.splitext(file_name)[0]

# Function to get resource path for bundled files
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Default folder location (same as in app.py)
DEFAULT_FOLDER = "C:\\Users\\akarsh\\Downloads"

# Global variable for current tab
current_tab = "organize"

def create_organize_content(source_folder, file_picker, default_folder):
    """Create organize content with pre-set source folder"""
    # Global variables for source and destination folders
    destination_folder = ""
    # Initialize Genre LLM lazily
    genre_llm = GenreLLM()
    
    # Create file picker for destination only
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
    
    # Tree view for showing file moves (force scrollbar visible)
    tree_view = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        spacing=2,
        height=420,
        controls=[ft.Text("📁 Source folder is pre-selected. Choose destination folder and click 'Organize Files' to begin", 
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
                height=420
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

                        # For documents, classify into a genre and create subfolder
                        display_category = category
                        if category == "Documents":
                            try:
                                # Extract ~1-2 pages of context and classify
                                context_snippet = extract_document_context(item_path, item_name)
                                predicted_genre = genre_llm.classify_genre(context_snippet) or "Other"
                                if predicted_genre not in GENRE_LABELS:
                                    predicted_genre = "Other"
                                # Use genre subfolder
                                category_dir = os.path.join(destination_folder, "Documents", predicted_genre)
                                display_category = f"Documents/{predicted_genre}"
                            except Exception as e:
                                try:
                                    print(f"[Genre] Classification failed for {item_name}: {e}", flush=True)
                                except Exception:
                                    pass
                                # Fallback: keep regular Documents folder
                                predicted_genre = "Other"
                        
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
                        
                        # Update tree view (use genre-aware label if available)
                        update_tree_view_with_move(original_path, dest_path, display_category)
                        
                        # Record the move for rollback
                        rollback_data["moves"].append({
                            "original_path": original_path,
                            "new_path": dest_path,
                            "type": "file",
                            "category": display_category
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
    banner_path = resource_path("reorg.png")
    header = ft.Container(
        content=ft.Image(
            src=banner_path if os.path.exists(banner_path) else None,
            fit=ft.ImageFit.COVER if os.path.exists(banner_path) else None,
        ) if os.path.exists(banner_path) else ft.Container(
            content=ft.Column([
                ft.Text("Organize Files", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE),
                ft.Text("Automatically organize files by category", size=14, color=ft.Colors.GREY_600)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20
        ),
        border_radius=ft.border_radius.all(15),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        margin=ft.margin.only(left=-20, right=-20, top=-20),
        padding=ft.padding.only(bottom=10),
    )
    
    # Source folder display (read-only since it's pre-set)
    source_display = ft.Text(f"Source: {source_folder}", size=12, color=ft.Colors.GREEN_600, weight=ft.FontWeight.BOLD)
    
    # Determine if using default folder or provided folder
    is_default_folder = source_folder == default_folder
    section_title = "Source Folder (Default - Downloads)" if is_default_folder else "Source Folder (Pre-selected)"
    
    source_section = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.FOLDER, color=ft.Colors.GREEN_600),
            source_display,
            ft.Text("💡 Tip: You can also run 'python organize.py <folder_path>' to organize a specific folder", 
                    size=10, color=ft.Colors.GREY_500, italic=True) if is_default_folder else ft.Text("")
        ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.padding.only(bottom=20),
        bgcolor=ft.Colors.GREEN_50,
        border_radius=8,
        border=ft.border.all(1, ft.Colors.GREEN_200)
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
    
    # Return content and file picker for overlay
    return organize_content, destination_picker

def get_content_for_tab(tab_name, organize_content=None):
    """Get content for the specified tab"""
    if tab_name == "organize":
        return organize_content if organize_content else create_organize_content("", None, DEFAULT_FOLDER)[0]
    else:
        return organize_content if organize_content else create_organize_content("", None, DEFAULT_FOLDER)[0]

def main(page: ft.Page):
    """Main function for organize.py - organize files from pre-selected folder"""
    page.title = "SnapIndex - Organize Files"
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
    
    # Get source folder from command line arguments or use default
    if len(sys.argv) >= 2:
        # Source folder provided via command line (from right-click context)
        source_folder = sys.argv[1]
        print(f"Using provided folder: {source_folder}")
    else:
        # No source folder provided, use default Downloads folder
        source_folder = DEFAULT_FOLDER
        print(f"Using default folder: {DEFAULT_FOLDER}")
    
    # Validate source folder
    if not os.path.exists(source_folder) or not os.path.isdir(source_folder):
        error_content = ft.Column([
            ft.Text("Error: Invalid source folder", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.RED),
            ft.Text(f"Path: {source_folder}", size=12, color=ft.Colors.GREY_600),
            ft.Text("Please ensure the folder exists and try again.", size=12, color=ft.Colors.GREY_600),
            ft.Text("Usage: python organize.py [folder_path]", size=12, color=ft.Colors.GREY_600),
            ft.Text("If no folder is provided, Downloads folder will be used as default.", size=12, color=ft.Colors.GREY_600)
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        page.add(error_content)
        return
    
    # Create file picker
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)
    
    # Create organize content with pre-set source folder
    organize_content, destination_picker = create_organize_content(source_folder, file_picker, DEFAULT_FOLDER)
    
    # Add destination picker to page overlay
    page.overlay.append(destination_picker)
    
    # Create sidebar
    def create_sidebar():
        def on_tab_change(tab_name):
            global current_tab
            current_tab = tab_name
            
            # Update button styles
            for btn in sidebar_buttons:
                btn.bgcolor = "white" if btn.data != tab_name else "#f5f7fa"
                # Update text weight
                if hasattr(btn.content, 'controls') and len(btn.content.controls) > 1:
                    btn.content.controls[1].weight = ft.FontWeight.BOLD if btn.data == tab_name else ft.FontWeight.NORMAL
            
            # Update content
            content_container.content = get_content_for_tab(tab_name, organize_content)
            content_container.update()
            page.update()
        
        sidebar_buttons = []
        
        # Organize button (highlighted as current)
        organize_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.FOLDER_OPEN, color="black"),
                ft.Text("Organize", weight=ft.FontWeight.BOLD)
            ]),
            bgcolor="#f5f7fa",
            padding=8,
            on_click=lambda e: on_tab_change("organize"),
            data="organize"
        )
        sidebar_buttons.append(organize_btn)
        
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
    
    # Create main layout
    sidebar = create_sidebar()
    content_container = ft.Container(
        content=get_content_for_tab(current_tab, organize_content),
        expand=True,
        padding=20
    )
    
    page.add(
        ft.Row([
            sidebar,
            content_container
        ], expand=True)
    )

if __name__ == "__main__":
    ft.app(target=main)
