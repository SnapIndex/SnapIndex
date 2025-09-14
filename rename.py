import os
import re
import threading
import traceback
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

import flet as ft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from file_loader import find_documents, extract_text_from_file, SUPPORTED_EXTENSIONS


# -------------------------
# LLM configuration
# -------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class LLMHelper:
    """Lazy loader wrapper around HF Transformers model used for filename suggestions."""

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
                print(f"[LLM] Loaded {self.model_id} on {device}", flush=True)
            except Exception:
                pass
        except Exception as e:
            # Mark as failed to avoid repeated attempts
            try:
                print(f"[LLM][error] Load failed: {e}", flush=True)
                traceback.print_exc()
            except Exception:
                pass
            self._load_failed = True

    def suggest_filename(self, original_name_without_ext: str, extension: str, context: str) -> str:
        """Return a concise filename suggestion (without extension)."""
        self._ensure_loaded()
        if self._load_failed or self._tokenizer is None or self._model is None:
            return self._fallback_suggestion(original_name_without_ext, context)

        system_msg = (
            "You are a helpful assistant that renames files. Return ONLY the new file name, "
            "without any file extension. Keep it short (5-10 words), Title Case, descriptive, "
            "avoid special characters other than spaces and hyphens."
        )
        user_msg = (
            f"Original name: {original_name_without_ext}\n"
            f"Extension: {extension}\n"
            f"Content snippet (short):\n{context[:800]}\n\n"
            "Propose a single, human-friendly filename. Return only the filename text."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            # Apply chat template if available
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
                    max_new_tokens=64,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                )

            text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            # Take first line and sanitize
            candidate = text.splitlines()[0].strip() if text.strip() else original_name_without_ext
            if candidate.strip().lower() in {"none", "n/a", "unknown"}:
                candidate = original_name_without_ext
            try:
                print(f"[LLM] generated: {candidate}", flush=True)
            except Exception:
                pass
            return sanitize_filename_base(candidate) or original_name_without_ext
        except Exception as e:
            try:
                print(f"[LLM][error] Generation failed: {e}", flush=True)
                traceback.print_exc()
            except Exception:
                pass
            self._load_failed = True
            return self._fallback_suggestion(original_name_without_ext, context)

    def _fallback_suggestion(self, original_name_without_ext: str, context: str) -> str:
        # Build from context first words if possible
        ctx = context.strip()
        if ctx:
            words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", ctx)
            if words:
                title = " ".join(words[:8]).title()
                return sanitize_filename_base(title) or original_name_without_ext
        return sanitize_filename_base(original_name_without_ext) or original_name_without_ext


# -------------------------
# Utilities
# -------------------------
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename_base(name: str) -> str:
    """Sanitize a filename base (without extension) for Windows safety.

    - Remove illegal chars: \\/:*?"<>|
    - Collapse whitespace, trim
    - Avoid trailing dots/spaces
    - Limit length to 120 chars
    """
    # Remove illegal characters
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Remove trailing dots/spaces
    name = name.rstrip(" .")
    # Avoid reserved device names
    if name.upper() in WINDOWS_RESERVED_NAMES:
        name = f"{name}_file"
    # Limit length
    if len(name) > 120:
        name = name[:120].rstrip()
    return name


def ensure_unique_path(directory: str, base: str, ext: str) -> str:
    """Ensure the final path is unique by appending a counter if needed."""
    candidate = os.path.join(directory, f"{base}{ext}")
    if not os.path.exists(candidate):
        return candidate
    count = 1
    while True:
        alt = os.path.join(directory, f"{base} ({count}){ext}")
        if not os.path.exists(alt):
            return alt
        count += 1


def get_first_paragraph(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    # Split by blank line first; fallback to first line
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    para = paragraphs[0] if paragraphs else text.strip().splitlines()[0] if text.strip().splitlines() else ""
    return para[:max_chars]


def extract_short_context_for_file(file_path: str, file_name: str) -> str:
    file_info = {
        "path": file_path,
        "name": file_name,
        "extension": os.path.splitext(file_name)[1].lower(),
        "type": None,
    }
    try:
        # Determine type from extension mapping for proper routing
        file_info["type"] = SUPPORTED_EXTENSIONS.get(file_info["extension"])  # may be None
        # Let file_loader figure out the chunks based on type
        chunks = extract_text_from_file(file_info)
        # Prefer non-filename content
        priority = [
            "image_description",
            "image_content",
            "content",
            "image_searchable",
            "image_possible",
        ]
        # First pass: preferred chunk types
        for p in priority:
            for chunk in chunks:
                if chunk.get("chunk_type") == p:
                    txt = (chunk.get("text") or "").strip()
                    if txt:
                        return get_first_paragraph(txt)
        # Second pass: any non-filename chunk with text
        for chunk in chunks:
            if chunk.get("chunk_type") != "filename":
                txt = (chunk.get("text") or "").strip()
                if txt:
                    return get_first_paragraph(txt)
    except Exception:
        pass
    # Fallback to filename without extension
    return os.path.splitext(file_name)[0]


def log_suggestion_line(original_name: str, suggested_base: str, ext: str) -> None:
    """Log suggestion to console (flushed) and to a local file for visibility when no console is attached."""
    line = f"[Suggest] {original_name} -> {suggested_base}{ext}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open("rename_suggestions.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {line}\n")
    except Exception:
        pass


# -------------------------
# Data structures for UI state
# -------------------------
@dataclass
class FileEntry:
    path: str
    name: str
    ext: str
    checkbox: ft.Checkbox
    suggestion_field: ft.TextField


# -------------------------
# Flet App
# -------------------------
def main(page: ft.Page) -> None:
    page.title = "SnapIndex - Rename"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1000
    page.window_height = 700
    page.padding = 0

    try:
        page.window_icon = "logo-dark.svg"
    except Exception:
        pass

    entries: List[FileEntry] = []
    selected_folder: Optional[str] = None
    llm = LLMHelper()
    current_tab = "rename"

    # Header
    header = ft.Container(
        content=ft.Image(src="banner.png"),
        border_radius=ft.border_radius.all(20),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        margin=ft.margin.only(left=0, right=0, top=-50),
        padding=ft.padding.only(bottom=0),
    )

    # Progress widgets
    progress_bar = ft.ProgressBar(value=0, visible=False, width=500, color=ft.Colors.ORANGE, bgcolor=ft.Colors.GREY_300, bar_height=12)
    progress_text = ft.Text("Processing...", visible=False, color=ft.Colors.ORANGE, size=16, weight=ft.FontWeight.BOLD)
    progress_percentage = ft.Text("0%", visible=False, color=ft.Colors.ORANGE, size=18, weight=ft.FontWeight.BOLD)
    progress_container = ft.Container(
        content=ft.Column([
            progress_text,
            ft.Row([progress_bar, progress_percentage], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
        visible=False,
        padding=25,
        bgcolor=ft.Colors.WHITE,
        border_radius=12,
        border=ft.border.all(2, ft.Colors.ORANGE),
        shadow=ft.BoxShadow(spread_radius=2, blur_radius=8, color=ft.Colors.GREY_400, offset=ft.Offset(0, 2)),
    )

    # Folder picker
    folder_display = ft.Text("No folder selected", size=12, color=ft.Colors.GREY_600)
    picker = ft.FilePicker()
    page.overlay.append(picker)

    def on_folder_picked(e: ft.FilePickerResultEvent) -> None:
        nonlocal selected_folder
        if not e.path:
            return
        selected_folder = e.path
        folder_display.value = f"Selected: {selected_folder}"
        folder_display.update()
        load_files()

    picker.on_result = on_folder_picked

    choose_btn = ft.ElevatedButton(
        "Choose Folder",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=lambda _: picker.get_directory_path(),
        bgcolor=ft.Colors.GREEN_600,
        color=ft.Colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8), padding=ft.padding.symmetric(horizontal=20, vertical=12), elevation=3, shadow_color=ft.Colors.GREEN_800),
    )

    folder_section = ft.Container(
        content=ft.Row([choose_btn, folder_display], alignment=ft.MainAxisAlignment.START, spacing=10),
        padding=ft.padding.only(bottom=20),
    )

    # Two-column lists
    left_list = ft.Column(spacing=8, expand=True)
    right_list = ft.Column(spacing=8, expand=True)

    lists_container = ft.Container(
        content=ft.Row([
            ft.Container(content=ft.Column([ft.Text("Original Filename", weight=ft.FontWeight.BOLD), ft.Divider(), left_list], expand=True), expand=True, padding=10, border=ft.border.all(1, ft.Colors.GREY_300), border_radius=8, bgcolor=ft.Colors.GREY_50),
            ft.Container(content=ft.Column([ft.Text("Suggested Filename (editable)", weight=ft.FontWeight.BOLD), ft.Divider(), right_list], expand=True), expand=True, padding=10, border=ft.border.all(1, ft.Colors.GREY_300), border_radius=8, bgcolor=ft.Colors.GREY_50),
        ], spacing=15),
        expand=True,
        visible=False,
    )

    # Buttons
    def set_progress(visible: bool, value: float = 0.0, text: str = "Processing...") -> None:
        progress_container.visible = visible
        progress_text.visible = visible
        progress_bar.visible = visible
        progress_percentage.visible = visible
        progress_bar.value = value
        progress_text.value = text
        progress_percentage.value = f"{int(value * 100)}%"
        page.update()

    def load_files() -> None:
        left_list.controls.clear()
        right_list.controls.clear()
        entries.clear()
        if not selected_folder:
            page.update()
            return
        docs = find_documents(selected_folder)
        for doc in docs:
            file_path = doc["path"]
            file_name = doc["name"]
            base, ext = os.path.splitext(file_name)
            checkbox = ft.Checkbox(label="", value=False)
            suggest_field = ft.TextField(value="", hint_text=f"Suggested for {base}", expand=True)
            left_list.controls.append(ft.Row([checkbox, ft.Text(file_name)], spacing=10))
            right_list.controls.append(suggest_field)
            entries.append(FileEntry(path=file_path, name=file_name, ext=ext, checkbox=checkbox, suggestion_field=suggest_field))
        page.update()
        # Auto-generate suggestions after loading
        generate_suggestions_async()

    def generate_suggestions_async() -> None:
        if not entries:
            return
        # Hide lists while generating
        lists_container.visible = False
        page.update()

        def worker() -> None:
            set_progress(True, 0.0, "Generating suggestions...")
            # Try loading LLM once at start
            llm._ensure_loaded()
            if llm._load_failed:
                try:
                    print("[LLM] Using fallback suggestions (model unavailable)", flush=True)
                except Exception:
                    pass
            total = len(entries)
            done = 0
            for entry in entries:
                try:
                    # Always attempt suggestion generation; user can choose which to rename later
                    base = os.path.splitext(entry.name)[0]
                    context = extract_short_context_for_file(entry.path, entry.name)
                    suggestion_base = llm.suggest_filename(base, entry.ext, context)
                    entry.suggestion_field.value = f"{suggestion_base}"
                    # Print to console
                    log_suggestion_line(entry.name, suggestion_base, entry.ext)
                except Exception:
                    pass
                finally:
                    done += 1
                    progress_bar.value = done / max(total, 1)
                    progress_percentage.value = f"{int(progress_bar.value * 100)}%"
                    page.update()
            set_progress(False)
            # Reveal lists after generation completes
            lists_container.visible = True
            page.update()

        threading.Thread(target=worker, daemon=True).start()

    def on_suggest_click(e) -> None:
        generate_suggestions_async()

    suggest_btn = ft.ElevatedButton(
        "Suggest Names",
        icon=ft.Icons.SMART_TOY,
        on_click=on_suggest_click,
        bgcolor=ft.Colors.ORANGE,
        color=ft.Colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
    )

    def on_confirm_click(e) -> None:
        if not entries:
            # Close app even if nothing to do
            try:
                page.window_close()
            except Exception:
                pass
            try:
                page.window_destroy()
            except Exception:
                pass
            return
        # Hide lists while renaming
        lists_container.visible = False
        set_progress(True, 0.0, "Renaming files...")
        page.update()

        renamed = 0
        selected_entries = [en for en in entries if en.checkbox.value]
        if not selected_entries:
            try:
                print("[Rename] No items selected; defaulting to all files.", flush=True)
            except Exception:
                pass
            selected_entries = list(entries)
        total_to_process = len(selected_entries)
        processed = 0
        for entry in selected_entries:
            try:
                suggestion = (entry.suggestion_field.value or "").strip()
                if not suggestion:
                    continue
                base = sanitize_filename_base(suggestion)
                if not base:
                    continue
                directory = os.path.dirname(entry.path)
                final_path = ensure_unique_path(directory, base, entry.ext)
                # Log planned operation
                try:
                    print(f"[Rename] Attempt: {entry.path} -> {final_path}", flush=True)
                except Exception:
                    pass
                # Handle case-only rename on Windows
                if os.path.abspath(final_path).lower() == os.path.abspath(entry.path).lower() and os.path.abspath(final_path) != os.path.abspath(entry.path):
                    temp_base = f"{base} __tmp__"
                    temp_path = ensure_unique_path(directory, temp_base, entry.ext)
                    try:
                        os.replace(entry.path, temp_path)
                        os.replace(temp_path, final_path)
                    except Exception:
                        try:
                            os.rename(entry.path, temp_path)
                            os.rename(temp_path, final_path)
                        except Exception as inner_ex:
                            try:
                                print(f"[Rename][error] Case-change workaround failed: {inner_ex}", flush=True)
                            except Exception:
                                pass
                            raise
                elif os.path.abspath(final_path) == os.path.abspath(entry.path):
                    processed += 1
                    progress_bar.value = (processed / max(total_to_process, 1))
                    progress_percentage.value = f"{int(progress_bar.value * 100)}%"
                    page.update()
                    continue
                try:
                    os.replace(entry.path, final_path)
                except Exception:
                    os.rename(entry.path, final_path)
                # Update UI and state
                log_suggestion_line(entry.name, os.path.splitext(os.path.basename(final_path))[0], entry.ext)
                print(f"[Rename] {entry.name} -> {os.path.basename(final_path)}", flush=True)
                # Verify existence
                try:
                    if os.path.exists(final_path):
                        print(f"[Rename] Success: {final_path}", flush=True)
                    else:
                        print(f"[Rename][warn] Target not found after rename: {final_path}", flush=True)
                except Exception:
                    pass
                entry.path = final_path
                entry.name = os.path.basename(final_path)
                renamed += 1
            except Exception as ex:
                try:
                    print(f"[Rename][error] {entry.name}: {ex}", flush=True)
                except Exception:
                    pass
            finally:
                processed += 1
                progress_bar.value = (processed / max(total_to_process, 1))
                progress_percentage.value = f"{int(progress_bar.value * 100)}%"
                page.update()

        set_progress(False)
        try:
            print(f"[Rename] Completed. Renamed {renamed} of {total_to_process} files.", flush=True)
        except Exception:
            pass
        # Close the app after confirming, regardless of rename count
        try:
            page.window_close()
        except Exception:
            pass
        try:
            page.window_destroy()
        except Exception:
            pass

    confirm_btn = ft.ElevatedButton(
        "Confirm",
        icon=ft.Icons.CHECK_CIRCLE,
        on_click=on_confirm_click,
        bgcolor=ft.Colors.GREEN,
        color=ft.Colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8), padding=ft.padding.symmetric(horizontal=30, vertical=15)),
    )

    buttons_row = ft.Row([
        suggest_btn,
        ft.Container(expand=True),  # spacer
        confirm_btn,
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    root = ft.Column([
        header,
        folder_section,
        progress_container,
        lists_container,
        ft.Divider(),
        buttons_row,
    ], expand=True, scroll=ft.ScrollMode.AUTO, spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    # Sidebar (navbar) similar styling to main app
    def create_sidebar() -> ft.Container:
        def show_info(_: ft.ControlEvent) -> None:
            page.snack_bar = ft.SnackBar(ft.Text("Open the main app for Search/Organize/Settings."))
            page.snack_bar.open = True
            page.update()

        def button(label: str, icon: str, tab_name: str) -> ft.Container:
            return ft.Container(
                content=ft.Row([
                    ft.Icon(icon, color="black"),
                    ft.Text(label, weight=ft.FontWeight.BOLD if tab_name == current_tab else ft.FontWeight.NORMAL),
                ]),
                bgcolor="#f5f7fa" if tab_name == current_tab else "white",
                padding=8,
                on_click=show_info if tab_name != "rename" else None,
                data=tab_name,
            )

        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([
                        ft.Image(src="logo-dark.svg", width=24, height=24, fit=ft.ImageFit.CONTAIN),
                        ft.Text("SnapIndex", size=20, weight=ft.FontWeight.BOLD),
                    ]),
                    padding=ft.padding.only(left=10, right=10, top=10, bottom=0),
                ),
                ft.Divider(),
                button("Search", ft.Icons.SEARCH, "search"),
                button("Organize", ft.Icons.FOLDER_OPEN, "organize"),
                button("Rename", ft.Icons.EDIT, "rename"),
                button("Settings", ft.Icons.SETTINGS, "settings"),
            ]),
            width=200,
            bgcolor="white",
            border=ft.border.only(right=ft.BorderSide(1, "gray")),
        )

    sidebar = create_sidebar()
    content_container = ft.Container(content=root, expand=True, padding=20)

    page.add(ft.Row([sidebar, content_container], expand=True))


if __name__ == "__main__":
    ft.app(target=main)


