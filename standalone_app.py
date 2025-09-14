import flet as ft
import os
import subprocess
import hashlib
import platform
import shutil
import json
import re
import threading
import traceback
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from simple_faiss import create_faiss_db, search_faiss_db, delete_faiss_db
from file_loader import find_documents, extract_text_from_file, SUPPORTED_EXTENSIONS

# Import the simplified LangChain agent for advanced chat functionality
try:
    from simple_langchain_agent import initialize_agent, chat_with_agent, get_search_results_for_ui
    LANGCHAIN_AVAILABLE = True
    print("✅ Simple LangChain agent available")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"❌ Simple LangChain agent not available: {e}")

# Try to import Ollama for local LLM, fallback to requests for API calls
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not available, using fallback command parsing")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Requests not available for API calls")

# Global variables
selected_folder = "C:\\Users\\akarsh\\Downloads"
current_tab = "search"
db_path = "./faiss_database"

# LLM Configuration for advanced chat functionality
LLM_MODEL = "llama3.2:3b"  # Lightweight model for Windows ARM compatibility
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL

# Initialize LangChain agent
agent_initialized = False
if LANGCHAIN_AVAILABLE:
    try:
        agent_initialized = initialize_agent()
        if agent_initialized:
            print("✅ Simple SnapIndex agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize simple SnapIndex agent: {e}")
        agent_initialized = False

# -------------------------
# LLM configuration for rename functionality
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
# Utilities for rename functionality
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
# Data structures for rename UI state
# -------------------------
@dataclass
class FileEntry:
    path: str
    name: str
    ext: str
    checkbox: ft.Checkbox
    suggestion_field: ft.TextField

# File extension categories - shared across the application
# Order matters: more specific categories should come first to avoid conflicts
FILE_CATEGORIES = {
    # 3D Models first (to catch .obj before Executables)
    '3D_Models': [
        '.obj', '.fbx', '.dae', '.3ds', '.blend', '.max', '.ma', '.mb', '.x3d', 
        '.wrl', '.ply', '.stl', '.off', '.3dm', '.3dmf', '.x', '.ase', 
        '.dxf', '.ifc', '.nff', '.smd', '.vta', '.mdl', '.md2', '.md3', '.pk3', 
        '.mdc', '.md5', '.iqm', '.b3d', '.q3d', '.q3s', '.ter', '.hmp', '.ndo'
    ],
    # Executables second (to catch .dmg, .bat, .sh, .msi before other categories)
    'Executables': [
        '.exe', '.msi', '.dmg', '.app', '.bat', '.sh', '.cmd', 
        '.com', '.scr', '.pif', '.run', '.bin', '.appimage', '.snap', '.flatpak', 
        '.apk', '.ipa', '.xap', '.msix', '.appx', '.pyc', '.pyo', '.so', '.dll', 
        '.dylib', '.a', '.lib', '.o', '.elf'
    ],
    # Code third (to catch .xml, .json, .yaml, .ini, .cfg, .conf before Documents)
    'Code': [
        '.py', '.js', '.html', '.htm', '.css', '.java', '.cpp', '.c', '.h', '.hpp', 
        '.php', '.rb', '.go', '.rs', '.sql', '.ts', '.tsx', '.jsx', '.vue', '.svelte', 
        '.scss', '.sass', '.less', '.styl', '.coffee', '.dart', '.swift', '.kt', 
        '.scala', '.clj', '.hs', '.ml', '.fs', '.vb', '.cs', '.asm', '.s', '.m', 
        '.mm', '.pl', '.pm', '.r', '.jl', '.lua', '.tcl', '.bash', '.zsh', 
        '.fish', '.ps1', '.psm1', '.psd1', '.vbs', '.reg', '.ini', 
        '.cfg', '.conf', '.yaml', '.yml', '.json', '.xml', '.toml', '.dockerfile', 
        '.makefile', '.cmake', '.gradle', '.maven', '.pom', '.sbt', '.cabal', 
        '.haskell', '.elm', '.ex', '.exs', '.erl', '.hrl', '.fsx', '.fsi', '.fs', 
        '.f90', '.f95', '.f03', '.f08', '.ada', '.adb', '.ads', '.nim', '.cr', 
        '.crystal', '.zig', '.odin', '.v', '.pas', '.pp', '.dpr', '.dfm', '.lfm'
    ],
    # Archives fourth (to catch .jar, .war, .ear before Executables)
    'Archives': [
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.iso', 
        '.pkg', '.deb', '.cab', '.arj', '.lzh', '.ace', '.z', 
        '.cpio', '.shar', '.lbr', '.mar', '.s7z', '.alz', '.arc', 
        '.b1', '.ba', '.bh', '.car', '.cfs', '.cpt', '.dar', '.dd', 
        '.dgc', '.ear', '.gca', '.ha', '.hki', '.ice', '.jar', '.kgb', 
        '.lbr', '.lha', '.lzh', '.lzx', '.pak', '.partimg', '.pea', '.pim', 
        '.pit', '.qda', '.rk', '.sda', '.sea', '.sen', '.sfx', '.shk', 
        '.sit', '.sitx', '.sqx', '.tbz2', '.tgz', '.tlz', '.txz', '.tz', 
        '.uc2', '.uha', '.uue', '.war', '.wim', '.xar', '.xx', '.yz1', 
        '.zipx', '.zoo', '.zpaq', '.zz'
    ],
    # Documents fifth (catch-all for text-based files)
    'Documents': [
        '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx', 
        '.csv', '.md', '.tex', '.pages', '.numbers', '.key', '.odp', '.ods', '.odg', 
        '.epub', '.mobi', '.azw', '.azw3', '.fb2', '.lit', '.lrf', '.pdb', '.prc', 
        '.tcr', '.trc', '.xps', '.oxps', '.djvu', '.djv', '.chm', '.hlp', '.inf', 
        '.log'
    ],
    # Images
    'Images': [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg', '.webp', 
        '.ico', '.psd', '.ai', '.eps', '.raw', '.cr2', '.nef', '.orf', '.sr2', 
        '.arw', '.dng', '.heic', '.heif', '.avif', '.jfif', '.pbm', '.pgm', '.ppm', 
        '.pnm', '.xbm', '.xpm', '.pcx', '.tga', '.exr', '.hdr', '.pic', '.pict'
    ],
    # Videos
    'Videos': [
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.3gp', 
        '.ogv', '.mpg', '.mpeg', '.m2v', '.m4p', '.m4b', '.3g2', '.asf', '.rm', 
        '.rmvb', '.vob', '.ogm', '.divx', '.xvid', '.f4v', '.f4p', '.f4a', '.f4b'
    ],
    # Audio
    'Audio': [
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.amr', 
        '.aiff', '.au', '.ra', '.ram', '.wv', '.ape', '.ac3', '.dts', '.mp2', 
        '.mpa', '.mka', '.spx', '.tta', '.tak', '.ofr', '.ofs', '.ofm', '.rka'
    ],
    # Fonts
    'Fonts': [
        '.ttf', '.otf', '.woff', '.woff2', '.eot', '.fon', '.fnt', '.pfb', '.pfa', 
        '.afm', '.pfm', '.ttc', '.dfont', '.bdf', '.pcf', '.snf', '.psf', '.psfu'
    ],
    # Database
    'Database': [
        '.db', '.sqlite', '.sqlite3', '.mdb', '.accdb', '.frm', '.myd', '.myi', 
        '.ibd', '.dbf', '.odb', '.ldb', '.sdf', '.db3', '.db2', '.dbs', 
        '.wdb', '.fdb', '.gdb', '.nsf', '.fp7', '.fp5', '.fp3', '.fmp12', '.fmp7'
    ]
}

def get_file_category(filename):
    """
    Determine file category based on extension with improved logic
    
    Args:
        filename (str): The filename to categorize
        
    Returns:
        str: The category name or 'Other' if not found
    """
    if not filename or not isinstance(filename, str):
        return 'Other'
    
    # Handle files without extensions
    if '.' not in filename:
        return 'Other'
    
    # Get the last extension (handle files like file.tar.gz)
    parts = filename.lower().split('.')
    if len(parts) < 2:
        return 'Other'
    
    # Check for multi-part extensions (like .tar.gz, .tar.bz2)
    ext = '.' + parts[-1]
    ext2 = '.' + '.'.join(parts[-2:]) if len(parts) >= 2 else ''
    
    # First check for multi-part extensions
    for category, extensions in FILE_CATEGORIES.items():
        if ext2 in extensions:
            return category
    
    # Then check for single extensions
    for category, extensions in FILE_CATEGORIES.items():
        if ext in extensions:
            return category
    
    # Try to get MIME type as fallback (if python-magic is available)
    try:
        import magic
        mime_type = magic.from_file(filename, mime=True)
        return _categorize_by_mime_type(mime_type)
    except (ImportError, Exception):
        # python-magic not available or file doesn't exist, continue with extension-based logic
        pass
    
    return 'Other'

def _categorize_by_mime_type(mime_type):
    """
    Categorize file by MIME type as fallback when extension-based categorization fails
    
    Args:
        mime_type (str): The MIME type of the file
        
    Returns:
        str: The category name or 'Other' if not found
    """
    if not mime_type:
        return 'Other'
    
    mime_type = mime_type.lower()
    
    # Document types
    if any(doc_type in mime_type for doc_type in [
        'application/pdf', 'application/msword', 'application/vnd.openxmlformats',
        'text/plain', 'text/rtf', 'application/rtf', 'text/csv', 'text/markdown',
        'application/vnd.ms-excel', 'application/vnd.ms-powerpoint',
        'application/vnd.oasis.opendocument', 'application/epub+zip'
    ]):
        return 'Documents'
    
    # Image types
    if mime_type.startswith('image/'):
        return 'Images'
    
    # Video types
    if mime_type.startswith('video/'):
        return 'Videos'
    
    # Audio types
    if mime_type.startswith('audio/'):
        return 'Audio'
    
    # Archive types
    if any(arch_type in mime_type for arch_type in [
        'application/zip', 'application/x-rar', 'application/x-7z',
        'application/gzip', 'application/x-bzip2', 'application/x-tar',
        'application/x-iso9660-image', 'application/vnd.ms-cab-compressed'
    ]):
        return 'Archives'
    
    # Code types
    if any(code_type in mime_type for code_type in [
        'text/html', 'text/css', 'text/javascript', 'application/javascript',
        'text/x-python', 'text/x-java', 'text/x-c', 'text/x-c++',
        'application/x-php', 'text/x-ruby', 'text/x-go', 'text/x-rust',
        'text/x-sql', 'application/json', 'application/xml', 'text/xml',
        'text/yaml', 'application/x-yaml'
    ]):
        return 'Code'
    
    # Executable types
    if any(exec_type in mime_type for exec_type in [
        'application/x-executable', 'application/x-msdownload',
        'application/x-msdos-program', 'application/x-sharedlib',
        'application/x-object', 'application/x-archive'
    ]):
        return 'Executables'
    
    # Font types
    if any(font_type in mime_type for font_type in [
        'font/', 'application/font-woff', 'application/font-woff2',
        'application/vnd.ms-fontobject'
    ]):
        return 'Fonts'
    
    # Database types
    if any(db_type in mime_type for db_type in [
        'application/x-sqlite3', 'application/vnd.ms-access',
        'application/x-dbase', 'application/x-msaccess'
    ]):
        return 'Database'
    
    return 'Other'

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
            # Determine file type from the file name or metadata
            file_name = meta["pdf_name"]
            file_type = meta.get("file_type", "unknown")
            if file_type == "unknown":
                file_type = get_file_category(file_name)
            
            # Determine if this is a filename match, content match, or image match
            chunk_type = meta.get("chunk_type", "content")
            if chunk_type == "filename":
                page_display = "Filename"
            elif chunk_type in ["image_description", "image_keyword", "image_filename", "image_content", "image_searchable", "image_possible"]:
                page_display = "Image Analysis"
            else:
                page_display = f"Page {meta['page_number']}"
            
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

# -------------------------
# Advanced Chat Functionality
# -------------------------

def parse_natural_language_command(user_input):
    """
    Parse natural language commands using LLM or fallback regex patterns
    Returns a structured command object
    """
    # Try LLM parsing first
    if OLLAMA_AVAILABLE:
        try:
            return parse_with_llm(user_input)
        except Exception as e:
            print(f"LLM parsing failed: {e}, falling back to regex")
    
    # Fallback to regex pattern matching
    return parse_with_regex(user_input)

def parse_with_llm(user_input):
    """
    Use Ollama LLM to parse natural language commands
    """
    prompt = f"""
    Parse this user command and return a JSON response with the following structure:
    {{
        "action": "search|select_folder|organize|rename",
        "folder": "folder_path_or_name",
        "keyword": "search_term",
        "file_type": "documents|images|videos|audio|all",
        "confidence": 0.0-1.0
    }}
    
    User command: "{user_input}"
    
    Examples:
    - "select Documents folder and find hello keyword" -> {{"action": "search", "folder": "Documents", "keyword": "hello", "file_type": "documents", "confidence": 0.9}}
    - "find images of landscapes" -> {{"action": "search", "folder": null, "keyword": "landscapes", "file_type": "images", "confidence": 0.8}}
    - "search for PDF files about finance" -> {{"action": "search", "folder": null, "keyword": "finance", "file_type": "documents", "confidence": 0.9}}
    
    Return only the JSON, no other text:
    """
    
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        
        # Extract JSON from response
        content = response['message']['content'].strip()
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in LLM response")
            
    except Exception as e:
        print(f"LLM parsing error: {e}")
        raise e

def parse_with_regex(user_input):
    """
    Fallback regex-based command parsing
    """
    user_input_lower = user_input.lower()
    
    # Initialize command structure
    command = {
        "action": "search",
        "folder": None,
        "keyword": None,
        "file_type": "all",
        "confidence": 0.7
    }
    
    # Extract folder names
    folder_patterns = [
        r'select\s+(\w+)\s+folder',
        r'in\s+(\w+)\s+folder',
        r'from\s+(\w+)\s+folder',
        r'(\w+)\s+folder'
    ]
    
    for pattern in folder_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            command["folder"] = match.group(1)
            break
    
    # Extract keywords
    keyword_patterns = [
        r'find\s+["\']?([^"\']+)["\']?\s+keyword',
        r'search\s+for\s+["\']?([^"\']+)["\']?',
        r'find\s+["\']?([^"\']+)["\']?',
        r'["\']([^"\']+)["\']'
    ]
    
    for pattern in keyword_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            command["keyword"] = match.group(1).strip()
            break
    
    # Extract file types
    if 'document' in user_input_lower or 'pdf' in user_input_lower or 'word' in user_input_lower:
        command["file_type"] = "documents"
    elif 'image' in user_input_lower or 'photo' in user_input_lower or 'picture' in user_input_lower:
        command["file_type"] = "images"
    elif 'video' in user_input_lower or 'movie' in user_input_lower:
        command["file_type"] = "videos"
    elif 'audio' in user_input_lower or 'music' in user_input_lower or 'sound' in user_input_lower:
        command["file_type"] = "audio"
    
    return command

def execute_command(command, current_folder):
    """
    Execute the parsed command and return results
    """
    try:
        # Handle folder selection
        if command.get("folder"):
            folder_path = resolve_folder_path(command["folder"], current_folder)
            if folder_path and os.path.exists(folder_path):
                global selected_folder
                selected_folder = folder_path
                print(f"Selected folder: {selected_folder}")
            else:
                return {"error": f"Folder '{command['folder']}' not found"}
        
        # Handle search action
        if command["action"] == "search" and command.get("keyword"):
            # Perform search in the selected folder
            results = search_in_documents(command["keyword"], selected_folder)
            
            # Filter by file type if specified
            if command["file_type"] != "all":
                results = filter_by_file_type(results, command["file_type"])
            
            return {
                "success": True,
                "results": results,
                "keyword": command["keyword"],
                "folder": selected_folder,
                "file_type": command["file_type"],
                "count": len(results)
            }
        
        return {"error": "Invalid command or missing parameters"}
        
    except Exception as e:
        return {"error": f"Command execution failed: {str(e)}"}

def resolve_folder_path(folder_name, current_folder):
    """
    Resolve folder name to full path
    """
    # Common folder mappings
    folder_mappings = {
        "documents": os.path.join(os.path.expanduser("~"), "Documents"),
        "downloads": os.path.join(os.path.expanduser("~"), "Downloads"),
        "desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
        "pictures": os.path.join(os.path.expanduser("~"), "Pictures"),
        "videos": os.path.join(os.path.expanduser("~"), "Videos"),
        "music": os.path.join(os.path.expanduser("~"), "Music"),
    }
    
    # Check if it's a mapped folder
    if folder_name.lower() in folder_mappings:
        return folder_mappings[folder_name.lower()]
    
    # Check if it's a relative path from current folder
    relative_path = os.path.join(current_folder, folder_name)
    if os.path.exists(relative_path):
        return relative_path
    
    # Check if it's an absolute path
    if os.path.exists(folder_name):
        return folder_name
    
    return None

def filter_by_file_type(results, file_type):
    """
    Filter search results by file type
    """
    if file_type == "all":
        return results
    
    filtered_results = []
    for result in results:
        result_file_type = result.get("file_type", "").lower()
        if file_type.lower() in result_file_type or result_file_type in file_type.lower():
            filtered_results.append(result)
    
    return filtered_results

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
                                        ft.Icons.TITLE if result.get('chunk_type') == 'filename' 
                                        else ft.Icons.IMAGE if result.get('file_type') == 'image'
                                        else ft.Icons.DESCRIPTION, 
                                        color="orange" if result.get('chunk_type') == 'filename' 
                                        else "purple" if result.get('file_type') == 'image'
                                        else "blue"
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
        margin=ft.margin.only(left=0, right=0, top=-100),
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
        label="Search All Documents & Images",
        hint_text="Search across PDFs, Word docs, Excel, PowerPoint, text files, and images...",
        expand=True,
        prefix_icon=ft.Icons.SEARCH,
        on_submit=lambda e: perform_search(e, search_field, results_container)
    )
    
    search_button = ft.ElevatedButton(
        "Search All Files",
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
    # Default source to current selected_folder or user Downloads
    try:
        default_downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    except Exception:
        default_downloads = ""
    if selected_folder and os.path.isdir(selected_folder):
        source_folder = selected_folder
    elif default_downloads and os.path.isdir(default_downloads):
        source_folder = default_downloads
    else:
        source_folder = ""
    destination_folder = ""
    
    # -------------------------
    # Genre classification wrapper (Qwen) used only for document routing
    # -------------------------
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
    
    class _GenreLLM:
        """Lazy-loaded classifier that returns one label from GENRE_LABELS."""
        
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
            except Exception as e:
                try:
                    print(f"[GenreLLM][error] Load failed: {e}", flush=True)
                    traceback.print_exc()
                except Exception:
                    pass
                self._load_failed = True
        
        def classify(self, context: str) -> str:
            self._ensure_loaded()
            if self._load_failed or self._tokenizer is None or self._model is None:
                return self._fallback(context)
            genres_str = ", ".join(GENRE_LABELS)
            system_msg = (
                "You are a strict document classifier. Choose EXACTLY ONE genre label from the provided list. "
                "Respond with only the label text."
            )
            user_msg = (
                f"Genres: {genres_str}\n\n"
                f"Document excerpt (short):\n{(context or '').strip()[:1600]}\n\n"
                "Return exactly one label from the list above."
            )
            messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
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
                if candidate in GENRE_LABELS:
                    return candidate
                for g in GENRE_LABELS:
                    if candidate.lower() == g.lower():
                        return g
                return "Other"
            except Exception as e:
                try:
                    print(f"[GenreLLM][error] Classification failed: {e}", flush=True)
                    traceback.print_exc()
                except Exception:
                    pass
                self._load_failed = True
                return self._fallback(context)
        
        def _fallback(self, context: str) -> str:
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
    
    genre_llm = _GenreLLM()
    
    def _extract_document_context(file_path: str, file_name: str, max_chars: int = 4000) -> str:
        try:
            ext = os.path.splitext(file_name)[1].lower()
            file_type = SUPPORTED_EXTENSIONS.get(ext)
            file_info = {"path": file_path, "name": file_name, "extension": ext, "type": file_type}
            chunks = extract_text_from_file(file_info)
            content_texts: List[str] = []
            for ch in chunks:
                if ch.get("chunk_type") == "content" and ch.get("text"):
                    content_texts.append(ch["text"].strip())
            if not content_texts:
                for ch in chunks:
                    if ch.get("chunk_type") != "filename" and ch.get("text"):
                        content_texts.append(ch["text"].strip())
            combined = "\n\n".join(content_texts)
            return combined[:max_chars] if combined else os.path.splitext(file_name)[0]
        except Exception:
            return os.path.splitext(file_name)[0]
    
    def get_genre_destination(file_path: str, file_name: str, dest_root: str) -> (str, str):
        """Wrapper: classify and return (category_dir, display_category)."""
        ctx = _extract_document_context(file_path, file_name)
        pred = genre_llm.classify(ctx) or "Other"
        if pred not in GENRE_LABELS:
            pred = "Other"
        category_dir = os.path.join(dest_root, "Documents", pred)
        display_category = f"Documents/{pred}"
        return category_dir, display_category
    
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
    
    # Tree view for showing file moves (force scrollbar visible for long lists)
    tree_view = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        spacing=2,
        height=420,
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
    
    # Use the shared file categorization function
    
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
    
    # Dictionary to store files by category for proper tree display
    files_by_category = {}
    
    def update_tree_view_with_move(source_path, dest_path, category):
        """Add a file move to the tree view (collects files for later display)"""
        filename = os.path.basename(source_path)
        
        # Debug logging
        print(f"DEBUG: update_tree_view_with_move called with:")
        print(f"  source_path: {source_path}")
        print(f"  dest_path: {dest_path}")
        print(f"  category: {category}")
        print(f"  filename: {filename}")
        
        # Add file to the category dictionary
        if category not in files_by_category:
            files_by_category[category] = []
        files_by_category[category].append(filename)
    
    def display_tree_view():
        """Display the tree view with files properly grouped by category"""
        # Clear the tree view first
        clear_tree_view()
        
        # Define category display order (logical order, not alphabetical)
        category_order = [
            "Documents", "Images", "Videos", "Audio", "Archives", 
            "Code", "Executables", "Fonts", "3D_Models", "Database", 
            "Directories", "Other"
        ]
        
        # Add categories and their files in the defined order
        for category in category_order:
            if category in files_by_category and files_by_category[category]:
                # Get category display name and icon
                category_name = category if category != "Directories" else "📁 Directories"
                
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
                elif category == "Fonts":
                    category_icon = ft.Icons.FONT_DOWNLOAD
                elif category == "3D_Models":
                    category_icon = ft.Icons.THREE_D_ROTATION
                elif category == "Database":
                    category_icon = ft.Icons.STORAGE
                elif category == "Directories":
                    category_icon = ft.Icons.FOLDER_OPEN
                else:
                    category_icon = ft.Icons.INSERT_DRIVE_FILE
                
                # Add category header
                category_node = create_tree_node(
                    f"📁 {category_name} ({len(files_by_category[category])} files)",
                    icon=category_icon,
                    color=ft.Colors.BLUE_600,
                    indent=0
                )
                add_to_tree_view(category_node)
                
                # Add files under this category (sorted alphabetically within category)
                sorted_files = sorted(files_by_category[category])
                for filename in sorted_files:
                    # Get appropriate file icon
                    file_icon = ft.Icons.INSERT_DRIVE_FILE
                    if category == "Images":
                        file_icon = ft.Icons.IMAGE
                    elif category == "Videos":
                        file_icon = ft.Icons.VIDEO_FILE
                    elif category == "Audio":
                        file_icon = ft.Icons.AUDIO_FILE
                    elif category == "Documents":
                        file_icon = ft.Icons.DESCRIPTION
                    elif category == "Code":
                        file_icon = ft.Icons.CODE
                    elif category == "Archives":
                        file_icon = ft.Icons.ARCHIVE
                    elif category == "Executables":
                        file_icon = ft.Icons.APPS
                    elif category == "Fonts":
                        file_icon = ft.Icons.FONT_DOWNLOAD
                    elif category == "3D_Models":
                        file_icon = ft.Icons.THREE_D_ROTATION
                    elif category == "Database":
                        file_icon = ft.Icons.STORAGE
                    elif category == "Directories":
                        file_icon = ft.Icons.FOLDER_OPEN
                    
                    file_node = create_tree_node(
                        f"📄 {filename}",
                        icon=file_icon,
                        color=ft.Colors.GREY_700,
                        indent=1
                    )
                    add_to_tree_view(file_node)

        # Render any additional dynamic categories (e.g., Documents/History)
        for category in sorted(files_by_category.keys()):
            if category in category_order:
                continue
            if not files_by_category.get(category):
                continue
            category_icon = ft.Icons.INSERT_DRIVE_FILE
            if category.startswith("Documents/"):
                category_icon = ft.Icons.DESCRIPTION
            elif category.startswith("Images/"):
                category_icon = ft.Icons.IMAGE
            elif category.startswith("Videos/"):
                category_icon = ft.Icons.VIDEO_FILE
            elif category.startswith("Audio/"):
                category_icon = ft.Icons.AUDIO_FILE
            category_node = create_tree_node(
                f"📁 {category} ({len(files_by_category[category])} files)",
                icon=category_icon,
                color=ft.Colors.BLUE_600,
                indent=0
            )
            add_to_tree_view(category_node)
            for filename in sorted(files_by_category[category]):
                file_node = create_tree_node(
                    f"📄 {filename}",
                    icon=ft.Icons.INSERT_DRIVE_FILE,
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
            
            # Clear files dictionary and tree view
            files_by_category.clear()
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
                        print(f"DEBUG: File {item_name} categorized as: {category}")
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
            
            # Display the organized tree view with files grouped by category
            display_tree_view()
            
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
    source_display = ft.Text(
        f"Source: {source_folder}" if source_folder else "No source folder selected",
        size=12,
        color=ft.Colors.GREY_600
    )
    
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
    
    # Organize buttons
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
    
    def categorize_further(e):
        # Re-run organization but only for Documents: move into Documents/<Genre>
        if not source_folder or not destination_folder:
            # Provide immediate user feedback in the UI
            progress_container.visible = True
            progress_text.visible = True
            progress_bar.visible = False
            progress_percentage.visible = False
            progress_text.value = "Please select both Source and Destination folders to organize."
            # Also show a message in the tree view area
            clear_tree_view()
            add_to_tree_view(create_tree_node("⚠️ Please choose both Source and Destination folders.", color=ft.Colors.ORANGE_600))
            tree_container.visible = True
            try:
                tree_container.update()
            except Exception:
                pass
            return
        try:
            # Collect all files currently in destination Documents (flat or nested)
            docs_root = os.path.join(destination_folder, "Documents")
            if not os.path.exists(docs_root):
                return
            moved = 0
            for root, _, files in os.walk(docs_root):
                for fname in files:
                    src_path = os.path.join(root, fname)
                    # Skip if already in a genre subfolder (root != docs_root)
                    # We still allow reclassification by always classifying and moving to predicted folder
                    try:
                        genre_dir, display_category = get_genre_destination(src_path, fname, destination_folder)
                        os.makedirs(genre_dir, exist_ok=True)
                        dest_path = os.path.join(genre_dir, fname)
                        if os.path.abspath(src_path) != os.path.abspath(dest_path):
                            # Handle duplicates
                            base, ext = os.path.splitext(fname)
                            counter = 1
                            final_dest = dest_path
                            while os.path.exists(final_dest):
                                alt_name = f"{base}_{counter}{ext}"
                                final_dest = os.path.join(genre_dir, alt_name)
                                counter += 1
                            shutil.move(src_path, final_dest)
                            update_tree_view_with_move(src_path, final_dest, display_category)
                            moved += 1
                    except Exception as _ge:
                        try:
                            print(f"[Genre] Categorize further failed for {src_path}: {_ge}", flush=True)
                        except Exception:
                            pass
            if moved > 0:
                display_tree_view()
        except Exception as ex:
            print(f"Categorize further error: {ex}")
    
    categorize_button = ft.ElevatedButton(
        "Categorize Further (LLM)",
        icon=ft.Icons.AUTO_AWESOME,
        on_click=categorize_further,
        bgcolor=ft.Colors.ORANGE,
        color=ft.Colors.WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=ft.padding.symmetric(horizontal=20, vertical=15)
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
            ft.Text("• Documents: PDF, Word, Excel, PowerPoint, Text, eBooks, Config files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Images: JPG, PNG, GIF, SVG, RAW, PSD, and other image formats", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Videos: MP4, AVI, MOV, MKV, and other video formats", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Audio: MP3, WAV, FLAC, AAC, and other audio formats", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Archives: ZIP, RAR, 7Z, TAR, and other compressed files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Code: Python, JavaScript, HTML, CSS, and other programming files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Executables: EXE, MSI, APP, and other executable files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Fonts: TTF, OTF, WOFF, and other font files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• 3D Models: OBJ, FBX, STL, and other 3D model files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Database: SQLite, Access, and other database files", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Directories: All subdirectories will be moved to 'Directories' folder", size=12, color=ft.Colors.GREY_600),
            ft.Text("• Other: Files with unrecognized extensions", size=12, color=ft.Colors.GREY_600),
        ], spacing=2),
        padding=ft.padding.all(15),
        bgcolor=ft.Colors.GREY_50,
        border_radius=8,
        border=ft.border.all(1, ft.Colors.GREY_300),
        height=280
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
        # First row: source and destination selectors
        ft.Row([
            source_section,
            destination_section,
        ], alignment=ft.MainAxisAlignment.SPACE_EVENLY, vertical_alignment=ft.CrossAxisAlignment.START, spacing=15),
        # Second row: action buttons
        ft.Row([
            ft.Container(
                content=ft.Row([
                    organize_button,
                    categorize_button,
                    rollback_button
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                padding=ft.padding.only(bottom=20)
            )
        ], alignment=ft.MainAxisAlignment.CENTER),
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

def create_rename_content(file_picker, page_update_callback=None):
    """Create rename tab content"""
    entries: List[FileEntry] = []
    rename_selected_folder: Optional[str] = None
    llm = LLMHelper()
    
    # Default page update function if none provided
    def default_page_update():
        pass
    
    if page_update_callback is None:
        page_update_callback = default_page_update

    # Header
    header = ft.Container(
        content=ft.Image(src="rename.png"),
        border_radius=ft.border_radius.all(20),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        margin=ft.margin.only(left=0, right=0, top=-100),
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
    
    def on_folder_picked(e: ft.FilePickerResultEvent) -> None:
        nonlocal rename_selected_folder
        if not e.path:
            return
        rename_selected_folder = e.path
        folder_display.value = f"Selected: {rename_selected_folder}"
        folder_display.update()
        load_files()

    file_picker.on_result = on_folder_picked

    choose_btn = ft.ElevatedButton(
        "Choose Folder",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=lambda _: file_picker.get_directory_path(),
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
        page_update_callback()

    def load_files() -> None:
        left_list.controls.clear()
        right_list.controls.clear()
        entries.clear()
        if not rename_selected_folder:
            page_update_callback()
            return
        docs = find_documents(rename_selected_folder)
        for doc in docs:
            file_path = doc["path"]
            file_name = doc["name"]
            base, ext = os.path.splitext(file_name)
            checkbox = ft.Checkbox(label="", value=False)
            suggest_field = ft.TextField(value="", hint_text=f"Suggested for {base}", expand=True)
            left_list.controls.append(ft.Row([checkbox, ft.Text(file_name)], spacing=10))
            right_list.controls.append(suggest_field)
            entries.append(FileEntry(path=file_path, name=file_name, ext=ext, checkbox=checkbox, suggestion_field=suggest_field))
        page_update_callback()
        # Auto-generate suggestions after loading
        generate_suggestions_async()

    def generate_suggestions_async() -> None:
        if not entries:
            return
        # Hide lists while generating
        lists_container.visible = False
        page_update_callback()

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
                    page_update_callback()
            set_progress(False)
            # Reveal lists after generation completes
            lists_container.visible = True
            page_update_callback()

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
            return
        # Hide lists while renaming
        lists_container.visible = False
        set_progress(True, 0.0, "Renaming files...")
        page_update_callback()

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
                    page_update_callback()
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
                page_update_callback()

        set_progress(False)
        try:
            print(f"[Rename] Completed. Renamed {renamed} of {total_to_process} files.", flush=True)
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

    return root

def create_chat_content(file_picker):
    """Create chat tab content with UI similar to the image"""
    
    # Chat messages container
    chat_messages = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        spacing=10,
        expand=True,
        controls=[]
    )
    
    # Chat input field
    chat_input = ft.TextField(
        hint_text="Ask me anything",
        expand=True,
        border_radius=25,
        content_padding=ft.padding.symmetric(horizontal=20, vertical=15),
        filled=True,
        bgcolor=ft.Colors.WHITE,
        border=ft.border.all(1, ft.Colors.GREY_300),
        multiline=False,
        max_lines=1
    )
    
    
    def handle_chat_submit(e):
        """Handle chat message submission"""
        message = chat_input.value.strip()
        if not message:
            return
        
        # Add user message
        add_chat_message(message, is_user=True)
        
        # Clear input
        chat_input.value = ""
        chat_input.update()
        
        # Use LangChain agent if available, otherwise fallback to old system
        if agent_initialized:
            try:
                # Get response from LangChain agent
                response = chat_with_agent(message)
                
                # Check if the response contains file search results
                if any(keyword in message.lower() for keyword in ['find', 'search', 'show', 'get', 'select', 'folder', 'keyword']):
                    # Try to extract search results for file display
                    try:
                        # Parse the message to extract search parameters
                        command = parse_natural_language_command(message)
                        if command.get("keyword"):
                            # Get search results for file display
                            search_results = get_search_results_for_ui(
                                command["keyword"], 
                                command.get("folder") and resolve_folder_path(command["folder"], selected_folder),
                                command.get("file_type", "all")
                            )
                            if search_results:
                                add_chat_message(response, is_user=False, files_found=search_results)
                            else:
                                add_chat_message(response, is_user=False)
                        else:
                            add_chat_message(response, is_user=False)
                    except:
                        add_chat_message(response, is_user=False)
                else:
                    add_chat_message(response, is_user=False)
                    
            except Exception as ex:
                print(f"LangChain agent error: {ex}")
                # Fallback to old system
                add_chat_message(f"Sorry, I encountered an error: {str(ex)}. Let me try a different approach.", is_user=False)
        else:
            # Fallback to old command parsing system
            if any(keyword in message.lower() for keyword in ['find', 'search', 'show', 'get', 'select', 'folder', 'keyword']):
                # Try to parse as natural language command
                try:
                    command = parse_natural_language_command(message)
                    print(f"Parsed command: {command}")
                    
                    # Execute the command
                    result = execute_command(command, selected_folder)
                    
                    if result.get("success"):
                        # Command executed successfully
                        if result.get("results"):
                            response = f"Found {result['count']} {result['file_type']} files containing '{result['keyword']}' in {os.path.basename(result['folder'])}:"
                            add_chat_message(response, is_user=False, files_found=result["results"])
                        else:
                            response = f"No {result['file_type']} files found containing '{result['keyword']}' in {os.path.basename(result['folder'])}."
                            add_chat_message(response, is_user=False)
                    else:
                        # Command failed
                        error_msg = result.get("error", "Unknown error occurred")
                        response = f"Sorry, I couldn't process that command: {error_msg}"
                        add_chat_message(response, is_user=False)
                        
                except Exception as ex:
                    print(f"Command processing error: {ex}")
                    # Fallback to simple search
                    try:
                        results = search_in_documents(message, selected_folder)
                        if results:
                            response = f"I found {len(results)} files matching your query. Here are the results:"
                            add_chat_message(response, is_user=False, files_found=results)
                        else:
                            response = "I couldn't find any files matching your query. Try a different search term."
                            add_chat_message(response, is_user=False)
                    except Exception as ex2:
                        response = f"Sorry, I encountered an error while searching: {str(ex2)}"
                        add_chat_message(response, is_user=False)
            else:
                # General chat response
                response = f"I understand you're asking about: '{message}'. I can help you find files, organize documents, or answer questions about your files. Try commands like 'select Documents folder and find hello keyword' or 'find images of landscapes'. What would you like to do?"
                add_chat_message(response, is_user=False)
    
    # Send button
    send_button = ft.IconButton(
        icon=ft.Icons.SEND,
        icon_color=ft.Colors.BLUE_600,
        tooltip="Send message",
        on_click=handle_chat_submit
    )
    
    voice_button = ft.IconButton(
        icon=ft.Icons.MIC,
        icon_color=ft.Colors.GREY_600,
        tooltip="Voice input"
    )
    
    def add_chat_message(message, is_user=True, files_found=None):
        """Add a message to the chat"""
        if is_user:
            # User message
            user_message = ft.Container(
                content=ft.Row([
                    ft.Text(message, color=ft.Colors.BLACK, size=14)
                ], alignment=ft.MainAxisAlignment.END),
                bgcolor=ft.Colors.BLUE_50,
                padding=ft.padding.all(12),
                border_radius=ft.border_radius.only(
                    top_left=20, top_right=20, bottom_left=20, bottom_right=5
                ),
                margin=ft.margin.only(left=50, right=10, top=5, bottom=5)
            )
            chat_messages.controls.append(user_message)
        else:
            # Assistant message
            assistant_content = [ft.Text(message, color=ft.Colors.BLACK, size=14)]
            
            if files_found:
                # Add file results similar to the image
                file_card = ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text(f"{len(files_found)} files", 
                                   weight=ft.FontWeight.BOLD, size=16),
                            ft.ElevatedButton(
                                "View all files",
                                icon=ft.Icons.OPEN_IN_NEW,
                                bgcolor=ft.Colors.WHITE,
                                color=ft.Colors.BLUE_600,
                                style=ft.ButtonStyle(
                                    shape=ft.RoundedRectangleBorder(radius=8),
                                    padding=ft.padding.symmetric(horizontal=12, vertical=6)
                                )
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Divider(),
                        *[create_file_item(file) for file in files_found[:2]]  # Show first 2 files
                    ], spacing=8),
                    bgcolor=ft.Colors.WHITE,
                    padding=ft.padding.all(15),
                    border_radius=12,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    shadow=ft.BoxShadow(
                        spread_radius=1,
                        blur_radius=4,
                        color=ft.Colors.GREY_400,
                        offset=ft.Offset(0, 2)
                    )
                )
                assistant_content.append(file_card)
            
            assistant_message = ft.Container(
                content=ft.Column(assistant_content, spacing=10),
                bgcolor=ft.Colors.WHITE,
                padding=ft.padding.all(12),
                border_radius=ft.border_radius.only(
                    top_left=20, top_right=20, bottom_left=5, bottom_right=20
                ),
                margin=ft.margin.only(left=10, right=50, top=5, bottom=5),
                border=ft.border.all(1, ft.Colors.GREY_200)
            )
            chat_messages.controls.append(assistant_message)
        
        chat_messages.update()
    
    def create_file_item(file_info):
        """Create a clickable file item similar to the image"""
        
        def on_file_click(e):
            """Handle file click to open the file"""
            try:
                # Construct the full file path
                file_name = file_info['file']
                folder_path = file_info['full_path']
                full_file_path = os.path.join(folder_path, file_name)
                
                # Open the file using the system default program
                open_file_location(full_file_path)
                
                # Add a message to chat showing the file was opened
                add_chat_message(f"Opened file: {file_name}", is_user=False)
                
            except Exception as ex:
                error_msg = f"Could not open file: {str(ex)}"
                add_chat_message(error_msg, is_user=False)
        
        def on_add_to_conversation(e):
            """Handle 'Add to conversation' button click"""
            file_name = file_info['file']
            add_chat_message(f"Added {file_name} to conversation", is_user=False)
        
        # Get appropriate icon based on file type
        file_type = file_info.get('file_type', '').lower()
        if 'image' in file_type:
            file_icon = ft.Icons.IMAGE
        elif 'video' in file_type:
            file_icon = ft.Icons.VIDEO_FILE
        elif 'audio' in file_type:
            file_icon = ft.Icons.AUDIO_FILE
        elif 'document' in file_type or 'pdf' in file_info.get('file', '').lower():
            file_icon = ft.Icons.DESCRIPTION
        elif 'code' in file_type or any(ext in file_info.get('file', '').lower() for ext in ['.py', '.js', '.html', '.css']):
            file_icon = ft.Icons.CODE
        else:
            file_icon = ft.Icons.INSERT_DRIVE_FILE
        
        # Get file modification time for display
        try:
            file_path = os.path.join(file_info['full_path'], file_info['file'])
            if os.path.exists(file_path):
                mod_time = os.path.getmtime(file_path)
                from datetime import datetime
                mod_date = datetime.fromtimestamp(mod_time)
                time_ago = f"{mod_date.strftime('%d/%m/%Y')}"
            else:
                time_ago = "Unknown"
        except:
            time_ago = "Unknown"
        
        return ft.GestureDetector(
            content=ft.Container(
                content=ft.Row([
                    ft.Icon(
                        file_icon,
                        color=ft.Colors.BLUE_600,
                        size=20
                    ),
                    ft.Column([
                        ft.Text(file_info['file'], weight=ft.FontWeight.BOLD, size=12),
                        ft.Text(time_ago, color=ft.Colors.GREY_600, size=10)
                    ], spacing=2, expand=True),
                    ft.ElevatedButton(
                        "Add to conversation",
                        bgcolor=ft.Colors.WHITE,
                        color=ft.Colors.BLUE_600,
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=6),
                            padding=ft.padding.symmetric(horizontal=8, vertical=4)
                        ),
                        on_click=on_add_to_conversation
                    ),
                    ft.IconButton(
                        icon=ft.Icons.MORE_VERT,
                        icon_color=ft.Colors.GREY_600,
                        icon_size=16
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=ft.padding.symmetric(vertical=8, horizontal=12),
                bgcolor=ft.Colors.GREY_50,
                border_radius=8,
                margin=ft.margin.only(bottom=5)
            ),
            on_tap=on_file_click
        )
    
    
    # Chat interface
    chat_interface = ft.Container(
        content=ft.Column([
            # Chat messages area
            ft.Container(
                content=chat_messages,
                expand=True,
                padding=ft.padding.all(20),
                bgcolor=ft.Colors.GREY_50
            ),
            
            # Chat input area
            ft.Container(
                content=ft.Row([
                    chat_input,
                    voice_button,
                    send_button
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, spacing=10),
                padding=ft.padding.all(20),
                bgcolor=ft.Colors.WHITE,
                border=ft.border.only(top=ft.BorderSide(1, ft.Colors.GREY_300))
            )
        ], expand=True),
        expand=True
    )
    
    # Set up chat input submission
    chat_input.on_submit = handle_chat_submit
    
    chat_content = ft.Column([
        chat_interface
    ], expand=True)
    
    # Add initial welcome message after content is created
    def add_welcome_message():
        if agent_initialized:
            welcome_msg = """🚀 Hello! I'm SnapIndex, your enthusiastic file management assistant! 

I can help you with:
🔍 **Smart File Search** - Find files using natural language
📁 **Folder Management** - Navigate and organize your folders  
📄 **File Operations** - Open, organize, and manage your files
📊 **File Analysis** - Get detailed information about your files

Try asking me things like:
• "Find all PDF files about finance in my Documents folder"
• "Show me images from last month"
• "Organize my Downloads folder"
• "What's in my Pictures folder?"

I'm here to make file management fun and easy! What would you like to do? 😊"""
        else:
            welcome_msg = """Hello! I can help you find and organize files on your PC using natural language commands. Try commands like:
• 'select Documents folder and find hello keyword'
• 'find images of landscapes'  
• 'search for PDF files about finance'
I'll automatically parse your commands and search the appropriate folders!"""
        
        # Add the welcome message to the chat messages list without calling update
        assistant_message = ft.Container(
            content=ft.Column([ft.Text(welcome_msg, color=ft.Colors.BLACK, size=14)], spacing=10),
            bgcolor=ft.Colors.WHITE,
            padding=ft.padding.all(12),
            border_radius=ft.border_radius.only(
                top_left=20, top_right=20, bottom_left=5, bottom_right=20
            ),
            margin=ft.margin.only(left=10, right=50, top=5, bottom=5),
            border=ft.border.all(1, ft.Colors.GREY_200)
        )
        chat_messages.controls.append(assistant_message)
    
    # Store the function to call later
    chat_content.add_welcome_message = add_welcome_message
    
    return chat_content

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
            
            # Add welcome message to chat if switching to chat tab
            if tab_name == "chat" and hasattr(content_container.content, 'add_welcome_message'):
                content_container.content.add_welcome_message()
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
        
        # Chat button
        chat_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.CHAT, color="black"),
                ft.Text("Chat", weight=ft.FontWeight.BOLD if current_tab == "chat" else ft.FontWeight.NORMAL)
            ]),
            bgcolor="#f5f7fa" if current_tab == "chat" else "white",
            padding=8,
            on_click=lambda e: on_tab_change("chat"),
            data="chat"
        )
        sidebar_buttons.append(chat_btn)
        
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
            return create_rename_content(file_picker, lambda: page.update())
        elif tab_name == "chat":
            content = create_chat_content(file_picker)
            return content
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
