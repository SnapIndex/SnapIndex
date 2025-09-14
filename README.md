# SnapIndex

**The Future of File Exploration with Scalable Semantic Search across next-gen snapdragon 
devices.**

SnapIndex is a powerful desktop application that revolutionizes how you search, organize, and manage your files using advanced AI and semantic search technology. Built with Python and featuring a modern GUI, it provides intelligent file management capabilities that go far beyond traditional file systems.

## 👨‍💻 Developers

- **Swastik Mishra** - sgm9198@nyu.edu
- **Akarsh Malik** -  am15764@nyu.edu
- **Shreyansh Saurabh** - ss21034@nyu.edu

## 🚀 Key Features

### 🔍 **Semantic Search Engine**
- **Harness NPU Accelaration** : Delivers lightning-fast inference by pushing Snapdragon to its limits, delivering millisecond-level inference 
- **Multi-format Support**: Search across PDFs, Word documents, Excel files, PowerPoint presentations, text files, and images
- **AI-Powered**: Uses state-of-the-art BGE embedding models for semantic understanding
- **Vector Database**: Built on FAISS for lightning-fast similarity search
- **Smart Results**: Find content by meaning, not just exact text matches
- **Incremental Updates**: Only processes new files, keeping your search index up-to-date efficiently

### 📁 **Intelligent File Organization**
- **Automatic Categorization**: Organize files by type (Documents, Images, Videos, Audio, Archives, Code, Executables)
- **Smart Folder Structure**: Creates organized directory trees automatically
- **Rollback Support**: Undo organization changes with complete path restoration
- **Progress Tracking**: Real-time progress bars and detailed operation logs

### 🏷️ **AI-Powered File Renaming**
- **Content-Aware**: Uses Qwen2.5 language model to suggest meaningful filenames
- **Context Understanding**: Analyzes file content to generate descriptive names
- **Batch Processing**: Rename multiple files simultaneously
- **Customizable**: Edit suggestions before applying changes
- **Safe Operations**: Ensures unique filenames and handles conflicts gracefully

### 🖼️ **Advanced Image Processing**
- **AI Image Classification**: Automatically identifies and tags image content
- **Searchable Descriptions**: Makes images discoverable through natural language queries
- **Multiple Formats**: Supports JPG, PNG, GIF, SVG, WebP, and many more
- **Intelligent Keywords**: Generates comprehensive searchable metadata

## 🛠️ **Technical Architecture**

### **Core Technologies**
- **Frontend**: Flet (Python-based GUI framework)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: BAAI/bge-small-en-v1.5 model via FastEmbed
- **AI Models**: Qwen2.5-0.5B-Instruct for file naming
- **Image AI**: Custom image classification pipeline
- **File Processing**: PyPDF, python-docx, openpyxl, python-pptx

### **Supported File Types**
- **Documents**: PDF, DOC, DOCX, TXT, RTF, MD, HTML, XML, JSON
- **Spreadsheets**: XLS, XLSX, CSV
- **Presentations**: PPT, PPTX
- **Images**: JPG, PNG, GIF, BMP, TIFF, SVG, WebP, ICO, PSD, RAW formats
- **Archives**: ZIP, RAR, 7Z, TAR, GZ, BZ2
- **Code**: PY, JS, HTML, CSS, Java, C++, PHP, Ruby, Go, Rust, SQL
- **Executables**: EXE, MSI, DEB, RPM, DMG, APP

## 📦 **Installation**

### **Prerequisites**
- Python 3.8 or higher
- Windows 10/11 (primary support)
- 4GB+ RAM recommended
- 2GB+ free disk space

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/SnapIndex/SnapIndex.git
cd SnapIndex

# Install dependencies
pip install -r requirements.txt

# Run the main application
python app.py

# Or run specific tools
python organize.py [folder_path]    # File organization
python rename.py [folder_path]      # AI file renaming
```

### **Building Executables**
```bash
# Build all applications
.\build.ps1

# Individual builds
pyinstaller --noconsole --onefile app.py
pyinstaller --noconsole --onefile organize.py
pyinstaller --noconsole --onefile rename.py
pyinstaller --noconsole --onefile standalone_app.py
```

## 🎯 **Usage Guide**

### **1. Semantic Search (app.py)**
```bash
# Search in default Downloads folder
python app.py

# Search in specific folder
python app.py "C:\MyDocuments"

# Features:
# - Select any folder to index
# - Natural language queries
# - Click results to open files
# - Real-time progress tracking
```

### **2. File Organization (organize.py)**
```bash
# Organize Downloads folder
python organize.py

# Organize specific folder
python organize.py "C:\Downloads\MessyFolder"

# Features:
# - Automatic file categorization
# - Visual progress tracking
# - Rollback capability
# - Detailed operation logs
```

### **3. AI File Renaming (rename.py)**
```bash
# Rename files in Downloads
python rename.py

# Rename files in specific folder
python rename.py "C:\Photos\Vacation2024"

# Features:
# - AI-generated suggestions
# - Batch processing
# - Customizable names
# - Safe conflict resolution
```

## 🔧 **Configuration**

### **Default Settings**
- **Default Folder**: `C:\Users\[username]\Downloads`
- **Embedding Model**: BAAI/bge-small-en-v1.5
- **Language Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Search Results**: Top 10 matches
- **Similarity Threshold**: 0.6

### **Customization**
Edit `config.yaml` to modify:
- Default search folders
- AI model preferences
- Search parameters
- UI themes

## 📊 **Performance Features**

### **Optimization**
- **Parallel Processing**: Multi-threaded file processing and embedding generation
- **Incremental Updates**: Only processes new or modified files
- **Memory Efficient**: Streaming processing for large file collections
- **Caching**: Intelligent caching of embeddings and metadata

### **Scalability**
- **Large Collections**: Tested with 10,000+ files
- **Multiple Folders**: Support for multiple indexed directories
- **Database Management**: Automatic cleanup and optimization

## 🛡️ **Safety & Reliability**

### **Data Protection**
- **Non-destructive**: Search operations never modify original files
- **Rollback Support**: Complete undo capability for organization
- **Backup Tracking**: Detailed logs of all file operations
- **Conflict Resolution**: Smart handling of filename conflicts

### **Error Handling**
- **Graceful Degradation**: Continues operation even if some files fail
- **Comprehensive Logging**: Detailed error reports and operation logs
- **Recovery Options**: Multiple fallback strategies for failed operations

## 🔮 **Advanced Features**

### **AI Integration**
- **Semantic Understanding**: Goes beyond keyword matching
- **Context Awareness**: Understands document relationships
- **Natural Language**: Search using everyday language
- **Learning**: Improves suggestions based on usage patterns

### **File Intelligence**
- **Content Analysis**: Deep understanding of file contents
- **Metadata Extraction**: Rich metadata from all supported formats
- **Relationship Mapping**: Discovers connections between files
- **Trend Analysis**: Identifies patterns in your file collections

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage
pytest --cov=.
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 **Support**

- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join our discussions for tips and tricks

---

**SnapIndex** - Transform your file management experience with the power of AI and semantic search. 🚀