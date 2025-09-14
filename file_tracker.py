import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Set

class FileTracker:
    """
    Tracks file changes in directories to enable incremental processing
    """
    
    def __init__(self, tracking_file: str = "file_tracking.json"):
        self.tracking_file = tracking_file
        self.tracked_files = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load existing tracking data from file"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading tracking data: {e}")
                return {}
        return {}
    
    def _save_tracking_data(self):
        """Save tracking data to file"""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracked_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving tracking data: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for change detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _get_file_info(self, file_path: str) -> Dict:
        """Get file information including size and modification time"""
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "hash": self._get_file_hash(file_path)
            }
        except Exception as e:
            print(f"Error getting file info for {file_path}: {e}")
            return {"size": 0, "mtime": 0, "hash": ""}
    
    def get_new_or_changed_files(self, folder_path: str, supported_extensions: Set[str]) -> List[str]:
        """
        Get list of new or changed files in the folder
        
        Args:
            folder_path: Path to the folder to scan
            supported_extensions: Set of supported file extensions
            
        Returns:
            List of file paths that are new or changed
        """
        new_or_changed = []
        folder_key = os.path.abspath(folder_path)
        
        # Initialize folder tracking if not exists
        if folder_key not in self.tracked_files:
            self.tracked_files[folder_key] = {
                "last_scan": datetime.now().isoformat(),
                "files": {}
            }
        
        # Scan current files
        current_files = {}
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions:
                    file_path = os.path.join(root, file)
                    current_files[file_path] = self._get_file_info(file_path)
        
        # Compare with tracked files
        tracked_files = self.tracked_files[folder_key].get("files", {})
        
        for file_path, file_info in current_files.items():
            tracked_file_info = tracked_files.get(file_path)
            
            # Check if file is new or changed
            if (tracked_file_info is None or 
                tracked_file_info.get("hash") != file_info["hash"] or
                tracked_file_info.get("mtime") != file_info["mtime"] or
                tracked_file_info.get("size") != file_info["size"]):
                
                new_or_changed.append(file_path)
                print(f"📄 New/Changed file detected: {os.path.basename(file_path)}")
        
        # Check for deleted files
        deleted_files = []
        for tracked_path in tracked_files.keys():
            if tracked_path not in current_files:
                deleted_files.append(tracked_path)
                print(f"🗑️ Deleted file detected: {os.path.basename(tracked_path)}")
        
        # Update tracking data
        self.tracked_files[folder_key]["files"] = current_files
        self.tracked_files[folder_key]["last_scan"] = datetime.now().isoformat()
        
        # Remove deleted files from tracking
        for deleted_file in deleted_files:
            if deleted_file in self.tracked_files[folder_key]["files"]:
                del self.tracked_files[folder_key]["files"][deleted_file]
        
        # Save updated tracking data
        self._save_tracking_data()
        
        return new_or_changed
    
    def get_all_tracked_files(self, folder_path: str) -> List[str]:
        """Get all currently tracked files in a folder"""
        folder_key = os.path.abspath(folder_path)
        if folder_key in self.tracked_files:
            tracked_files = self.tracked_files[folder_key].get("files", {})
            # Verify files still exist
            existing_files = []
            for file_path in tracked_files.keys():
                if os.path.exists(file_path):
                    existing_files.append(file_path)
                else:
                    print(f"⚠️ Tracked file no longer exists: {os.path.basename(file_path)}")
            return existing_files
        return []
    
    def get_folder_stats(self, folder_path: str) -> Dict:
        """Get statistics about tracked files in a folder"""
        folder_key = os.path.abspath(folder_path)
        if folder_key not in self.tracked_files:
            return {"total_files": 0, "last_scan": None}
        
        tracked_files = self.tracked_files[folder_key].get("files", {})
        last_scan = self.tracked_files[folder_key].get("last_scan")
        
        return {
            "total_files": len(tracked_files),
            "last_scan": last_scan,
            "folder_path": folder_path
        }
    
    def reset_folder_tracking(self, folder_path: str):
        """Reset tracking data for a specific folder"""
        folder_key = os.path.abspath(folder_path)
        if folder_key in self.tracked_files:
            del self.tracked_files[folder_key]
            self._save_tracking_data()
            print(f"🔄 Reset tracking data for folder: {folder_path}")
    
    def reset_all_tracking(self):
        """Reset all tracking data"""
        self.tracked_files = {}
        self._save_tracking_data()
        print("🔄 Reset all tracking data")
