"""
Processing status tracking for bulk operations
"""

import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime


class ProcessingStatus:
    """Thread-safe processing status tracker"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self._is_processing = False
        self._current_document = None
        self._progress = 0
        self._total_files = 0
        self._queue_size = 0
        self._start_time = None
        self._status_message = "Idle"
    
    def start_processing(self, total_files: int = 0):
        """Mark processing as started"""
        with self.lock:
            self._is_processing = True
            self._total_files = total_files
            self._queue_size = total_files
            self._progress = 0
            self._start_time = datetime.now()
            self._status_message = "Processing started"
    
    def update_progress(self, current_document: str, processed_count: int):
        """Update processing progress"""
        with self.lock:
            self._current_document = current_document
            self._progress = processed_count
            if self._total_files > 0:
                self._queue_size = max(0, self._total_files - processed_count)
            self._status_message = f"Processing {current_document}"
    
    def finish_processing(self, success: bool = True):
        """Mark processing as finished"""
        with self.lock:
            self._is_processing = False
            self._current_document = None
            self._progress = self._total_files if success else self._progress
            self._queue_size = 0
            self._status_message = "Processing completed" if success else "Processing failed"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        with self.lock:
            progress_percent = 0
            if self._total_files > 0 and self._is_processing:
                progress_percent = (self._progress / self._total_files) * 100
            
            return {
                "is_indexing": self._is_processing,
                "current_document": self._current_document,
                "progress": progress_percent,
                "processed_count": self._progress,
                "total_count": self._total_files,
                "queue_size": self._queue_size,
                "status_message": self._status_message,
                "start_time": self._start_time.isoformat() if self._start_time else None
            }


# Global processing status instance
_processing_status = ProcessingStatus()


def get_processing_status() -> ProcessingStatus:
    """Get the global processing status instance"""
    return _processing_status