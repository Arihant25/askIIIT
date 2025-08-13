"""
Log capture system for admin panel
Captures logs in memory and provides API access to recent log entries
"""

import logging
import time
from collections import deque
from typing import List, Dict, Optional
import threading
from datetime import datetime
import re


class LogCapture:
    """Thread-safe log capture for admin panel access"""
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.logs = deque(maxlen=max_entries)
        self.lock = threading.Lock()
    
    def add_log(self, record: logging.LogRecord, formatted_message: str):
        """Add a log entry to the capture buffer"""
        with self.lock:
            # Parse component from logger name
            component = self._get_component_from_name(record.name)
            
            # Clean the formatted message of color codes
            clean_message = self._strip_ansi_codes(formatted_message)
            
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': clean_message,
                'source': component,
                'logger_name': record.name,
                'raw_message': record.getMessage()
            }
            
            self.logs.append(log_entry)
    
    def get_recent_logs(self, count: Optional[int] = None, level_filter: Optional[str] = None) -> List[Dict]:
        """Get recent log entries"""
        with self.lock:
            logs_list = list(self.logs)
        
        # Filter by log level if specified
        if level_filter and level_filter.strip():
            level_upper = level_filter.strip().upper()
            # Handle both 'WARN' and 'WARNING' variations
            if level_upper == 'WARN':
                level_upper = 'WARNING'
            logs_list = [log for log in logs_list if log['level'] == level_upper]
        
        # Return most recent entries
        if count and count > 0:
            return logs_list[-count:] if len(logs_list) > count else logs_list
        
        return logs_list
    
    def _get_component_from_name(self, logger_name: str) -> str:
        """Extract component name from logger name"""
        if not logger_name or logger_name == 'root':
            return 'system'
        
        # Get base name
        base_name = logger_name.split('.')[-1]
        
        # Map common logger names to components
        component_mapping = {
            'main': 'main',
            'run': 'server',
            'bulk_process': 'bulk_process',
            'document_processor': 'document_processor',
            'ollama_client': 'ollama',
            'auth_utils': 'auth',
            'uvicorn': 'server',
            'fastapi': 'api'
        }
        
        return component_mapping.get(base_name, base_name)
    
    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI color codes from text"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text).strip()


class AdminLogHandler(logging.Handler):
    """Custom log handler that captures logs for admin panel"""
    
    def __init__(self, log_capture: LogCapture):
        super().__init__()
        self.log_capture = log_capture
    
    def emit(self, record: logging.LogRecord):
        """Capture log record"""
        try:
            # Format the message using the main formatter if available
            formatted_msg = self.format(record)
            self.log_capture.add_log(record, formatted_msg)
        except Exception:
            # Don't let log capture errors break the application
            pass


# Global log capture instance
_log_capture = LogCapture()


def get_log_capture() -> LogCapture:
    """Get the global log capture instance"""
    return _log_capture


def setup_log_capture():
    """Set up log capture for the root logger"""
    handler = AdminLogHandler(_log_capture)
    
    # Set up a simple formatter for the admin handler
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    return handler


def get_recent_logs(count: int = 100, level_filter: Optional[str] = None) -> List[Dict]:
    """Convenience function to get recent logs"""
    return _log_capture.get_recent_logs(count, level_filter)