"""
Enhanced colored logging with clean formatting and component-based categorization
"""

import logging
import re
from typing import Optional

import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Clean formatter with component-based coloring and enhanced readability"""

    LEVEL_COLORS = {
        'DEBUG': Fore.CYAN + Style.DIM,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW + Style.BRIGHT,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }

    COMPONENT_COLORS = {
        'main': Fore.BLUE + Style.BRIGHT,
        'run': Fore.BLUE + Style.BRIGHT,
        'setup': Fore.MAGENTA + Style.BRIGHT,
        'auth_utils': Fore.YELLOW + Style.DIM,
        'cas': Fore.YELLOW + Style.DIM,
        'document_processor': Fore.CYAN + Style.BRIGHT,
        'bulk_process': Fore.CYAN + Style.DIM,
        'ollama_client': Fore.MAGENTA,
        'embedding': Fore.MAGENTA + Style.DIM,
        'chromadb': Fore.GREEN + Style.DIM,
        'database': Fore.GREEN + Style.DIM,
        'api': Fore.BLUE,
        'fastapi': Fore.BLUE,
        'uvicorn': Fore.BLUE + Style.DIM,
        'pdfminer': Fore.WHITE + Style.DIM,
        'pdf': Fore.WHITE + Style.DIM,
        'unknown': Fore.WHITE,
    }

    LEVEL_SYMBOLS = {
        'DEBUG': '▪',
        'INFO': '●',
        'WARNING': '▲',
        'ERROR': '✗',
        'CRITICAL': '●●',
    }

    def _get_component_category(self, logger_name: str) -> str:
        """Determine component category from logger name"""
        if not logger_name or logger_name == 'root':
            return 'unknown'

        base_name = logger_name.split('.')[-1]

        if base_name in self.COMPONENT_COLORS:
            return base_name

        name_lower = base_name.lower()

        if any(pattern in name_lower for pattern in ['auth', 'cas', 'security']):
            return 'auth_utils'
        elif any(pattern in name_lower for pattern in ['document', 'process', 'pdf']):
            return 'document_processor'
        elif any(pattern in name_lower for pattern in ['ollama', 'llm', 'ai', 'model']):
            return 'ollama_client'
        elif any(pattern in name_lower for pattern in ['chroma', 'db', 'database']):
            return 'chromadb'
        elif any(pattern in name_lower for pattern in ['api', 'fastapi', 'endpoint']):
            return 'api'
        elif any(pattern in name_lower for pattern in ['setup', 'config', 'init']):
            return 'setup'

        return 'unknown'

    def _format_component(self, logger_name: str, component: str) -> str:
        """Format component name with color styling"""
        color = self.COMPONENT_COLORS.get(
            component, self.COMPONENT_COLORS['unknown'])
        display_name = f"...{logger_name[-17:]}" if len(
            logger_name) > 20 else logger_name
        return f"{color}[{display_name.upper()}]{Style.RESET_ALL}"

    def _enhance_message(self, message: str) -> str:
        """Clean and highlight important patterns in log message"""
        message = re.sub(r'\s+', ' ', message.strip())

        highlights = {
            r'\b(SUCCESS|SUCCESSFUL|COMPLETE|COMPLETED)\b':
                Fore.GREEN + Style.BRIGHT + r'\1' + Style.RESET_ALL,
            r'\b(FAILED|ERROR|FAILURE)\b':
                Fore.RED + Style.BRIGHT + r'\1' + Style.RESET_ALL,
            r'\b(WARNING|WARN)\b':
                Fore.YELLOW + Style.BRIGHT + r'\1' + Style.RESET_ALL,
            r'\b(\d+\.\d+)s\b':
                Fore.CYAN + Style.DIM + r'\1s' + Style.RESET_ALL,
            r'\b(\d+)\s*(chunks?|documents?|models?)\b':
                Fore.MAGENTA + r'\1 \2' + Style.RESET_ALL,
        }

        for pattern, replacement in highlights.items():
            message = re.sub(pattern, replacement,
                             message, flags=re.IGNORECASE)

        return message

    def format(self, record) -> str:
        """Format log record with colors and styling"""
        component = self._get_component_category(record.name)

        level_color = self.LEVEL_COLORS.get(record.levelname, '')
        level_symbol = self.LEVEL_SYMBOLS.get(record.levelname, '')

        timestamp = f"{Style.DIM}{self.formatTime(record, self.datefmt)}{Style.RESET_ALL}"
        component_name = self._format_component(record.name, component)
        level = f"{level_color}{level_symbol} {record.levelname:<8}{Style.RESET_ALL}"
        message = self._enhance_message(record.getMessage())

        return f"{timestamp} | {component_name} | {level} | {message}"


def setup_logging(level: int = logging.INFO, component_filter: Optional[str] = None) -> logging.Logger:
    """Configure colored logging for the application"""
    formatter = ColoredFormatter(datefmt='%H:%M:%S')

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if component_filter:
        console_handler.addFilter(
            lambda record: component_filter.lower() in record.name.lower()
        )

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    _suppress_noisy_loggers()

    return root_logger


def _suppress_noisy_loggers() -> None:
    """Set higher log levels for verbose third-party libraries"""
    noisy_loggers = {
        'pdfminer': logging.ERROR,
        'pdfplumber': logging.WARNING,
        'sentence_transformers': logging.WARNING,
        'torch': logging.WARNING,
        'transformers': logging.WARNING,
        'urllib3': logging.WARNING,
        'httpx': logging.WARNING,
    }

    for logger_name, level in noisy_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific component"""
    return logging.getLogger(name)
