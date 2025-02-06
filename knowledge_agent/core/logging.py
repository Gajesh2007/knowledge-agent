"""Logging functionality for the knowledge base agent."""
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for different log levels
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "debug": "grey50",
    "success": "green",
})

class KnowledgeAgentLogger:
    """Logger for the knowledge base agent with rich formatting and file output."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the logger.
        
        Args:
            log_file: Optional path to the log file. If not provided,
                     logs will be written to .logs/knowledge_agent_{timestamp}.log
        """
        # Create console with custom theme
        self.console = Console(theme=CUSTOM_THEME)
        
        # Create .logs directory if it doesn't exist
        log_dir = Path(".logs")
        log_dir.mkdir(exist_ok=True)
        
        # Generate default log file name if not provided
        if not log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"knowledge_agent_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(console=self.console, rich_tracebacks=True),
                logging.FileHandler(log_file),
            ],
        )
        
        self.logger = logging.getLogger("knowledge_agent")
        self.log_file = log_file
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.logger.info(message, extra=kwargs)
        self.console.print(f"[info]{message}[/info]")
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, extra=kwargs)
        self.console.print(f"[warning]{message}[/warning]")
    
    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log an error message with optional exception info."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
        if exc_info:
            self.console.print(f"[error]{message}: {str(exc_info)}[/error]")
        else:
            self.console.print(f"[error]{message}[/error]")
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, extra=kwargs)
        if os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG":
            self.console.print(f"[debug]{message}[/debug]")
    
    def success(self, message: str, **kwargs):
        """Log a success message."""
        self.logger.info(f"SUCCESS: {message}", extra=kwargs)
        self.console.print(f"[success]{message}[/success]")
    
    def progress(self, message: str, total: Optional[int] = None):
        """Create and return a progress bar."""
        return self.console.status(message, spinner="dots") if total is None else self.console.progress()
    
    @contextmanager
    def section(self, title: str, message: str = ""):
        """Create a section with a title and message.
        
        Can be used as a context manager:
        with logger.section("Title", "Message"):
            # do something
        """
        self.console.rule(f"[bold]{title}")
        if message:
            self.console.print(message)
        try:
            yield
        finally:
            self.console.rule()

# Create a global logger instance
logger = KnowledgeAgentLogger() 