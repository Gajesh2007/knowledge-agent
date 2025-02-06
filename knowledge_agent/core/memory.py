"""Conversation memory management for the knowledge base agent."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
import json
from pathlib import Path

from knowledge_agent.core.logging import logger

@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    query_context: Optional[List[Dict]] = None  # Store relevant chunks for context

class ConversationMemory:
    """Manages conversation history for multi-turn queries."""
    
    def __init__(self, max_messages: int = 10):
        """Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The user's message
        """
        message = Message(
            role="user",
            content=content,
            timestamp=datetime.now()
        )
        self._add_message(message)
    
    def add_assistant_message(self, content: str, context: Optional[List[Dict]] = None) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The assistant's response
            context: Optional list of context chunks used for the response
        """
        message = Message(
            role="assistant",
            content=content,
            timestamp=datetime.now(),
            query_context=context
        )
        self._add_message(message)
    
    def _add_message(self, message: Message) -> None:
        """Add a message and maintain the maximum history size.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)  # Remove oldest message
    
    def get_recent_context(self, num_messages: int = 3) -> str:
        """Get formatted context from recent messages.
        
        Args:
            num_messages: Number of recent message pairs to include
        
        Returns:
            Formatted conversation context
        """
        if not self.messages:
            return ""
        
        # Get recent messages
        recent = self.messages[-min(num_messages * 2, len(self.messages)):]
        
        # Format conversation context
        context_parts = []
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
    
    def save_to_file(self, file_path: Path) -> None:
        """Save the conversation history to a file.
        
        Args:
            file_path: Path to save the conversation
        """
        try:
            # Convert messages to serializable format
            history = []
            for msg in self.messages:
                history.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "query_context": msg.query_context
                })
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.debug(f"Conversation saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def load_from_file(self, file_path: Path) -> None:
        """Load conversation history from a file.
        
        Args:
            file_path: Path to load the conversation from
        """
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)
            
            # Convert back to Message objects
            self.messages = []
            for msg in history:
                self.messages.append(Message(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=datetime.fromisoformat(msg["timestamp"]),
                    query_context=msg.get("query_context")
                ))
            
            logger.debug(f"Conversation loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            self.messages = []  # Reset on error 