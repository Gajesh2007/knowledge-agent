"""Module for managing conversation memory with summarization and persistence."""

import json
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.logging import logger

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    metadata: Dict[str, str]

@dataclass
class ConversationSummary:
    """Summary of a portion of conversation."""
    messages: List[Message]
    summary: str
    start_time: float
    end_time: float

class ConversationMemory:
    """Enhanced conversation memory with summarization and persistence."""
    
    def __init__(
        self,
        llm_handler: Optional['LLMHandler'] = None,
        session_id: Optional[str] = None,
        max_messages: int = 10,
        storage_dir: str = "./.conversations"
    ):
        """Initialize conversation memory.
        
        Args:
            llm_handler: LLM handler for generating summaries
            session_id: Optional session ID for persistence
            max_messages: Maximum number of recent messages to keep before summarizing
            storage_dir: Directory for storing conversation history
        """
        self.llm_handler = llm_handler
        self.session_id = session_id or str(uuid.uuid4())
        self.max_messages = max_messages
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory components
        self.recent_messages: List[Message] = []
        self.summaries: List[ConversationSummary] = []
        
        # Load existing session if provided
        if session_id:
            self._load_session()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, str]] = None):
        """Add a message to the conversation.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata about the message
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().timestamp(),
            metadata=metadata or {}
        )
        
        self.recent_messages.append(message)
        
        # Check if we need to summarize older messages
        if len(self.recent_messages) > self.max_messages:
            self._summarize_oldest_messages()
        
        # Save the updated conversation
        self._save_session()
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """Get conversation context for the LLM.
        
        Args:
            max_tokens: Optional maximum number of tokens to include
            
        Returns:
            Formatted conversation context
        """
        context = []
        
        # Add summaries first
        if self.summaries:
            context.append("Previous conversation summary:")
            for summary in self.summaries:
                context.append(summary.summary)
            context.append("\nRecent messages:")
        
        # Add recent messages
        for msg in self.recent_messages:
            context.append(f"{msg.role.title()}: {msg.content}")
        
        return "\n".join(context)
    
    def _summarize_oldest_messages(self, count: Optional[int] = None):
        """Summarize the oldest messages in the conversation.
        
        Args:
            count: Number of messages to summarize, defaults to half of max_messages
        """
        if not self.recent_messages:
            return
        
        count = count or (self.max_messages // 2)
        messages_to_summarize = self.recent_messages[:count]
        
        # Prepare messages for summarization
        context = "\n".join(
            f"{msg.role.title()}: {msg.content}"
            for msg in messages_to_summarize
        )
        
        prompt = (
            "Please provide a concise summary of this conversation segment. "
            "Focus on the key points, questions asked, and answers provided. "
            "The summary should help maintain context for future interactions."
        )
        
        try:
            summary = self.llm_handler.generate_response(prompt, [context])
            
            # Create and store the summary
            conversation_summary = ConversationSummary(
                messages=messages_to_summarize,
                summary=summary,
                start_time=messages_to_summarize[0].timestamp,
                end_time=messages_to_summarize[-1].timestamp
            )
            self.summaries.append(conversation_summary)
            
            # Remove summarized messages from recent list
            self.recent_messages = self.recent_messages[count:]
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {str(e)}")
    
    def _get_session_file(self) -> Path:
        """Get the path to the session storage file."""
        return self.storage_dir / f"{self.session_id}.json"
    
    def _save_session(self):
        """Save the current session to disk."""
        try:
            session_data = {
                'session_id': self.session_id,
                'recent_messages': [asdict(msg) for msg in self.recent_messages],
                'summaries': [
                    {
                        'messages': [asdict(msg) for msg in summary.messages],
                        'summary': summary.summary,
                        'start_time': summary.start_time,
                        'end_time': summary.end_time
                    }
                    for summary in self.summaries
                ]
            }
            
            with open(self._get_session_file(), 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save session: {str(e)}")
    
    def _load_session(self):
        """Load a session from disk."""
        session_file = self._get_session_file()
        if not session_file.exists():
            return
        
        try:
            with open(session_file) as f:
                session_data = json.load(f)
            
            # Restore recent messages
            self.recent_messages = [
                Message(**msg_data)
                for msg_data in session_data['recent_messages']
            ]
            
            # Restore summaries
            self.summaries = [
                ConversationSummary(
                    messages=[Message(**msg_data) for msg_data in summary_data['messages']],
                    summary=summary_data['summary'],
                    start_time=summary_data['start_time'],
                    end_time=summary_data['end_time']
                )
                for summary_data in session_data['summaries']
            ]
            
        except Exception as e:
            logger.error(f"Failed to load session: {str(e)}")
    
    @classmethod
    def list_sessions(cls, storage_dir: str = "./.conversations") -> List[Tuple[str, float]]:
        """List all available conversation sessions.
        
        Args:
            storage_dir: Directory containing session files
            
        Returns:
            List of (session_id, last_modified_timestamp) tuples
        """
        storage_path = Path(storage_dir)
        if not storage_path.exists():
            return []
        
        sessions = []
        for file in storage_path.glob("*.json"):
            try:
                sessions.append((
                    file.stem,
                    file.stat().st_mtime
                ))
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x[1], reverse=True) 