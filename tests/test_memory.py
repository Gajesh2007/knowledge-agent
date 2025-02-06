"""Tests for conversation memory functionality."""
import json
from datetime import datetime
from pathlib import Path
import tempfile

import pytest

from knowledge_agent.core.memory import ConversationMemory, Message

def test_conversation_memory_initialization():
    """Test that ConversationMemory initializes correctly."""
    memory = ConversationMemory(max_messages=5)
    assert len(memory.messages) == 0
    assert memory.max_messages == 5

def test_add_messages():
    """Test adding messages to conversation memory."""
    memory = ConversationMemory(max_messages=3)
    
    # Add messages
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi there!")
    memory.add_user_message("How are you?")
    
    assert len(memory.messages) == 3
    assert memory.messages[0].role == "user"
    assert memory.messages[0].content == "Hello"
    assert isinstance(memory.messages[0].timestamp, datetime)

def test_max_messages_limit():
    """Test that conversation memory respects max_messages limit."""
    memory = ConversationMemory(max_messages=2)
    
    memory.add_user_message("First")
    memory.add_assistant_message("Second")
    memory.add_user_message("Third")
    
    assert len(memory.messages) == 2
    assert memory.messages[0].content == "Second"
    assert memory.messages[1].content == "Third"

def test_get_recent_context():
    """Test getting formatted conversation context."""
    memory = ConversationMemory()
    
    # Add a conversation
    memory.add_user_message("What is Python?")
    memory.add_assistant_message("Python is a programming language.")
    memory.add_user_message("Show me an example.")
    memory.add_assistant_message("Here's a simple example:\n```python\nprint('Hello')\n```")
    
    # Get context with different limits
    context_2 = memory.get_recent_context(num_messages=2)
    context_1 = memory.get_recent_context(num_messages=1)
    
    assert len(context_2.split("\n\n")) == 4  # 4 messages
    assert len(context_1.split("\n\n")) == 2  # 2 messages
    assert "What is Python?" in context_2
    assert "What is Python?" not in context_1

def test_clear_memory():
    """Test clearing conversation memory."""
    memory = ConversationMemory()
    
    memory.add_user_message("Test")
    assert len(memory.messages) == 1
    
    memory.clear()
    assert len(memory.messages) == 0

def test_save_and_load():
    """Test saving and loading conversation history."""
    memory = ConversationMemory()
    
    # Add some messages
    memory.add_user_message("Test question")
    memory.add_assistant_message("Test answer", context=[{"text": "context"}])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        temp_path = Path(tf.name)
        memory.save_to_file(temp_path)
    
    # Create new memory and load
    new_memory = ConversationMemory()
    new_memory.load_from_file(temp_path)
    
    # Clean up
    temp_path.unlink()
    
    # Verify loaded content
    assert len(new_memory.messages) == 2
    assert new_memory.messages[0].content == "Test question"
    assert new_memory.messages[1].query_context == [{"text": "context"}]

def test_invalid_file_handling():
    """Test handling of invalid files when loading."""
    memory = ConversationMemory()
    
    # Try to load non-existent file
    memory.load_from_file(Path("nonexistent.json"))
    assert len(memory.messages) == 0
    
    # Try to save to invalid location
    memory.save_to_file(Path("/invalid/path/file.json"))
    # Should not raise exception, just log error 