"""Shared test fixtures and configuration."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.vector_store import VectorStore
from langchain_core.documents import Document

@pytest.fixture
def mock_llm_handler():
    """Create a mock LLM handler."""
    handler = Mock(spec=LLMHandler)
    handler.generate_response.return_value = "Generated content"
    return handler

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock(spec=VectorStore)
    store.embeddings = Mock()
    store.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return store

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Test content 1",
            metadata={"name": "doc1", "type": "code"}
        ),
        Document(
            page_content="Test content 2",
            metadata={"name": "doc2", "type": "documentation"}
        ),
    ]

@pytest.fixture
def test_repo_path(tmp_path):
    """Create a test repository structure."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Create some test files
    (repo_path / "src").mkdir()
    (repo_path / "docs").mkdir()
    
    with open(repo_path / "src" / "test.py", "w") as f:
        f.write("def test_function():\n    pass\n")
    
    with open(repo_path / "docs" / "readme.md", "w") as f:
        f.write("# Test Documentation\n")
    
    return repo_path 