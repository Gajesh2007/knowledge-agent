"""Tests for advanced retrieval functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from knowledge_agent.core.retrieval import AdvancedRetrieval, SearchResult
from knowledge_agent.core.vector_store import VectorStore
from langchain_core.documents import Document

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

def test_search_basic(mock_vector_store, sample_documents):
    """Test basic search functionality."""
    # Setup
    retrieval = AdvancedRetrieval(mock_vector_store)
    mock_vector_store.similarity_search.return_value = [
        (doc, 0.8) for doc in sample_documents
    ]
    
    # Execute
    results = retrieval.search("test query", k=2)
    
    # Assert
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.relevance_score == 0.8 for r in results)
    assert results[0].document.metadata["name"] == "doc1"

def test_search_with_metadata_filter(mock_vector_store, sample_documents):
    """Test search with metadata filtering."""
    # Setup
    retrieval = AdvancedRetrieval(mock_vector_store)
    mock_vector_store.similarity_search.return_value = [
        (sample_documents[0], 0.8)
    ]
    
    # Execute
    results = retrieval.search(
        "test query",
        metadata_filter={"type": "code"}
    )
    
    # Assert
    assert len(results) == 1
    assert results[0].document.metadata["type"] == "code"

def test_search_with_min_relevance(mock_vector_store, sample_documents):
    """Test search with minimum relevance threshold."""
    # Setup
    retrieval = AdvancedRetrieval(mock_vector_store)
    mock_vector_store.similarity_search.return_value = [
        (sample_documents[0], 0.8),
        (sample_documents[1], 0.3)
    ]
    
    # Execute
    results = retrieval.search("test query", min_relevance=0.5)
    
    # Assert
    assert len(results) == 1
    assert results[0].relevance_score == 0.8

def test_cluster_results(mock_vector_store, sample_documents):
    """Test semantic clustering of results."""
    # Setup
    retrieval = AdvancedRetrieval(mock_vector_store, num_clusters=2)
    mock_vector_store.similarity_search.return_value = [
        (doc, 0.8) for doc in sample_documents
    ]
    
    # Execute
    results = retrieval.search("test query")
    
    # Assert
    assert all(hasattr(r, 'semantic_cluster') for r in results)
    assert all(r.semantic_cluster in {0, 1} for r in results)

def test_get_cluster_summary(mock_vector_store, sample_documents):
    """Test cluster summary generation."""
    # Setup
    retrieval = AdvancedRetrieval(mock_vector_store, num_clusters=2)
    mock_vector_store.similarity_search.return_value = [
        (doc, 0.8) for doc in sample_documents
    ]
    
    # Execute
    results = retrieval.search("test query")
    summary = retrieval.get_cluster_summary(results)
    
    # Assert
    assert isinstance(summary, dict)
    assert all(isinstance(cluster_id, int) for cluster_id in summary.keys())
    assert all(isinstance(docs, list) for docs in summary.values()) 