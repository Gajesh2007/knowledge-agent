"""Module for advanced retrieval and context management."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from knowledge_agent.core.vector_store import VectorStore
from knowledge_agent.core.logging import logger
from knowledge_agent.core.conversation import ConversationMemory

@dataclass
class SearchResult:
    """Enhanced search result with additional context."""
    document: Document
    relevance_score: float
    semantic_cluster: Optional[int] = None
    context_score: Optional[float] = None
    conversation_relevance: Optional[float] = None

class AdvancedRetrieval:
    """Enhanced retrieval system with semantic clustering and context awareness."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        memory: Optional[ConversationMemory] = None,
        num_clusters: int = 3
    ):
        """Initialize the advanced retrieval system.
        
        Args:
            vector_store: Vector store for document retrieval
            memory: Optional conversation memory for context
            num_clusters: Number of semantic clusters to use
        """
        self.vector_store = vector_store
        self.memory = memory
        self.num_clusters = num_clusters
        self.embeddings = vector_store.embeddings
        
        # Cache for embeddings and clusters
        self._document_embeddings = {}
        self._clusters = None
    
    def search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, str]] = None,
        min_relevance: float = 0.3
    ) -> List[SearchResult]:
        """Perform an enhanced search with semantic clustering and context awareness.
        
        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional metadata filters
            min_relevance: Minimum relevance score threshold
            
        Returns:
            List of enhanced search results
        """
        # Get base results from vector store
        base_results = self.vector_store.similarity_search(
            query,
            k=k * 2,  # Get more results initially for filtering
            metadata_filter=metadata_filter
        )
        
        if not base_results:
            return []
        
        # Convert to SearchResult objects
        search_results = [
            SearchResult(doc, score)
            for doc, score in base_results
            if score >= min_relevance
        ]
        
        # Apply semantic clustering
        self._cluster_results(search_results)
        
        # Apply conversation context scoring if available
        if self.memory:
            self._score_conversation_context(search_results)
        
        # Sort by combined relevance
        search_results = self._rank_results(search_results)
        
        return search_results[:k]
    
    def _cluster_results(self, results: List[SearchResult]):
        """Apply semantic clustering to search results."""
        if not results:
            return
        
        # Get or compute document embeddings
        embeddings = []
        for result in results:
            doc_id = result.document.metadata.get('name', result.document.page_content[:100])
            if doc_id not in self._document_embeddings:
                self._document_embeddings[doc_id] = self.embeddings.embed_query(
                    result.document.page_content
                )
            embeddings.append(self._document_embeddings[doc_id])
        
        # Perform clustering
        if len(results) >= self.num_clusters:
            kmeans = KMeans(
                n_clusters=min(self.num_clusters, len(results)),
                random_state=42
            )
            clusters = kmeans.fit_predict(embeddings)
            
            # Assign clusters to results
            for result, cluster in zip(results, clusters):
                result.semantic_cluster = int(cluster)
        else:
            # Not enough results for clustering
            for result in results:
                result.semantic_cluster = 0
    
    def _score_conversation_context(self, results: List[SearchResult]):
        """Score results based on conversation context."""
        if not self.memory or not results:
            return
        
        # Get conversation context
        context = self.memory.get_context()
        if not context:
            return
        
        # Embed context
        context_embedding = self.embeddings.embed_query(context)
        
        # Score each result against context
        for result in results:
            doc_id = result.document.metadata.get('name', result.document.page_content[:100])
            if doc_id not in self._document_embeddings:
                self._document_embeddings[doc_id] = self.embeddings.embed_query(
                    result.document.page_content
                )
            
            # Calculate cosine similarity with context
            similarity = cosine_similarity(
                [context_embedding],
                [self._document_embeddings[doc_id]]
            )[0][0]
            
            result.conversation_relevance = float(similarity)
    
    def _rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank results using all available scores."""
        for result in results:
            # Combine scores (with weights)
            base_score = result.relevance_score
            context_score = result.conversation_relevance or 0.0
            
            # Calculate combined score
            # Weight: 70% base relevance, 30% conversation context
            result.context_score = (0.7 * base_score) + (0.3 * context_score)
        
        # Sort by combined score
        return sorted(results, key=lambda x: x.context_score or 0.0, reverse=True)
    
    def get_cluster_summary(self, results: List[SearchResult]) -> Dict[int, List[str]]:
        """Get a summary of documents in each semantic cluster.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary mapping cluster IDs to lists of document titles/names
        """
        clusters = {}
        for result in results:
            if result.semantic_cluster is not None:
                cluster_id = result.semantic_cluster
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                # Get a meaningful name for the document
                doc_name = (
                    result.document.metadata.get('name')
                    or result.document.metadata.get('file_path')
                    or result.document.page_content[:50] + "..."
                )
                clusters[cluster_id].append(doc_name)
        
        return clusters 