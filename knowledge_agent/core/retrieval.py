"""Module for advanced retrieval and context management."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from knowledge_agent.core.logging import logger

if TYPE_CHECKING:
    from knowledge_agent.core.conversation import ConversationMemory
    from knowledge_agent.core.vector_store import VectorStore

@dataclass
class SearchResult:
    """Enhanced search result with additional context."""
    document: Document
    relevance_score: float
    semantic_cluster: Optional[int] = None
    related_results: List[Document] = None
    context_snippets: List[str] = None
    relationship_chain: List[str] = None
    semantic_summary: Optional[str] = None

class AdvancedRetrieval:
    """Enhanced retrieval system with semantic clustering and context awareness."""
    
    def __init__(
        self,
        vector_store: 'VectorStore',
        memory: Optional['ConversationMemory'] = None,
        num_clusters: int = 3,
        min_relevance: float = 0.3,
        max_context_results: int = 5,
        max_relationship_depth: int = 2
    ):
        """Initialize the advanced retrieval system."""
        self.vector_store = vector_store
        self.memory = memory
        self.num_clusters = num_clusters
        self.min_relevance = min_relevance
        self.max_context_results = max_context_results
        self.max_relationship_depth = max_relationship_depth
        self.embeddings = vector_store.embeddings
        
        # Cache for embeddings and clusters
        self._document_embeddings = {}
        self._clusters = None
    
    def search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, str]] = None,
        min_relevance: Optional[float] = None,
        search_strategy: str = "comprehensive"
    ) -> List[SearchResult]:
        """
        Perform enhanced search with multiple strategies.
        
        Args:
            query: Search query
            k: Number of results
            metadata_filter: Optional metadata filter
            min_relevance: Minimum relevance score (overrides instance default)
            search_strategy: One of "fast", "comprehensive", or "deep"
                - fast: Basic similarity search
                - comprehensive: Multiple search passes with relationship exploration
                - deep: Exhaustive search with full relationship traversal
        """
        min_relevance = min_relevance or self.min_relevance
        
        if search_strategy == "fast":
            return self._basic_search(query, k, metadata_filter, min_relevance)
        elif search_strategy == "deep":
            return self._deep_search(query, k, metadata_filter, min_relevance)
        else:  # comprehensive (default)
            return self._comprehensive_search(query, k, metadata_filter, min_relevance)
    
    def _basic_search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, str]],
        min_relevance: float
    ) -> List[SearchResult]:
        """Perform basic similarity search."""
        # Get raw results
        results = self.vector_store.similarity_search(
            query,
            k=k * 2,  # Get more results for filtering
            metadata_filter=metadata_filter,
            _from_advanced=True  # Prevent recursion
        )
        
        # Filter and convert results
        search_results = []
        for doc, score in results:
            if score >= min_relevance:
                search_results.append(SearchResult(
                    document=doc,
                    relevance_score=score,
                    semantic_cluster=None,
                    related_results=[],
                    context_snippets=[],
                    relationship_chain=[],
                    semantic_summary=None
                ))
        
        # Sort by relevance and limit
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results[:k]
    
    def _comprehensive_search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, str]],
        min_relevance: float
    ) -> List[SearchResult]:
        """Perform comprehensive search with relationship exploration."""
        # First get basic results
        results = self._basic_search(query, k, metadata_filter, min_relevance)
        if not results:
            return []
        
        # Enhance results with related information
        enhanced_results = []
        for result in results:
            # Get related documents through relationships
            related_docs = self._get_related_documents(result.document, depth=1)
            
            # Get context snippets
            context_snippets = self._extract_context_snippets(result.document)
            
            # Build relationship chain
            relationship_chain = self._build_relationship_chain(result.document)
            
            # Generate semantic summary if possible
            semantic_summary = self._generate_summary(result.document)
            
            # Create enhanced result
            enhanced_results.append(SearchResult(
                document=result.document,
                relevance_score=result.relevance_score,
                semantic_cluster=result.semantic_cluster,
                related_results=related_docs,
                context_snippets=context_snippets,
                relationship_chain=relationship_chain,
                semantic_summary=semantic_summary
            ))
        
        # Perform semantic clustering
        if len(enhanced_results) > 1:
            self._cluster_results(enhanced_results)
        
        return enhanced_results
    
    def _deep_search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, str]],
        min_relevance: float
    ) -> List[SearchResult]:
        """Perform deep search with exhaustive relationship traversal."""
        # Start with comprehensive search
        results = self._comprehensive_search(query, k * 2, metadata_filter, min_relevance)
        if not results:
            return []
        
        # Collect all related documents up to max depth
        all_docs = set()
        for result in results:
            # Get deeply related documents
            related = self._get_related_documents(
                result.document,
                depth=self.max_relationship_depth
            )
            all_docs.update(related)
        
        # Re-rank all documents
        all_results = []
        query_embedding = self.embeddings.embed_query(query)
        
        for doc in all_docs:
            # Get document embedding
            if doc.page_content not in self._document_embeddings:
                self._document_embeddings[doc.page_content] = (
                    self.embeddings.embed_documents([doc.page_content])[0]
                )
            doc_embedding = self._document_embeddings[doc.page_content]
            
            # Calculate similarity
            similarity = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]
            
            if similarity >= min_relevance:
                # Create search result with full context
                result = SearchResult(
                    document=doc,
                    relevance_score=similarity,
                    semantic_cluster=None,
                    related_results=self._get_related_documents(doc, depth=1),
                    context_snippets=self._extract_context_snippets(doc),
                    relationship_chain=self._build_relationship_chain(doc),
                    semantic_summary=self._generate_summary(doc)
                )
                all_results.append(result)
        
        # Sort by relevance and cluster
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        if len(all_results) > 1:
            self._cluster_results(all_results)
        
        return all_results[:k]
    
    def _get_related_documents(
        self,
        document: Document,
        depth: int = 1
    ) -> List[Document]:
        """Get related documents through relationships."""
        related = []
        
        # Extract relationship information
        relationships = {}
        for key, value in document.metadata.items():
            if key.endswith('_entities') or key in ('dependencies', 'related_files'):
                relationships[key] = set(value.split(',')) if value else set()
        
        # Collect related documents
        for rel_type, entities in relationships.items():
            for entity in entities:
                # Search for the related entity
                related_results = self.vector_store.similarity_search(
                    entity,
                    k=1,
                    metadata_filter={'name': entity},
                    _from_advanced=True
                )
                if related_results:
                    related.append(related_results[0][0])
                    
                    # Recurse if needed
                    if depth > 1:
                        deeper_related = self._get_related_documents(
                            related_results[0][0],
                            depth=depth-1
                        )
                        related.extend(deeper_related)
        
        return related[:self.max_context_results]
    
    def _extract_context_snippets(self, document: Document) -> List[str]:
        """Extract relevant context snippets from the document."""
        snippets = []
        
        # Add surrounding context if available
        if 'surrounding_context' in document.metadata:
            snippets.append(document.metadata['surrounding_context'])
        
        # Add file context if available
        if 'file_context' in document.metadata:
            snippets.append(document.metadata['file_context'])
        
        # Add module documentation if available
        if 'module_doc' in document.metadata:
            snippets.append(document.metadata['module_doc'])
        
        return snippets
    
    def _build_relationship_chain(self, document: Document) -> List[str]:
        """Build a chain of relationships for the document."""
        chain = []
        
        # Start with parent relationship
        if document.metadata.get('parent'):
            chain.append(f"Child of {document.metadata['parent']}")
        
        # Add package/module information
        if document.metadata.get('package'):
            chain.append(f"In package {document.metadata['package']}")
        if document.metadata.get('module'):
            chain.append(f"In module {document.metadata['module']}")
        
        # Add implementation relationships
        if document.metadata.get('implements'):
            chain.append(f"Implements {document.metadata['implements']}")
        
        # Add usage relationships
        if document.metadata.get('used_by'):
            chain.append(f"Used by {document.metadata['used_by']}")
        
        return chain
    
    def _generate_summary(self, document: Document) -> Optional[str]:
        """Generate or retrieve semantic summary."""
        # Use existing summary if available
        if document.metadata.get('semantic_summary'):
            return document.metadata['semantic_summary']
        
        # Combine available documentation
        summary_parts = []
        if document.metadata.get('docstring'):
            summary_parts.append(document.metadata['docstring'])
        if document.metadata.get('semantic_type'):
            summary_parts.append(
                f"This is a {document.metadata['semantic_type']} "
                f"in {document.metadata.get('language', 'unknown')}"
            )
        
        return " ".join(summary_parts) if summary_parts else None
    
    def _cluster_results(self, results: List[SearchResult]) -> None:
        """Perform semantic clustering on search results."""
        if len(results) < 2:
            return
        
        # Get embeddings for all documents
        embeddings = []
        for result in results:
            if result.document.page_content not in self._document_embeddings:
                self._document_embeddings[result.document.page_content] = (
                    self.embeddings.embed_documents([result.document.page_content])[0]
                )
            embeddings.append(self._document_embeddings[result.document.page_content])
        
        # Perform clustering
        num_clusters = min(self.num_clusters, len(results))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Assign clusters
        for result, cluster in zip(results, clusters):
            result.semantic_cluster = int(cluster)
    
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