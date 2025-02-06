"""Vector store functionality for storing and retrieving document chunks."""
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

from chromadb import PersistentClient, Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from knowledge_agent.core.logging import logger
from knowledge_agent.core.retrieval import AdvancedRetrieval, SearchResult

class VectorStore:
    """Vector store for document chunks using ChromaDB."""
    
    def __init__(self, persist_directory: str):
        """Initialize the vector store."""
        self.persist_directory = persist_directory
        logger.debug(f"Initializing vector store at {persist_directory}")
        
        logger.debug("Loading HuggingFace embeddings model")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )
        
        # Create the persist directory if it doesn't exist
        persist_path = Path(persist_directory).resolve()
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with proper settings
        logger.debug("Initializing ChromaDB")
        try:
            # Create a PersistentClient with proper settings
            client = PersistentClient(
                path=str(persist_path),
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Initialize Chroma with the client
            self.db = Chroma(
                client=client,
                embedding_function=self.embeddings,
                collection_name="knowledge_base",
                persist_directory=str(persist_path)
            )
            logger.success("Vector store initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize vector store", exc_info=e)
            raise
        
        # Initialize advanced retrieval
        self.advanced_retrieval = None
    
    def add_chunk(self, chunk: Document) -> None:
        """Add a single document chunk to the vector store."""
        if not chunk:
            logger.warning("Empty chunk provided")
            return
        
        try:
            self.db.add_documents([chunk])
            # ChromaDB automatically persists changes
            logger.debug(f"Added chunk to vector store: {chunk.page_content[:100]}...")
        except Exception as e:
            logger.error("Error adding chunk to vector store", exc_info=e)
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.db.add_documents(documents)
            # ChromaDB automatically persists changes
            logger.success(f"Added {len(documents)} chunks to the vector store")
        except Exception as e:
            logger.error("Error adding documents to vector store", exc_info=e)
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, str]] = None,
        use_advanced: bool = True
    ) -> Union[List[Tuple[Document, float]], List[SearchResult]]:
        """Search for similar documents in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional metadata filters
            use_advanced: Whether to use advanced retrieval features
            
        Returns:
            List of (document, score) tuples or SearchResult objects
        """
        try:
            logger.debug(f"Performing similarity search for query: {query} (k={k})")
            
            if use_advanced:
                # Use advanced retrieval if requested
                if not self.advanced_retrieval:
                    self.advanced_retrieval = AdvancedRetrieval(self)
                
                results = self.advanced_retrieval.search(
                    query,
                    k=k,
                    metadata_filter=metadata_filter
                )
                
                # Get cluster summary for logging
                clusters = self.advanced_retrieval.get_cluster_summary(results)
                logger.debug("Search results by cluster:")
                for cluster_id, docs in clusters.items():
                    logger.debug(f"Cluster {cluster_id}: {', '.join(docs)}")
                
                return results
            else:
                # Use basic similarity search
                results = self.db.similarity_search_with_score(
                    query,
                    k=k,
                    filter=metadata_filter
                )
                logger.debug(f"Found {len(results)} results")
                return [(doc, score) for doc, score in results]
                
        except Exception as e:
            logger.error("Error searching vector store", exc_info=e)
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        try:
            logger.debug("Retrieving collection statistics")
            count = len(self.db.get())
            stats = {
                "total_chunks": count,
                "persist_directory": self.persist_directory,
            }
            logger.debug(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error("Error getting collection stats", exc_info=e)
            return {
                "total_chunks": 0,
                "persist_directory": self.persist_directory,
            } 