"""Vector store functionality for storing and retrieving document chunks."""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from chromadb import PersistentClient, Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from knowledge_agent.core.logging import logger

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
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        logger.debug("Initializing ChromaDB")
        try:
            self.db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
            logger.success("Vector store initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize vector store", exc_info=e)
            raise
    
    def add_documents(self, documents: List[dict]) -> None:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.db.add_documents(documents)
            logger.success(f"Added {len(documents)} chunks to the vector store")
        except Exception as e:
            logger.error("Error adding documents to vector store", exc_info=e)
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[dict, float]]:
        """Search for similar documents in the vector store."""
        try:
            logger.debug(f"Performing similarity search for query: {query} (k={k})")
            results = self.db.similarity_search_with_score(query, k=k)
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