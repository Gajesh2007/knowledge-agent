"""Vector store functionality for storing and retrieving document chunks."""
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

from chromadb import PersistentClient, Settings
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_agent.core.logging import logger
from knowledge_agent.core.retrieval import AdvancedRetrieval, SearchResult

class VectorStore:
    """Vector store for document chunks using ChromaDB."""
    
    def __init__(self, persist_directory: str):
        """Initialize the vector store."""
        self.persist_directory = persist_directory
        logger.debug(f"Initializing vector store at {persist_directory}")
        
        # Initialize embeddings based on configuration
        self.embeddings = self._initialize_embeddings()
        
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
    
    def _initialize_embeddings(self) -> Embeddings:
        """Initialize embeddings based on configuration.
        
        Returns:
            Embeddings: Configured embedding model
        """
        embedding_type = os.getenv("EMBEDDING_TYPE", "hosted").lower()
        model_name = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-7B-instruct")
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        endpoint_url = os.getenv("HUGGINGFACE_ENDPOINT_URL")

        # Determine device and check CUDA compatibility
        import torch
        import subprocess
        import re

        def get_cuda_version():
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
                cuda_version = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi)
                if cuda_version:
                    return cuda_version.group(1)
            except:
                return None
            return None

        def check_torch_cuda_compatibility():
            cuda_version = get_cuda_version()
            if cuda_version is None:
                logger.warning("CUDA not found on system")
                return False

            if not hasattr(torch.version, 'cuda') or not torch.version.cuda:
                logger.error(f"PyTorch not compiled with CUDA support. System CUDA version: {cuda_version}")
                logger.error("Please install PyTorch with CUDA support using one of these commands:")
                major_version = cuda_version.split('.')[0]
                minor_version = cuda_version.split('.')[1]
                cuda_version_str = f"{major_version}{minor_version}"
                logger.error(f"For system Python: pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_version_str}")
                logger.error(f"For Poetry environment: poetry run pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_version_str}")
                logger.error(f"Current environment: {'Poetry' if 'POETRY_ACTIVE' in os.environ else 'System Python'}")
                return False

            # Check if CUDA is actually working
            if not torch.cuda.is_available():
                logger.error("PyTorch CUDA support is installed but CUDA is not available")
                logger.error("This might be due to:")
                logger.error("1. NVIDIA drivers not properly installed")
                logger.error("2. CUDA toolkit not properly installed")
                logger.error("3. Incompatible CUDA versions between PyTorch and system")
                return False

            logger.info(f"System CUDA version: {cuda_version}, PyTorch CUDA version: {torch.version.cuda}")
            logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
            return True

        # Check CUDA compatibility
        cuda_compatible = check_torch_cuda_compatibility()
        device = os.getenv("EMBEDDING_DEVICE", "cuda" if cuda_compatible else "cpu")
        
        if device.startswith("cuda") and not cuda_compatible:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        logger.debug(f"Initializing embeddings with type: {embedding_type} on device: {device}")
        
        try:
            if embedding_type == "hosted":
                if not api_key:
                    raise ValueError("HUGGINGFACE_API_KEY is required for hosted embeddings")
                    
                if endpoint_url:
                    logger.debug("Using HuggingFace Inference Endpoint")
                    return HuggingFaceEndpointEmbeddings(
                        endpoint_url=endpoint_url,
                        huggingfacehub_api_token=api_key,
                        task="sentence-similarity",  # Correct task for GTE models in endpoints
                        retry_on_error=True,
                    )
                else:
                    logger.debug("Using HuggingFace Inference API")
                    return HuggingFaceEmbeddings(
                        model_name=model_name,
                        api_key=api_key,
                        task="feature-extraction",  # Correct task for Inference API
                        encode_kwargs={"batch_size": 32}
                    )
            else:
                logger.debug("Using local HuggingFace embeddings")
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": device},
                    encode_kwargs={"device": device, "batch_size": 32}
                )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise ValueError(f"Failed to initialize embeddings: {str(e)}") from e
    
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
        use_advanced: bool = True,
        _from_advanced: bool = False  # Internal flag to prevent recursion
    ) -> Union[List[Tuple[Document, float]], List[SearchResult]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            metadata_filter: Optional metadata filter
            use_advanced: Whether to use advanced retrieval
            _from_advanced: Internal flag to indicate if this call is from AdvancedRetrieval
            
        Returns:
            List of (document, score) tuples or SearchResults
        """
        try:
            # Get raw results from vector store
            results = self.db.similarity_search_with_score(
                query,
                k=k,
                filter=metadata_filter
            )
            
            # Convert to document tuples
            doc_results = []
            for doc, score in results:
                doc_results.append((doc, float(score)))
            
            if use_advanced and not _from_advanced:  # Only use advanced if not already in advanced
                # Use advanced retrieval if requested
                if not self.advanced_retrieval:
                    self.advanced_retrieval = AdvancedRetrieval(self)
                
                results = self.advanced_retrieval.search(
                    query,
                    k=k,
                    metadata_filter=metadata_filter
                )
                
                return results
                
            return doc_results
            
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