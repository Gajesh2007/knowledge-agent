"""Code and documentation ingestion functionality."""
import os
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from knowledge_agent.core.logging import logger

SUPPORTED_CODE_EXTENSIONS = {
    ".py": "Python",
    ".go": "Go",
    ".rs": "Rust",
    ".sol": "Solidity",
}

SUPPORTED_DOC_EXTENSIONS = {
    ".md": "Markdown",
    ".txt": "Text",
    ".rst": "reStructuredText",
}

def is_supported_file(file_path: str) -> bool:
    """Check if a file is supported for ingestion."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_CODE_EXTENSIONS or ext in SUPPORTED_DOC_EXTENSIONS

def get_file_language(file_path: str) -> Optional[str]:
    """Get the programming language or document type of a file."""
    ext = os.path.splitext(file_path)[1].lower()
    return SUPPORTED_CODE_EXTENSIONS.get(ext) or SUPPORTED_DOC_EXTENSIONS.get(ext)

def find_files(directory: str, extensions: set[str]) -> List[Path]:
    """Recursively find all files with given extensions in a directory."""
    logger.debug(f"Scanning directory: {directory} for extensions: {extensions}")
    files = []
    for ext in extensions:
        found = list(Path(directory).rglob(f"*{ext}"))
        logger.debug(f"Found {len(found)} files with extension {ext}")
        files.extend(found)
    return files

def ingest_file(file_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
    """Ingest a single file and return chunks with metadata."""
    try:
        logger.debug(f"Loading file: {file_path}")
        loader = TextLoader(str(file_path))
        documents = loader.load()
        
        logger.debug(f"Splitting content with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        
        chunks = splitter.split_documents(documents)
        
        # Add metadata
        language = get_file_language(str(file_path))
        for chunk in chunks:
            chunk.metadata.update({
                "file_path": str(file_path),
                "language": language,
            })
        
        logger.debug(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        logger.error(f"Error processing {file_path}", exc_info=e)
        return []

def ingest_directory(directory: Path, is_code: bool = True) -> List[dict]:
    """Ingest all supported files in a directory."""
    extensions = set(SUPPORTED_CODE_EXTENSIONS.keys()) if is_code else set(SUPPORTED_DOC_EXTENSIONS.keys())
    logger.info(f"Starting {'code' if is_code else 'documentation'} ingestion from {directory}")
    
    files = find_files(directory, extensions)
    if not files:
        logger.warning(f"No {'code' if is_code else 'documentation'} files found in {directory}")
        return []
    
    all_chunks = []
    with logger.progress(f"Processing {len(files)} files...") as progress:
        for file_path in files:
            logger.info(f"Processing {file_path}")
            chunks = ingest_file(file_path)
            all_chunks.extend(chunks)
    
    logger.success(f"Processed {len(files)} files, created {len(all_chunks)} chunks")
    return all_chunks 