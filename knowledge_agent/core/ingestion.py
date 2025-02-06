"""Code and documentation ingestion functionality."""
import os
from pathlib import Path
from typing import List, Optional, Union, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from knowledge_agent.core.logging import logger
from knowledge_agent.core.code_parser import parse_codebase, CodeEntity
from knowledge_agent.core.version_control import VersionManager, VersionMetadata

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

def code_entity_to_document(
    entity: CodeEntity,
    version_meta: Optional[VersionMetadata] = None
) -> Document:
    """Convert a CodeEntity to a LangChain Document."""
    # Combine code and docstring for better context
    content = f"{entity.docstring}\n\n{entity.code}" if entity.docstring else entity.code
    
    # Create metadata
    metadata = {
        **entity.metadata,
        'type': entity.type,
        'name': entity.name,
        'parent': entity.parent or '',
        'dependencies': list(entity.dependencies),
        'start_line': entity.start_line,
        'end_line': entity.end_line,
    }
    
    # Add version metadata if available
    if version_meta:
        metadata.update({
            'commit_hash': version_meta.commit_hash,
            'branch': version_meta.branch,
            'commit_date': version_meta.commit_date.isoformat(),
            'author': version_meta.author,
            'commit_message': version_meta.message,
        })
    
    return Document(page_content=content, metadata=metadata)

def ingest_code_file(
    file_path: Path,
    version_meta: Optional[VersionMetadata] = None
) -> List[Document]:
    """Ingest a code file using AST parsing."""
    try:
        logger.debug(f"Parsing code file: {file_path}")
        entities = parse_codebase(file_path)
        documents = [code_entity_to_document(entity, version_meta) for entity in entities]
        logger.debug(f"Created {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error processing code file {file_path}", exc_info=e)
        return []

def ingest_doc_file(
    file_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    version_meta: Optional[VersionMetadata] = None
) -> List[Document]:
    """Ingest a documentation file using text chunking."""
    try:
        logger.debug(f"Loading documentation file: {file_path}")
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
            metadata = {
                "file_path": str(file_path),
                "language": language,
                "type": "documentation",
            }
            
            # Add version metadata if available
            if version_meta:
                metadata.update({
                    'commit_hash': version_meta.commit_hash,
                    'branch': version_meta.branch,
                    'commit_date': version_meta.commit_date.isoformat(),
                    'author': version_meta.author,
                    'commit_message': version_meta.message,
                })
            
            chunk.metadata.update(metadata)
        
        logger.debug(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        logger.error(f"Error processing documentation file {file_path}", exc_info=e)
        return []

def ingest_directory(
    directory: Path,
    is_code: bool = True,
    ref: Optional[str] = None
) -> List[Document]:
    """Ingest all supported files in a directory.
    
    Args:
        directory: Directory to ingest
        is_code: Whether to ingest code files (True) or documentation files (False)
        ref: Optional Git reference (commit, branch, tag) to use for versioning
        
    Returns:
        List of document chunks
    """
    extensions = set(SUPPORTED_CODE_EXTENSIONS.keys()) if is_code else set(SUPPORTED_DOC_EXTENSIONS.keys())
    logger.info(f"Starting {'code' if is_code else 'documentation'} ingestion from {directory}")
    
    # Initialize version manager if ref is provided
    version_meta = None
    if ref:
        try:
            version_manager = VersionManager(directory)
            version_meta = version_manager.get_version_metadata(ref)
            logger.info(f"Using version: {version_meta.commit_hash} ({version_meta.branch})")
        except Exception as e:
            logger.error(f"Failed to initialize version manager: {str(e)}")
            return []
    
    files = find_files(directory, extensions)
    if not files:
        logger.warning(f"No {'code' if is_code else 'documentation'} files found in {directory}")
        return []
    
    all_chunks = []
    with logger.progress(f"Processing {len(files)} files...") as progress:
        for file_path in files:
            logger.info(f"Processing {file_path}")
            if is_code:
                chunks = ingest_code_file(file_path, version_meta)
            else:
                chunks = ingest_doc_file(file_path, version_meta=version_meta)
            all_chunks.extend(chunks)
    
    logger.success(f"Processed {len(files)} files, created {len(all_chunks)} chunks")
    return all_chunks 