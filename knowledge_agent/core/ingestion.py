"""Module for ingesting code and documentation files into the vector store."""

import logging
from pathlib import Path
from typing import List, Optional, Set, Union

from knowledge_agent.core.code_parser import CodeEntity, ParserFactory as CodeParserFactory
from knowledge_agent.core.doc_parser import DocSection, ParserFactory as DocParserFactory
from knowledge_agent.core.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_CODE_EXTENSIONS = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript (React)',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript (React)',
    '.cpp': 'C++',
    '.c': 'C',
    '.html': 'HTML',
    '.css': 'CSS',
    '.go': 'Go',
    '.rs': 'Rust',
    '.sol': 'Solidity',
}

SUPPORTED_DOC_EXTENSIONS = {
    '.md': 'Markdown',
    '.txt': 'Text',
    '.rst': 'reStructuredText',
    '.pdf': 'PDF',
    '.tex': 'LaTeX',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.json': 'JSON',
    '.toml': 'TOML',
}

def is_supported_file(file_path: Path) -> bool:
    """Check if a file is supported for ingestion."""
    return (file_path.suffix.lower() in SUPPORTED_CODE_EXTENSIONS or
            file_path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS)

def get_file_language(file_path: Path) -> Optional[str]:
    """Get the language or document type of a file."""
    ext = file_path.suffix.lower()
    return SUPPORTED_CODE_EXTENSIONS.get(ext) or SUPPORTED_DOC_EXTENSIONS.get(ext)

def ingest_file(file_path: Path, vector_store: VectorStore) -> int:
    """Ingest a single file into the vector store."""
    if not is_supported_file(file_path):
        logger.warning(f"Unsupported file type: {file_path}")
        return 0
    
    try:
        # Try code parser first
        parser = CodeParserFactory.get_parser(file_path)
        if parser:
            entities = parser.parse_file(file_path)
            return _ingest_code_entities(entities, vector_store)
        
        # Try doc parser if code parser not found
        parser = DocParserFactory.get_parser(file_path)
        if parser:
            sections = parser.parse_file(file_path)
            return _ingest_doc_sections(sections, vector_store)
        
        logger.warning(f"No parser found for file: {file_path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to ingest file {file_path}: {str(e)}")
        return 0

def _ingest_code_entities(entities: List[CodeEntity], vector_store: VectorStore) -> int:
    """Ingest code entities into the vector store."""
    chunks_created = 0
    for entity in entities:
        try:
            # Create a chunk with code entity information
            chunk = {
                'content': entity.code,
                'metadata': {
                    'type': 'code',
                    'entity_type': entity.type,
                    'name': entity.name,
                    'language': entity.metadata.get('language', 'unknown'),
                    'file_path': entity.metadata.get('path', ''),
                    'start_line': entity.start_line,
                    'end_line': entity.end_line,
                    'parent': entity.parent or '',
                    'dependencies': list(entity.dependencies),
                }
            }
            
            # Add docstring as a separate chunk if it exists
            if entity.docstring:
                doc_chunk = {
                    'content': entity.docstring,
                    'metadata': {
                        'type': 'docstring',
                        'entity_name': entity.name,
                        'entity_type': entity.type,
                        'language': entity.metadata.get('language', 'unknown'),
                        'file_path': entity.metadata.get('path', ''),
                        'start_line': entity.start_line,
                    }
                }
                vector_store.add_chunk(doc_chunk)
                chunks_created += 1
            
            # Add the code chunk
            vector_store.add_chunk(chunk)
            chunks_created += 1
        except Exception as e:
            logger.error(f"Failed to ingest code entity {entity.name}: {str(e)}")
    
    return chunks_created

def _ingest_doc_sections(sections: List[DocSection], vector_store: VectorStore) -> int:
    """Ingest documentation sections into the vector store."""
    chunks_created = 0
    for section in sections:
        try:
            # Create a chunk with documentation section information
            chunk = {
                'content': section.content,
                'metadata': {
                    'type': 'documentation',
                    'title': section.title,
                    'file_path': str(section.source_file),
                    'line_number': section.line_number,
                    **(section.metadata or {})
                }
            }
            
            # Add the documentation chunk
            vector_store.add_chunk(chunk)
            chunks_created += 1
        except Exception as e:
            logger.error(f"Failed to ingest doc section {section.title}: {str(e)}")
    
    return chunks_created

def ingest_directory(directory: Path, vector_store: VectorStore,
                    exclude_patterns: Optional[Set[str]] = None) -> int:
    """Recursively ingest all supported files in a directory."""
    if exclude_patterns is None:
        exclude_patterns = set()
    
    total_chunks = 0
    try:
        for file_path in directory.rglob('*'):
            # Skip directories and excluded patterns
            if not file_path.is_file() or any(pattern in str(file_path) for pattern in exclude_patterns):
                continue
            
            # Ingest supported files
            if is_supported_file(file_path):
                chunks = ingest_file(file_path, vector_store)
                total_chunks += chunks
                logger.info(f"Ingested {file_path}: {chunks} chunks created")
    except Exception as e:
        logger.error(f"Failed to ingest directory {directory}: {str(e)}")
    
    return total_chunks 